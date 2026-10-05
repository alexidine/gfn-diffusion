"""Per-atom energy model on an instantiated Z' = 1 crystal: the trunk a policy can read atoms through.

The graph has one node per atom of ONE molecule. An intermolecular edge joins a reference
atom to an atom of a periodic image, and the image atom is represented by the node it is a
copy of, so every image shares its parent's state and any number of message-passing stages
describes the infinite crystal. Node states are invariant (element, distances), which is
what makes that sharing exact.

    e_i   = local(h_i) + tail(element_i, density)        per atom, in kT
    E     = sum_i e_i                                     per crystal

`local` sees pairs inside `cutoff` only; `tail` is there for whatever of the target lies
beyond it, which for a pair potential is close to a function of the cell's number density.
Forces are not an output: differentiate E with respect to whatever the distances and the
density were built from (the crystal latent, through
`mxtaltools.crystal_building.image_pairs`).

Layers are MXtalTools' own (`MConv`, `scalarMLP`, `EmbeddingBlock`, `BesselBasisLayer`),
with no normalisation layers, so an output depends on its own crystal only and never on
the rest of the batch. `FoldedMConv` is `MConv` with the same parameters and output and
less per-edge work; it is the default here and nothing else in either repository uses it.
"""
import torch
from torch import nn

from mxtaltools.models.modules.basis_functions import BesselBasisLayer
from mxtaltools.models.modules.components import EmbeddingBlock, scalarMLP
from mxtaltools.models.modules.graph_convolution import MConv

#: multiplies molecules-per-cubic-Angstrom so that an organic crystal's density reads near 1
DENSITY_SCALE = 200.0


def crystal_density(tables, T_fc):
    """Molecules per cell volume, scaled by DENSITY_SCALE; [B], differentiable in the cell matrix."""
    return DENSITY_SCALE * tables.kmask.sum(1).to(T_fc.dtype) / torch.linalg.det(T_fc).abs()


def intramolecular_edges(tables, cutoff: float):
    """All ordered atom pairs of each molecule within `cutoff`, as indices among the batch's real atoms.

    Returns (edge_index [2, E] source then target, dist [E]). The molecule is rigid in this
    chart, so these do not depend on the cell or pose.
    """
    p, am = tables.p, tables.amask
    d = torch.cdist(p, p)
    ok = am[:, :, None] & am[:, None, :] & (d <= cutoff)
    ok &= ~torch.eye(p.shape[1], dtype=torch.bool, device=p.device)[None]
    b, i, j = ok.nonzero(as_tuple=True)
    return torch.stack((tables.ptr[b] + j, tables.ptr[b] + i)), d[b, i, j]


class FoldedMConv(MConv):
    """`MConv` with the same parameters and the same output, computed with less per-edge work.

    `MConv.message` applies a bias-free linear map to each endpoint's node state AFTER
    gathering it per edge, another to the edge attributes, and a third to their
    concatenation. Everything up to the normalisation is linear, so it equals

        pre = A x[target] + B x[source] + C e,
        A = W[:, :m] S,   B = W[:, m:2m] T,   C = W[:, 2m:] U

    with W, S, T, U the weights of `generate_message`, `source_node2message`,
    `tgt_node2message` and `edge2message`. A and B act on NODES before the gather, which
    is where the saving is. The fold is rebuilt from the module's own parameters on every
    call, so the state dict is `MConv`'s and gradients reach the original weights.
    Normalisation, activation, the softmax aggregation, the projection back to node width
    and the residual are `MConv`'s own, called in the same order.

    Not for `SparseTensor` adjacency: `edge_index` must be a [2, E] index tensor.
    """

    def forward(self, x, edge_index, edge_attr):
        m = self.message_dim
        w = self.generate_message.weight
        a = w[:, :m] @ self.source_node2message.weight        # `message` gives this map the TARGET node, x_i
        b = w[:, m:2 * m] @ self.tgt_node2message.weight       # and this one the SOURCE node, x_j
        c = w[:, 2 * m:] @ self.edge2message.weight
        src, tgt = edge_index[0], edge_index[1]
        pre = (x @ a.T)[tgt] + (x @ b.T)[src] + edge_attr @ c.T
        out = self.aggr_module(self.activation(self.norm(pre)), tgt, dim_size=x.size(0), dim=0)
        return x + self.message2node(out)


class AtomTrunk(nn.Module):
    """Invariant message passing over one molecule's atoms and their periodic neighbours.

    Parameters
    ----------
    node_dim, message_dim : hidden widths of the node states and of the messages.
    num_convs : number of message-passing stages.
    num_radial : radial basis functions per edge.
    cutoff : pair cutoff in Angstrom; edges handed to `forward` must lie within it.
    fcs_per_gc : blocks in the `scalarMLP` after each stage.
    type_dim : width of the element embedding.
    folded : use `FoldedMConv` (same parameters and output as `MConv`, less per-edge work).
    """

    def __init__(self, node_dim: int = 128, message_dim: int = 64, num_convs: int = 3, num_radial: int = 32,
                 cutoff: float = 5.0, fcs_per_gc: int = 1, type_dim: int = 16, activation: str = 'gelu',
                 folded: bool = True):
        super().__init__()
        conv_cls = FoldedMConv if folded else MConv
        self.cutoff = float(cutoff)
        self.node_dim = node_dim
        self.rbf = BesselBasisLayer(num_radial, cutoff)
        self.embed = EmbeddingBlock(node_dim, 100, 1, type_dim)
        self.pre = scalarMLP(layers=fcs_per_gc, filters=node_dim, input_dim=node_dim, output_dim=node_dim,
                             activation=activation)
        # edge attribute: radial basis plus a two-way flag, intramolecular or intermolecular
        self.convs = nn.ModuleList([
            conv_cls(message_dim=message_dim, node_dim=node_dim, edge_embedding_dim=num_radial + 2,
                     norm=None, activation_fn=activation) for _ in range(num_convs)])
        self.fcs = nn.ModuleList([
            scalarMLP(layers=fcs_per_gc, filters=node_dim, input_dim=node_dim, output_dim=node_dim,
                      activation=activation) for _ in range(num_convs)])
        self.local_head = scalarMLP(layers=2, filters=node_dim, input_dim=node_dim, output_dim=1,
                                    activation=activation)
        self.tail_head = scalarMLP(layers=2, filters=node_dim // 2, input_dim=node_dim + 1, output_dim=1,
                                   activation=activation)

    def forward(self, z, node_graph, intra_index, intra_dist, inter_ref, inter_img, inter_dist, density):
        """
        z [N] atomic numbers of the batch's real atoms; node_graph [N] their crystal index.
        intra_index [2, Ei] (source, target) and intra_dist [Ei]: pairs within one molecule.
        inter_ref [Ee], inter_img [Ee], inter_dist [Ee]: reference atom, the atom its image
        neighbour is a copy of, and their distance; all within `cutoff`.
        density [B]: `crystal_density`.

        Returns a dict: `h` [N, node_dim] node states, `e_atom` [N] per-atom energies and
        `energy` [B] their per-crystal sums.
        """
        num_graphs = density.shape[0]
        h0 = self.pre(self.embed(z), batch=node_graph)
        edge_index = torch.cat((intra_index, torch.stack((inter_img, inter_ref))), dim=1)
        kind = torch.zeros(edge_index.shape[1], 2, dtype=intra_dist.dtype, device=z.device)
        kind[:intra_dist.shape[0], 0] = 1.0
        kind[intra_dist.shape[0]:, 1] = 1.0
        edge_attr = torch.cat((self.rbf(torch.cat((intra_dist, inter_dist))), kind), dim=1)

        h = h0
        for conv, fc in zip(self.convs, self.fcs):
            h = fc(conv(h, edge_index, edge_attr), batch=node_graph)

        e_local = self.local_head(h, batch=node_graph).squeeze(-1)
        e_tail = self.tail_head(torch.cat((h0, density[node_graph, None]), dim=1), batch=node_graph).squeeze(-1)
        e_atom = e_local + e_tail
        energy = torch.zeros(num_graphs, dtype=e_atom.dtype, device=z.device).index_add_(0, node_graph, e_atom)
        return {'h': h, 'e_atom': e_atom, 'energy': energy}


def pooled_features(h, node_graph, num_graphs: int):
    """[B, 2 * node_dim]: the mean and the sum of the node states of each crystal."""
    total = torch.zeros(num_graphs, h.shape[1], dtype=h.dtype, device=h.device).index_add_(0, node_graph, h)
    count = torch.zeros(num_graphs, dtype=h.dtype, device=h.device).index_add_(
        0, node_graph, torch.ones_like(node_graph, dtype=h.dtype))
    return torch.cat((total / count[:, None], total), dim=1)
