"""`AtomTrunk` on one molecule's atoms in the geometry a conformer state places them in.

    state x  ->  positions (energies/conformer_data.py::states_to_positions)
             ->  every ordered pair of the molecule's atoms within `cutoff`
             ->  AtomTrunk  ->  per-atom states h, per-atom energies, their sum per row

The model sees an atom's element, how many bonds it has and how many of them are to hydrogen,
and for each pair its distance and how many bonds apart the two atoms are. All of it is read
off the condition graph (`MoleculeTopology`): no SMILES, no encoder. Nothing it sees changes
under a rotation, a translation or a mirror image of the positions, so its energy is the same
for a conformation and its mirror image, and for a centre and the same centre turned inside
out. The force-field energy has that symmetry; the stereo lock (energies/stereo_lock.py),
which holds each centre to its labelled handedness, does not, and this model cannot
represent it.

The gradient with respect to the state is not an output: differentiate the energy through the
build, as `energy_and_gradient` does.
"""
import torch
from torch import nn

from models.atom_trunk import AtomTrunk

#: classes of atom pair by bonds apart: 1, 2, 3, 4, and five or more
SPD_KINDS = 5
#: bonds per atom and bonds to hydrogen per atom are one-hot up to this count
MAX_DEGREE = 4
NODE_FEATURES = 2 * (MAX_DEGREE + 1)
#: pairs closer than this present as this distance: the radial basis grows as 1 / distance
MIN_DISTANCE = 0.4


class MoleculeTopology:
    """What the trunk reads of each condition that no state changes, padded to the widest molecule.

    Built once from the conditions batch (atoms in placement order, as every batch drawn from
    it keeps them): `kind` [C, A, A] the pair class above, `node_feat` [C, A, NODE_FEATURES],
    `n` [C] atoms per condition.
    """

    def __init__(self, cond, device, chunk: int = 4096):
        from energies.conformer_data import batch_tree

        ptr = cond.ptr.cpu()
        n = ptr[1:] - ptr[:-1]
        C, A = int(n.numel()), int(n.max())
        graph = cond.batch.cpu()
        local = torch.arange(int(ptr[-1])) - ptr[graph]
        bonds = batch_tree(cond).graph_bond_index.cpu()
        g, i, j = graph[bonds[:, 0]], local[bonds[:, 0]], local[bonds[:, 1]]
        if not torch.equal(g, graph[bonds[:, 1]]):
            raise ValueError('a bond joins atoms of two conditions')
        adj = torch.zeros(C, A, A, dtype=torch.bool)
        adj[g, i, j] = True
        adj[g, j, i] = True
        z = torch.zeros(C, A, dtype=torch.long)
        z[graph, local] = cond.z.cpu().long()

        kind = torch.full((C, A, A), SPD_KINDS - 1, dtype=torch.int8)
        for lo in range(0, C, chunk):
            a = adj[lo:lo + chunk].float()
            seen = torch.eye(A, dtype=torch.bool).expand_as(a).clone()
            front = seen.clone()
            for s in range(SPD_KINDS - 1):
                front = (torch.bmm(front.float(), a) > 0) & ~seen
                kind[lo:lo + chunk][front] = s
                seen |= front
        degree = adj.sum(-1).clamp(max=MAX_DEGREE)
        to_h = (adj & (z == 1)[:, None, :]).sum(-1).clamp(max=MAX_DEGREE)
        one_hot = nn.functional.one_hot
        self.node_feat = torch.cat((one_hot(degree, MAX_DEGREE + 1), one_hot(to_h, MAX_DEGREE + 1)),
                                   dim=-1).float().to(device)
        self.kind = kind.to(device)
        self.n = n.to(device)
        self.A = A
        self._eye = torch.eye(A, dtype=torch.bool, device=device)

    def batch(self, slots):
        """The pairs and per-atom inputs of a batch whose row b is condition `slots[b]`.

        Returns (edge_index [2, E] source then target among the batch's atoms, kind [E],
        node_feat [N, NODE_FEATURES], offsets [B] of each row's first atom).
        """
        n = self.n[slots]
        amask = torch.arange(self.A, device=slots.device)[None] < n[:, None]
        offs = torch.cumsum(n, 0) - n
        pair = amask[:, :, None] & amask[:, None, :] & ~self._eye
        b, i, j = pair.nonzero(as_tuple=True)
        return (torch.stack((offs[b] + j, offs[b] + i)), self.kind[slots][b, i, j].long(),
                self.node_feat[slots][amask], offs)


class ConformerTrunk(nn.Module):
    """Per-atom energy model of a molecule in the geometry of a conformer state.

    Parameters are `AtomTrunk`'s, and `energy_scale`: the network's outputs times this are the
    energies returned, so a target in kcal/mol can be fitted by a network whose outputs are of
    order one. `cutoff` should cover the molecules: a pair beyond it has no edge.
    """

    def __init__(self, node_dim: int = 128, message_dim: int = 64, num_convs: int = 3, num_radial: int = 48,
                 cutoff: float = 10.0, fcs_per_gc: int = 1, type_dim: int = 16, activation: str = 'gelu',
                 energy_scale: float = 1.0):
        super().__init__()
        trunk = dict(node_dim=node_dim, message_dim=message_dim, num_convs=num_convs, num_radial=num_radial,
                     cutoff=cutoff, fcs_per_gc=fcs_per_gc, type_dim=type_dim, activation=activation)
        #: the constructor arguments, as a checkpoint stores them
        self.args = dict(trunk, energy_scale=float(energy_scale))
        self.energy_scale = float(energy_scale)
        self.trunk = AtomTrunk(**trunk, intra_kinds=SPD_KINDS, node_features=NODE_FEATURES, tail=False)

    def forward(self, pos, sub, slots, topology):
        """pos [N, 3] the batch's atoms; sub the condition batch they belong to, row b being
        condition `slots[b]` of the set `topology` was built from.

        Returns `AtomTrunk.forward`'s dict: `h` [N, node_dim], `e_atom` [N], `energy` [B].
        """
        edge_index, kind, node_feat, offs = topology.batch(slots)
        if not torch.equal(offs, sub.ptr[:-1]):
            raise ValueError('the batch is not the conditions `slots` names, in that order')
        d = pos[edge_index[0]] - pos[edge_index[1]]
        dist = (d.square().sum(1) + 1e-12).sqrt()
        keep = dist.detach() <= self.trunk.cutoff
        out = self.trunk(sub.z.long(), sub.batch, edge_index[:, keep], dist[keep].clamp_min(MIN_DISTANCE),
                         intra_kind=kind[keep], node_feat=node_feat.to(pos.dtype), num_graphs=int(slots.numel()))
        return {'h': out['h'], 'e_atom': out['e_atom'] * self.energy_scale,
                'energy': out['energy'] * self.energy_scale}

    def energy_and_gradient(self, x, sub, slots, topology, create_graph: bool = False):
        """(energy [B], its gradient with respect to the state [B, K], per-atom states [N, node_dim]).

        `x` [B, K] is the state; it is rebuilt into positions here, so the gradient carries the
        chart: its entry for a coordinate the molecule does not have is zero.
        """
        from energies.conformer_data import states_to_positions

        with torch.enable_grad():
            xg = x if x.requires_grad else x.detach().requires_grad_(True)
            out = self(states_to_positions(sub, xg), sub, slots, topology)
            grad, = torch.autograd.grad(out['energy'].sum(), xg, create_graph=create_graph)
        return out['energy'], grad, out['h']
