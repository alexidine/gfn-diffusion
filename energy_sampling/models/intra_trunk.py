"""The intra trunk: one molecule's atoms as a typed point cloud, every atom reading every other.

    elements [B, A], positions [B, A, 3], mask [B, A]      padded to the widest molecule
        -> every ordered pair of a row's real atoms, with its distance
        -> AtomTrunk (no periodic neighbours, no density tail)
        -> per-atom states h, per-atom energies, their sum per molecule

It sees elements and distances and nothing else: no bond list, no SMILES. A pair beyond
`cutoff` keeps its edge, presenting at the cutoff, where the radial basis is zero, so the
two atoms still exchange their states. Nothing it sees changes under a rotation, a
translation or a mirror image, so neither does its energy.

ENERGY UNITS. The network's own per-atom output is in units of `scale` above a per-element
reference `e0`, both fixed when the trunk is fitted and stored with it (the convention of
machine-learned potentials: a shift per element, one scale from the force RMS):

    e_atom [kcal/mol] = scale * network(atom) + e0[element]

The force is not an output: differentiate the energy with respect to the positions
(`energy_and_force`).
"""
import torch
from torch import nn

from .atom_trunk import AtomTrunk

#: pairs closer than this present as this distance: the radial basis grows as 1 / distance
MIN_DISTANCE = 0.4


def molecule_pairs(pos: torch.Tensor, mask: torch.Tensor, cutoff: float):
    """Every ordered pair of each row's real atoms, indexed among the batch's real atoms.

    pos [B, A, 3], mask [B, A] (True on real atoms). Returns (edge_index [2, E] source then
    target, dist [E] held to [MIN_DISTANCE, cutoff], node_graph [N] row of each real atom).
    Differentiable in `pos`.
    """
    b_of, a_of = mask.nonzero(as_tuple=True)
    flat = torch.full(mask.shape, -1, dtype=torch.long, device=mask.device)
    flat[b_of, a_of] = torch.arange(b_of.numel(), device=mask.device)
    ok = mask[:, :, None] & mask[:, None, :]
    ok &= ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    b, i, j = ok.nonzero(as_tuple=True)
    dist = (pos[b, i] - pos[b, j]).square().sum(-1).clamp(min=MIN_DISTANCE ** 2).sqrt().clamp(max=cutoff)
    return torch.stack((flat[b, j], flat[b, i])), dist, b_of


class IntraTrunk(nn.Module):
    """`AtomTrunk` over all pairs of one molecule, with the energy units it was fitted in.

    Parameters are `AtomTrunk`'s; `cutoff` is the range of the radial basis, not a limit on
    which atoms see each other.
    """

    def __init__(self, node_dim: int = 128, message_dim: int = 64, num_convs: int = 3, num_radial: int = 48,
                 cutoff: float = 10.0, fcs_per_gc: int = 1, type_dim: int = 16, activation: str = 'gelu'):
        super().__init__()
        self.trunk = AtomTrunk(node_dim=node_dim, message_dim=message_dim, num_convs=num_convs,
                               num_radial=num_radial, cutoff=cutoff, fcs_per_gc=fcs_per_gc, type_dim=type_dim,
                               activation=activation, tail=False)
        self.cutoff = float(cutoff)
        self.node_dim = int(node_dim)
        self.register_buffer('e0', torch.zeros(101))
        self.register_buffer('scale', torch.ones(()))

    def set_units(self, e0: torch.Tensor, scale: float):
        """Install the per-element reference [101] (kcal/mol) and the energy scale (kcal/mol)."""
        self.e0.copy_(e0.to(self.e0))
        self.scale.fill_(float(scale))

    def forward(self, z: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor):
        """z [B, A] atomic numbers, pos [B, A, 3] Angstrom, mask [B, A].

        Returns a dict over the batch's real atoms, in row order: `h` [N, node_dim], `e_atom` [N]
        and `energy` [B] in kcal/mol, `e_model` [N] the network's own output, `node_graph` [N].
        """
        edge_index, dist, node_graph = molecule_pairs(pos, mask, self.cutoff)
        zr = z[mask].long()
        out = self.trunk(zr, node_graph, edge_index, dist, num_graphs=z.shape[0])
        e_atom = self.scale * out['e_atom'] + self.e0[zr]
        energy = torch.zeros(z.shape[0], dtype=e_atom.dtype, device=e_atom.device).index_add_(0, node_graph, e_atom)
        return {'h': out['h'], 'e_atom': e_atom, 'energy': energy, 'e_model': out['e_atom'], 'node_graph': node_graph}

    def energy_and_force(self, z: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, create_graph: bool = False):
        """(energy [B] kcal/mol, force [B, A, 3] kcal/mol/Angstrom, zero on padding, the forward dict)."""
        with torch.enable_grad():
            x = pos if pos.requires_grad else pos.detach().requires_grad_(True)
            out = self(z, x, mask)
            grad, = torch.autograd.grad(out['energy'].sum(), x, create_graph=create_graph)
        return out['energy'], -grad, out


def load_intra_trunk(checkpoint: str, device='cpu') -> IntraTrunk:
    """The trunk a `pretrain_intra_trunk.py` checkpoint holds, with its units, frozen and in eval mode."""
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = IntraTrunk(**ck['trunk_args'])
    model.load_state_dict(ck['model'])
    return model.to(device).requires_grad_(False).eval()
