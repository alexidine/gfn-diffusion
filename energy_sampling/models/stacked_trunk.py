"""The crystal energy trunk stacked on the intra trunk.

    molecule (elements, positions)  -> IntraTrunk, frozen  -> per-atom states e_m      once per molecule
    crystal at a state: e_m on the molecule's atoms, intermolecular pairs and their distances
                                    -> AtomTrunk stages over those pairs -> h, per-atom energy, energy

The stages here are `AtomTrunk`'s with the intra trunk's states as extra per-atom inputs beside
the element, and with no intramolecular edges of their own: what one atom knows of the rest
of its molecule arrives in e_m. An image atom is still the node it is a copy of, and e_m is a
property of the atom in its molecule (it does not change with the cell or the pose), so the
sharing stays exact.

The intra trunk's parameters are part of this module's state dict and are never trained
here. `feature_scale` divides e_m before it is read, set once from the molecules the trunk
is first fitted on (`calibrate`) and stored.
"""
import torch
from torch import nn

from .atom_trunk import AtomTrunk
from .intra_trunk import IntraTrunk


class StackedTrunk(nn.Module):
    """Intermolecular stages reading a frozen intra trunk's per-atom states.

    Parameters
    ----------
    intra_args : `IntraTrunk`'s constructor arguments (a checkpoint's `trunk_args`).
    node_dim, message_dim, num_convs, cutoff, folded : `AtomTrunk`'s, for the intermolecular stages.
    """

    def __init__(self, intra_args: dict, node_dim: int = 128, message_dim: int = 64, num_convs: int = 3,
                 cutoff: float = 5.0, folded: bool = True):
        super().__init__()
        self.intra = IntraTrunk(**intra_args).requires_grad_(False).eval()
        self.inter = AtomTrunk(node_dim=node_dim, message_dim=message_dim, num_convs=num_convs, cutoff=cutoff,
                               folded=folded, node_features=self.intra.node_dim)
        self.cutoff = float(cutoff)
        self.register_buffer('feature_scale', torch.ones(()))

    def train(self, mode: bool = True):
        super().train(mode)
        self.intra.eval()                    # the intra trunk is a fixed function here
        return self

    def molecule_states(self, z: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """e_m [N, D] over the batch's real atoms in row order, from padded z [B, A], pos [B, A, 3], mask [B, A].
        No gradient: the intra trunk is frozen and a rigid molecule's own geometry is not a function of the state."""
        with torch.no_grad():
            return self.intra(z, pos, mask)['h'] / self.feature_scale

    def calibrate(self, z: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor):
        """Set `feature_scale` to the root mean square of the intra states over these molecules."""
        with torch.no_grad():
            self.feature_scale.fill_(1.0)
            self.feature_scale.copy_(self.molecule_states(z, pos, mask).square().mean().sqrt().clamp(min=1e-6))

    def forward(self, z, node_graph, inter_ref, inter_img, inter_dist, density, e_m):
        """`AtomTrunk.forward` without intramolecular edges and with `e_m` [N, D] (`molecule_states`).
        Returns its dict: `h` [N, node_dim], `e_atom` [N], `energy` [B]."""
        none = torch.zeros(2, 0, dtype=torch.long, device=z.device)
        return self.inter(z, node_graph, none, inter_dist[:0], inter_ref, inter_img, inter_dist, density,
                          node_feat=e_m)
