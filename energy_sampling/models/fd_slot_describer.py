"""Typed FD rows over several space groups: models/fd_row_describer.py::FDRowDescriber per group, placed in the slots
of a mixed-group FD state (energies/fd_coordinates.py::FDSlotLayout).

A slot means a different coordinate in different groups (slot 0 is log_c on a triclinic or monoclinic chart, log_a
on an orthorhombic one), so the conditioner's per-slot type embedding cannot say what a row is. The per-row features
carry it instead: each row's block is its group's FDRowDescriber features, then a one-hot of the row's coordinate
type (FD_TYPES) and a pad flag (1 on the slots the row's group leaves empty, whose other features are 0). The per-slot
types handed to the conditioner (``types``) name each slot by the widest group of its block, so slots stay distinct.

Each row's group is set per call, in the rows' order (``set_row_groups``), as the flow's pad mask is
(spline_flow.py::_ContextFlow.set_row_live): a row's chart is its own state's, whatever condition it is read under.
"""
import torch

from models.fd_row_describer import FD_TYPES, FDRowDescriber

F64 = torch.float64


class SlotRowDescriber:
    """FDRowDescriber's interface (dim, types, k, prefix, __call__, stepper) over an FDSlotLayout's slots."""

    def __init__(self, layout, prefix: bool = False):
        self.layout = layout
        self.prefix = bool(prefix)
        self.per_group = {g: FDRowDescriber(g, prefix=self.prefix) for g in layout.groups}
        self.dim = int(layout.dim)
        self.k_group = next(iter(self.per_group.values())).k
        self.n_types = len(FD_TYPES)
        self.k = self.k_group + self.n_types + 1
        # per slot: the type of the widest group of its block (distinct within a block, and the blocks' names differ)
        names = ['pad'] * self.dim
        for k, (start, width) in enumerate(zip(layout.block_start, layout.block_width)):
            for g in layout.groups:
                if int(len(self._block_slots(g, k))) == width:
                    for i in range(start, start + width):
                        names[i] = layout.group_names[g][i]
                    break
        self.types = [FD_TYPES.index(n) for n in names]
        # per group: [dim, n_types + 1] the type one-hot and pad flag of each slot
        tags = {}
        for g in layout.groups:
            t = torch.zeros(self.dim, self.n_types + 1, dtype=F64)
            for i, n in enumerate(layout.group_names[g]):
                if n == 'pad':
                    t[i, -1] = 1.0
                else:
                    t[i, FD_TYPES.index(n)] = 1.0
            tags[g] = t
        self.tags = tags
        self.row_sg = None

    def _block_slots(self, g, k):
        s = self.layout.slots[g]
        start, width = self.layout.block_start[k], self.layout.block_width[k]
        return s[(s >= start) & (s < start + width)]

    def set_row_groups(self, sg):
        """Each row's space group [n] for the next calls, in the rows' order."""
        self.row_sg = torch.as_tensor(sg).long().reshape(-1).cpu()
        self.layout.group_index(self.row_sg)   # raises on a group outside the layout

    def _groups(self, b):
        if self.row_sg is None or self.row_sg.numel() != b:
            raise RuntimeError(f"slot describer: {None if self.row_sg is None else self.row_sg.numel()} row groups "
                               f"for {b} rows (set_row_groups)")
        return self.row_sg

    def __call__(self, s, radius, mol=None):
        """[B, dim, k] of states s [B, dim] at molecule radii [B] and, with prefix features, molecule inputs
        [B, len(MOL_INPUTS)] (models/fd_row_describer.py::mol_inputs)."""
        st = self.stepper(radius, mol, s.shape[0], s.device)
        return torch.stack([st.row(s, i) for i in range(self.dim)], 1).to(dtype=s.dtype)

    def stepper(self, radius, mol=None, b=None, device=None):
        return SlotStepper(self, radius, mol, b, device)


class SlotStepper:
    """Slot i's features [B, k] from slots < i: each group's RowStepper over its own rows, asked for its chart row in
    slot i (slots and chart rows increase together within a group)."""

    def __init__(self, desc: SlotRowDescriber, radius, mol, b, device):
        r = torch.as_tensor(radius)
        dev = r.device if device is None else device
        r = r.to(device=dev, dtype=F64).reshape(-1)
        self.b = r.numel() if b is None else int(b)
        sg = desc._groups(self.b)
        vol = None if mol is None else torch.as_tensor(mol).to(device=dev, dtype=F64).reshape(self.b, -1)
        self.desc, self.dev = desc, dev
        self.parts = []   # (group, rows, chart row of each slot or -1, stepper)
        for g in sorted(set(sg.tolist())):
            rows = torch.nonzero(sg == g).flatten().to(dev)
            chart_row = torch.full((desc.dim,), -1, dtype=torch.long)
            chart_row[desc.layout.slots[g]] = torch.arange(len(desc.layout.slots[g]))
            st = desc.per_group[g].stepper(r[rows], None if vol is None else vol[rows], len(rows), dev)
            self.parts.append((g, rows, chart_row.tolist(), st))

    def row(self, s, i):
        d = self.desc
        out = torch.zeros(self.b, d.k, dtype=F64, device=self.dev)
        for g, rows, chart_row, st in self.parts:
            tag = d.tags[g][i].to(self.dev)
            j = chart_row[i]
            if j >= 0:
                xs = d.layout.unpack(g, s[rows.to(s.device)])
                out[rows, :d.k_group] = st.row(xs, j).to(device=self.dev, dtype=F64)
            out[rows, d.k_group:] = tag
        return out
