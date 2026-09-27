"""The one-pass multi-molecule energy scores every row exactly as its own member would.

`MultiConformerTorsions.energy` used to loop over the molecules in a batch, one member call
each; it now builds the whole mixed batch from the condition graphs and a gathered force
field in one pass (energies/ff_library.py). Every claim below is asserted against each
MEMBER's own `energy` on its own rows at its own width -- an oracle that shares none of the
one-pass machinery, reassembled here from the assignment the test itself made -- because a
dispatch error returns plausible numbers, not an exception.

The batch is chosen to reach the paths a friendly batch would miss: eight members spanning
k = 6..33 (a non-identity carrier), terms present on some members and absent on others
(torsion, out-of-plane, nonbonded pairs, a ring closure), unequal group sizes including
singletons, states uniform in [-1.2, 1.2] so the box wall and steric clashes are live, and a
different temperature on every row. Level `full`, force field `mmff`.
"""
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from energies.conformer_carrier import carrier_pad_condition
from energies.conformer_data import collate_conditions, condition_from_energy
from energies.multi_conformer import MultiConformerTorsions

KW = dict(device='cpu', level='full', force_field='mmff')
SMIS = ['C', 'CO', 'N', 'CC(=O)N', 'c1ccccc1O', 'C1CCOC1', 'CCCO', 'OCC=O']
SIZES = [1, 3, 1, 12, 7, 15, 20, 5]           # B = 64: unequal, two singletons
B = sum(SIZES)
ISOMERS = ['CCCO', 'CC(C)O']                  # identical block counts, different z order


@contextmanager
def _default_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextmanager
def _clip(multi, clip):
    """energy_clip on the dispatcher AND every member, so the oracle scores the same target."""
    held = [(m, m.energy_clip) for m in [multi, *multi._members.values()]]
    for m, _ in held:
        m.energy_clip = clip
    try:
        yield
    finally:
        for m, c in held:
            m.energy_clip = c


class _Set:
    """A built energy, a mixed batch over it, and the assignment the oracle is built from."""

    def __init__(self, smis, sizes, dtype, seed=0, scale=1.2, **kw):
        self.smis, self.dtype = list(smis), dtype
        with _default_dtype(dtype):
            self.multi = MultiConformerTorsions(self.smis, identifiers=self.smis,
                                                **{**KW, **kw})
            self.lay = self.multi.carrier
            conds = {}
            for ident, m in self.multi._members.items():
                c = condition_from_energy(m, identifier=ident)
                conds[ident] = (carrier_pad_condition(c, self.lay, ident, m)
                                if self.lay is not None else c)
            # as init_identifiers mints it: sorted identifiers -> 0..M-1, so a row's mol_id
            # is NOT its library index and a lookup that confused the two would show
            self.registry = {k: j for j, k in enumerate(sorted(self.smis, reverse=True))}
            self.multi.bind_identifier_registry(self.registry)
            rng = np.random.default_rng(seed)
            self.assign = np.repeat(np.arange(len(self.smis)), sizes)[
                rng.permutation(int(sum(sizes)))]
            rows, xs = [], []
            for j in self.assign:
                ident = self.smis[j]
                c = conds[ident].__copy__()
                c.mol_id = torch.tensor([self.registry[ident]])
                rows.append(c)
                k = self.multi._members[ident].data_ndim
                x = torch.as_tensor(rng.uniform(-scale, scale, (1, k)), dtype=dtype)
                xs.append(self.lay.to_carrier(ident, x) if self.lay is not None else x)
            self.batch = collate_conditions(rows)
            self.X = torch.cat(xs)
            self.logT = torch.as_tensor(rng.uniform(-0.5, 0.5, len(self.assign)), dtype=dtype)

    def member_x(self, ident, X):
        return self.lay.from_carrier(ident, X) if self.lay is not None else X

    def oracle(self, X=None, logT=None, keep_grads=False):
        """Each member's own `energy` on its own rows, placed back in batch order.

        In the member's OWN output dtype. A member built under a float32 default still
        returns float64: its reference DoF vector and scales are built from float64 numpy
        with the constructor's `dtype=None`, so its geometry runs in float64 while its force
        field is float32. The one-pass path runs wholly in the run dtype; the float32 tests
        measure that difference, which is the change production sees.
        """
        X = self.X if X is None else X
        logT = self.logT if logT is None else logT
        out = None
        for j, ident in enumerate(self.smis):
            rows = torch.as_tensor(np.flatnonzero(self.assign == j), dtype=torch.long)
            if rows.numel() == 0:
                continue
            m = self.multi._members[ident]
            e = m.energy(self.member_x(ident, X.index_select(0, rows)), None,
                         logT.index_select(0, rows), keep_grads=keep_grads)
            out = (torch.zeros(X.shape[0], dtype=e.dtype) if out is None else out)
            out = out.index_put((rows,), e)
        return out

    def potential_oracle(self):
        """Each member's baked potential (`potential_energy` at T = 1), in batch order."""
        out = torch.zeros(self.X.shape[0], dtype=torch.float64)
        one = torch.tensor(1.0, dtype=self.dtype)
        for j, ident in enumerate(self.smis):
            rows = torch.as_tensor(np.flatnonzero(self.assign == j), dtype=torch.long)
            m = self.multi._members[ident]
            out[rows] = m.potential_energy(self.member_x(ident, self.X[rows]),
                                           one).to(torch.float64)
        return out

    @property
    def pad_mask(self):
        return ~self.batch.state_mask.reshape(self.X.shape).bool()


@pytest.fixture(scope='module')
def s64():
    return _Set(SMIS, SIZES, torch.float64)


@pytest.fixture(scope='module')
def s32():
    return _Set(SMIS, SIZES, torch.float32)


def _close(got, want, rtol=1e-12, atol=1e-9):
    return torch.allclose(got, want, rtol=rtol, atol=atol)


# ------------------------------------------------------------------ the batch is what it claims

def test_the_batch_reaches_the_paths_it_is_meant_to(s64):
    """Premises, so a later edit to the fixture cannot quietly make every test below easy."""
    lay = s64.multi.carrier
    assert s64.multi.is_carrier and not lay.is_identity
    counts = np.bincount(s64.assign, minlength=len(SMIS))
    assert (counts == SIZES).all() and (counts == 1).sum() == 2
    assert len(set(counts.tolist())) > 3, 'group sizes must be unequal'
    ffs = {i: m.ff_single for i, m in s64.multi._members.items()}
    has = lambda f, t: getattr(f, t) is not None and getattr(f, t).numel() > 0
    for term in ('torsion_index', 'oop_index', 'pair_index'):
        assert any(has(f, term) for f in ffs.values()) and \
            not all(has(f, term) for f in ffs.values()), term
    assert any(has(f, 'closure_index') for f in ffs.values())
    # the wall is live: some row sits past the box on a non-periodic column
    lin = s64.multi._lin_free_idx
    assert bool((s64.X.index_select(1, lin).abs() > 1).any())
    assert float(s64.logT.std()) > 0.1
    # rows of one molecule are scattered, not contiguous
    assert (np.diff(s64.assign) != 0).sum() > len(SMIS)


# ------------------------------------------------------------------ values and gradients

@pytest.mark.parametrize('clip', [None, 30.0])
def test_energy_matches_the_member_oracle(s64, clip):
    with _clip(s64.multi, clip):
        got = s64.multi.energy(s64.X, s64.batch, s64.logT)
        want = s64.oracle()
        assert _close(got, want), float((got - want).abs().max())
        # the private per-member loop is the same oracle, so timing it is a fair comparison
        assert torch.equal(s64.multi._energy_per_member(s64.X, s64.batch, s64.logT), want)
    if clip is None:
        # clashes are live: the unclipped force field reaches far past any clip
        assert float(want.max()) > 1e3


@pytest.mark.parametrize('clip', [None, 30.0])
def test_gradient_matches_the_member_oracle(s64, clip):
    """The pathwise force: dispatcher gradient == member gradient; pads carry none."""
    with _clip(s64.multi, clip):
        xa = s64.X.clone().requires_grad_(True)
        g, = torch.autograd.grad(s64.multi.energy(xa, s64.batch, s64.logT,
                                                  keep_grads=True).sum(), xa)
        xb = s64.X.clone().requires_grad_(True)
        go, = torch.autograd.grad(s64.oracle(xb, keep_grads=True).sum(), xb)
    scale = float(go.abs().max())
    assert float((g - go).abs().max()) <= 1e-9 * scale
    pads = s64.pad_mask
    assert pads.any() and torch.equal(g[pads], torch.zeros_like(g[pads]))


def test_without_keep_grads_there_is_no_graph(s64):
    xa = s64.X.clone().requires_grad_(True)
    assert not s64.multi.energy(xa, s64.batch, s64.logT).requires_grad


def test_return_exp_bakes_the_members_potential(s64):
    """`conformer_energy` is the member's potential at T = 1 (wall included, measure not);
    `gfn_energy` is E before the division by T."""
    e, out = s64.multi.energy(s64.X, s64.batch.clone(), s64.logT, return_exp=True)
    assert torch.equal(e, s64.multi.energy(s64.X, s64.batch, s64.logT))
    assert _close(out.conformer_energy, s64.potential_oracle())
    assert _close(out.gfn_energy, s64.oracle() * 10 ** s64.logT)


def test_rows_permute_with_the_batch(s64):
    e = s64.multi.energy(s64.X, s64.batch, s64.logT)
    perm = np.random.default_rng(1).permutation(B)
    p = torch.as_tensor(perm)
    got = s64.multi.energy(s64.X[p], s64.batch.subsample_new_batch(perm), s64.logT[p])
    assert _close(got, e[p])


def test_subsample_and_append_preserve_every_score(s64):
    """The buffer's own operations: draw two subsets, append them, score the result."""
    e = s64.multi.energy(s64.X, s64.batch, s64.logT)
    perm = np.random.default_rng(2).permutation(B)
    a, b = perm[:23], perm[23:]
    batch = s64.batch.subsample_new_batch(a).append_batch(s64.batch.subsample_new_batch(b))
    order = torch.as_tensor(np.concatenate([a, b]))
    got = s64.multi.energy(s64.X[order], batch, s64.logT[order])
    assert _close(got, e[order])

    # a row that has been through a buffer carries mol_id only; one straight off a file
    # carries identifier only. Both must resolve to the same charts.
    mol_id_only = batch.clone()
    del mol_id_only.identifier
    assert _close(s64.multi.energy(s64.X[order], mol_id_only, s64.logT[order]), e[order])
    ident_only = batch.clone()
    del ident_only.mol_id
    assert _close(s64.multi.energy(s64.X[order], ident_only, s64.logT[order]), e[order])


@pytest.mark.parametrize('clip', [None, 30.0])
def test_float32_matches_the_member_oracle(s32, clip):
    """PRODUCTION DTYPE: conformer_modeller.py pins float32. Same comparison, both sides in
    float32, at relative 1e-5 -- two summation orders of the same float32 arithmetic."""
    with _clip(s32.multi, clip):
        got = s32.multi.energy(s32.X, s32.batch, s32.logT)
        want = s32.oracle()
    assert got.dtype == torch.float32
    rel = ((got - want).abs() / want.abs().clamp_min(1.0)).max()
    assert float(rel) < 1e-5, float(rel)


def test_float32_gradient_matches_the_member_oracle(s32):
    xa = s32.X.clone().requires_grad_(True)
    g, = torch.autograd.grad(s32.multi.energy(xa, s32.batch, s32.logT,
                                              keep_grads=True).sum(), xa)
    xb = s32.X.clone().requires_grad_(True)
    go, = torch.autograd.grad(s32.oracle(xb, keep_grads=True).sum(), xb)
    assert float((g - go).abs().max()) <= 1e-5 * float(go.abs().max())
    assert torch.equal(g[s32.pad_mask], torch.zeros_like(g[s32.pad_mask]))


# ------------------------------------------------------------------ prebuilt rewards

def test_prebuilt_reward_matches_minus_the_member_energy(s64):
    """Backward draws and the prior/anchor seed read the BAKED potential and add the
    measure back per row. At T = 1 that is exactly -energy; at other temperatures the
    baked potential (wall included) is divided by T and the measure is not."""
    zero = torch.zeros_like(s64.logT)
    _, baked = s64.multi.energy(s64.X, s64.batch.clone(), zero, return_exp=True)
    want = -s64.oracle(logT=zero)
    assert _close(s64.multi.prebuilt_sample_to_reward(baked, 1.0), want)

    t = 10 ** s64.logT
    u = baked.conformer_energy
    measure = u - s64.oracle(logT=zero)          # log J + log|dq/dx| per row, from members
    assert _close(s64.multi.prebuilt_sample_to_reward(baked, t), -(u / t) + measure)

    perm = np.random.default_rng(3).permutation(B)
    sub = baked.subsample_new_batch(perm)
    assert _close(s64.multi.prebuilt_sample_to_reward(sub, 1.0), want[torch.as_tensor(perm)])


def test_prebuilt_reward_at_torsion_uses_each_members_constant():
    """At `torsion` log J is a per-molecule constant; the reward reads no geometry."""
    s = _Set(['CCCO', 'CCCCO', 'CCC(=O)CC'], [3, 1, 4], torch.float64, level='torsion',
             scale=0.9)
    assert s.multi.is_carrier and s.multi._log_jac_const_of_lib is not None
    zero = torch.zeros_like(s.logT)
    _, baked = s.multi.energy(s.X, s.batch.clone(), zero, return_exp=True)
    assert _close(s.multi.prebuilt_sample_to_reward(baked, 1.0), -s.oracle(logT=zero))


# ------------------------------------------------------------------ refusals

def test_a_nonzero_pad_is_refused(s64):
    X2 = s64.X.clone()
    i = int(np.flatnonzero(s64.assign == SMIS.index('C'))[0])
    X2[i, int(s64.lay.pad_cols('C')[0])] = 1e-3
    with pytest.raises(RuntimeError, match='PAD'):
        s64.multi.energy(X2, s64.batch, s64.logT)


def test_a_foreign_state_mask_is_refused(s64):
    b2 = s64.batch.clone()
    b2.state_mask = torch.ones_like(b2.state_mask)
    with pytest.raises(RuntimeError, match='state_mask'):
        s64.multi.energy(s64.X, b2, s64.logT)


def test_an_unregistered_mol_id_is_refused(s64):
    b2 = s64.batch.clone()
    b2.mol_id = b2.mol_id.clone()
    b2.mol_id[5] = 999
    with pytest.raises(RuntimeError, match='not in the identifier registry'):
        s64.multi.energy(s64.X, b2, s64.logT)


def test_an_unknown_identifier_is_refused(s64):
    b2 = s64.batch.clone()
    del b2.mol_id
    b2.identifier = ['CCCCCCO'] + list(b2.identifier[1:])
    with pytest.raises(RuntimeError, match='not built for'):
        s64.multi.energy(s64.X, b2, s64.logT)


@pytest.mark.parametrize('extra', [['C'], []], ids=['carrier', 'identity'])
def test_a_mol_id_swapped_between_isomers_is_refused_by_z(extra):
    """CCCO and CC(C)O have IDENTICAL block counts, so they share every carrier column: the
    pad check and the state_mask check both pass on a swapped registration. Only the z check
    sees it -- without it, one isomer's force field is applied to the other's geometry."""
    s = _Set(ISOMERS + extra, [3, 3] + [2] * len(extra), torch.float64, scale=0.3)
    assert s.multi.is_carrier == bool(extra)
    m1, m2 = (s.multi._members[i] for i in ISOMERS)
    assert m1.data_ndim == m2.data_ndim and m1.spec.n_atoms == m2.spec.n_atoms
    assert (np.asarray(m1._free_block) == np.asarray(m2._free_block)).all()
    assert not (np.asarray(m1.spec.z) == np.asarray(m2.spec.z)).all()
    assert _close(s.multi.energy(s.X, s.batch, s.logT), s.oracle())   # unswapped: fine

    b2 = s.batch.clone()
    a, c = s.registry[ISOMERS[0]], s.registry[ISOMERS[1]]
    b2.mol_id = torch.where(b2.mol_id == a, c, torch.where(b2.mol_id == c, a, b2.mol_id))
    with pytest.raises(RuntimeError, match="not that molecule's"):
        s.multi.energy(s.X, b2, s.logT)


def test_a_row_count_mismatch_is_refused(s64):
    with pytest.raises(RuntimeError, match='disagree about how many'):
        s64.multi.energy(s64.X[:-1], s64.batch, s64.logT[:-1])


def test_a_batch_in_another_dtype_is_refused(s64, s32):
    """A float64 conditions file under a float32 run: refused, not built in mixed precision."""
    with pytest.raises(RuntimeError, match='run dtype'):
        s32.multi.energy(s64.X.float(), s64.batch, s64.logT.float())


def test_a_transverse_flag_its_member_does_not_have_is_refused(s64):
    """No member of this set has a linear bend. A row flagged transverse would build a (u, v)
    bend out of an ordinary atom's theta/phi slots; the flag is compared atom by atom with the
    member's chart and refused. The nitrile half is tests/conformer/test_transverse_carrier.py."""
    b2 = s64.batch.clone()
    b2.ctree_transverse = b2.ctree_transverse.clone()
    b2.ctree_transverse[7] = True
    with pytest.raises(RuntimeError, match='transverse flags'):
        s64.multi.energy(s64.X, b2, s64.logT)


# ------------------------------------------------------------------ per-member dispatch

def test_member_groups_partition_the_rows_by_molecule(s64):
    groups = s64.multi.member_groups(s64.batch)
    seen = torch.cat([rows for _, rows in groups])
    assert sorted(seen.tolist()) == list(range(B))
    for ident, rows in groups:
        assert (s64.assign[rows.numpy()] == SMIS.index(ident)).all()
    assert len(groups) == len(SMIS)
