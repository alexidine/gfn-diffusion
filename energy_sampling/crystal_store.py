"""One compact store of scored crystal rows: plain per-row columns plus one molecule template per condition.

Row columns are host tensors preallocated to `max_rows` and written by slot:

    x            f32 [d]  chart row
    cid          i32      condition id
    kind         i8       ANCHOR, CHILD, ONPOLICY, FREE
    flags        u8       bit 0 UNDRAWN, bit 1 EVAL
    origin       i8       ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_BOOTSTRAP, ORIGIN_SEED, ORIGIN_KICK
    parent       i32      anchor slot of a child, -1 otherwise
    log_r        f32      log R in the run's currency
    energy       f32      mol_energy, unreferenced
    log_q        f32      log q at the last touch or at on-policy admission; NaN = none
    log_q_birth  f32      log q at on-policy admission, written once; NaN on anchors and children
    q_step       i32      step of the log_q write; -1 = none
    resid        f32      signed log_pf - log_pb - log_r + log Z at that write; NaN = none
    draws        i32      times trained on, by occurrence
    birth_step   i32      step the row was written; 0 for seed rows

Per-condition tables: tmpl_of, cond_sg, cond_radius, cond_rg, best_energy, window_floor, cond_stats.

Invariants, verified by check():

    I1  A row's id is its slot until it is freed. A freed slot has kind FREE. epoch increments whenever slots are freed,
        and touch with another epoch writes nothing.
    I2  Every CHILD has parent = an ANCHOR slot of its cid. Anchors are never thinned. remove refuses an anchor that keeps
        a live child. ONPOLICY and ANCHOR rows have parent -1.
    I3  add is O(rows added) outside a thin. No column is rebuilt.
    I4  A row is in pool (cid, kind) iff kind != FREE and UNDRAWN is clear.
    I5  n_rows = high_water - n_free; every slot below high_water is live or on the free stack.
    I6  best_energy[c] <= energy of every finite row ever given for c, including rows thinned away.
    I7  window_floor is written once, at seed.
    I8  Every live row's condition has a template.
    I9  draw(kind, B) returns B rows of kind, conditions uniform over the conditions with a drawable row of kind. It
        raises on an empty pool.
    I10 touch refuses anchors, so no reward or log q column of an anchor is written after its seed.
    I11 max_rows >= n_anchors + 2 * headroom. add raises on a single add above headroom.
    I12 log_q_birth is written once, at on-policy admission. A drawable on-policy row has a finite log_q and log_q_birth.

The store owns one numpy Generator. It never reads or advances the global torch or numpy generators.

This module imports hashlib, json, math, numpy, torch and noised_prior at load. Joining template batches imports
mxtaltools.dataset_utils.utils when called.
"""
import hashlib
import json
import math

import numpy as np
import torch

from energy_sampling import noised_prior as NP

ANCHOR, CHILD, ONPOLICY, FREE = 0, 1, 2, -1
KIND_NAMES = ('anchor', 'child', 'onpolicy')
N_KINDS = len(KIND_NAMES)
UNDRAWN, EVAL = 1, 2
ORIGIN_ROLLOUT, ORIGIN_EVAL, ORIGIN_BOOTSTRAP = 0, 1, 2     # the values of buffer.ORIGIN_*
ORIGIN_SEED, ORIGIN_KICK = 3, 4
FORMAT_VERSION = 3
THIN_BLOCK = 0.05                       # share of max_rows one thin frees, when that many rows are available
MIN_POOL_CAP = 8
MAX_ROWS_LIMIT = 2 ** 31 - 1
_SEED_THIN_STREAM = 29                  # second word of the seed-thin generator's seed sequence

_COLUMNS = (('x', torch.float32), ('cid', torch.int32), ('kind', torch.int8), ('flags', torch.uint8),
            ('origin', torch.int8), ('parent', torch.int32), ('log_r', torch.float32), ('energy', torch.float32),
            ('log_q', torch.float32), ('log_q_birth', torch.float32), ('q_step', torch.int32),
            ('resid', torch.float32), ('draws', torch.int32), ('birth_step', torch.int32))
_TABLES = ('tmpl_of', 'cond_sg', 'cond_radius', 'cond_rg', 'best_energy', 'window_floor')
_COUNTERS = ('n_free', 'high_water', 'epoch', 'generation', 'n_seed_given', 'n_seed_kept', 'seed_nonfinite',
             'seed_orphans', 'thinned_total', 'nonfinite_rows', 'seeded', 'seed_intact')
_SEED_MUTABLE = ('log_q', 'q_step', 'resid', 'draws')   # the only columns of a seed row written after seed()
_SEED_EXACT = ('cid', 'kind', 'flags', 'origin', 'parent', 'birth_step')     # seed-row columns hashed byte for byte
_SEED_CLOSE_TABLES = ('cond_radius', 'cond_rg', 'window_floor')             # compared within the tolerances below
SEED_CLOSE_RTOL, SEED_CLOSE_ATOL = 1e-4, 1e-3   # per-condition means of x, log_r, energy and the tables above
_STAMP_CLOSE = ('energy_clip', 'energy_reference_table')    # stamp entries recomputed at launch, read within the same
_DIGEST_CHUNK = 2 ** 20


class CrystalStoreError(ValueError):
    """A call that would break a store invariant."""


def uniform(store, rows, step):
    """The default replay weight: None selects the uniform draw within each condition."""
    return None


#: rule(store, rows, step) -> [len(rows)] weights >= 0, or None for uniform within the condition
REPLAY_WEIGHTS = {'uniform': uniform}


def _blank_stat(t, k):
    """k new entries of the cond_stats tensor t: NaN on a float tensor, zero on any other."""
    shape = (k,) + tuple(t.shape[1:])
    return torch.full(shape, math.nan, dtype=t.dtype) if t.is_floating_point() else torch.zeros(shape, dtype=t.dtype)


def _stamp_close(a, b):
    """Two values of a _STAMP_CLOSE key agree entrywise within SEED_CLOSE_RTOL and SEED_CLOSE_ATOL; None matches only
    None."""
    if a is None or b is None:
        return a is None and b is None
    a, b = torch.as_tensor(a, dtype=torch.float64).flatten(), torch.as_tensor(b, dtype=torch.float64).flatten()
    return a.shape == b.shape and bool(torch.isclose(a, b, rtol=SEED_CLOSE_RTOL, atol=SEED_CLOSE_ATOL,
                                                     equal_nan=True).all())


def stamp_diff(a, b):
    """The top-level keys on which two stamps differ. Tuples read as lists and NaN equals NaN. The _STAMP_CLOSE keys are
    recomputed from the rows at each launch, so they are compared within the seed digest's tolerances."""
    def canon(v):
        return json.dumps(v, sort_keys=True, default=str)
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [] if canon(a) == canon(b) else ['<whole stamp>']

    def same(k):
        return _stamp_close(a.get(k), b.get(k)) if k in _STAMP_CLOSE else canon(a.get(k)) == canon(b.get(k))
    return sorted(k for k in set(a) | set(b) if not same(k))


def _ragged(starts, lens):
    """The concatenated integer ranges starts[i] .. starts[i] + lens[i] - 1, as int64."""
    total = int(lens.sum())
    offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts.astype(np.int64), lens) + offs


class _SparseSet:
    """A set of ids below n as a dense list plus each id's position in it."""

    def __init__(self, n):
        self.items = np.empty(n, dtype=np.int32)
        self.pos = np.full(n, -1, dtype=np.int32)
        self.n = 0

    def add(self, ids):
        """ids: unique; those already present are skipped."""
        new = ids[self.pos[ids] < 0]
        m = len(new)
        self.items[self.n:self.n + m] = new
        self.pos[new] = np.arange(self.n, self.n + m, dtype=np.int32)
        self.n += m

    def remove(self, ids):
        """ids: unique and all present. The survivors in the last len(ids) positions fill the holes."""
        new_n = self.n - len(ids)
        at = self.pos[ids]
        holes = at[at < new_n]
        self.pos[ids] = -1
        tail = self.items[new_n:self.n]
        movers = tail[self.pos[tail] >= 0]
        self.items[holes] = movers
        self.pos[movers] = holes
        self.n = new_n

    def grow(self, k):
        self.items = np.concatenate([self.items, np.empty(k, dtype=np.int32)])
        self.pos = np.concatenate([self.pos, np.full(k, -1, dtype=np.int32)])

    def members(self):
        return self.items[:self.n]

    def load(self, ids):
        self.pos[:] = -1
        self.n = 0
        self.add(ids)


class PoolIndex:
    """pool -> rows as a sparse set in one flat arena. arena[start[p] : start[p] + count[p]] are the pool's rows, in no
    order; where[row] is the row's position inside its pool. Insert O(k), remove O(k log k), uniform draw O(B).

    The arena holds 2 * max_rows + 8 * n_pools entries. A compaction gives each pool max(1.5 count, need, 8), which sums
    to at most 1.5 * max_rows + 8 * n_pools while the indexed rows plus the rows being added are at most max_rows, so a
    compaction always fits and one compaction always satisfies the add that asked for it."""

    def __init__(self, n_pools, max_rows):
        self.max_rows = int(max_rows)
        self.count = np.zeros(n_pools, dtype=np.int64)
        self.cap = np.zeros(n_pools, dtype=np.int64)
        self.start = np.zeros(n_pools, dtype=np.int64)
        self.where = np.empty(self.max_rows, dtype=np.int32)
        self.arena = np.empty(2 * self.max_rows + MIN_POOL_CAP * n_pools, dtype=np.int32)
        self.tail = 0
        self.compactions = 0

    def build(self, rows, pools):
        """Fill an empty index by one sort. rows: unique; pools: their pool ids."""
        order = np.argsort(pools, kind='stable')
        rows, pools = rows[order], pools[order]
        n_pools = len(self.count)
        cnt = np.bincount(pools, minlength=n_pools).astype(np.int64)
        self.cap = (cnt * 3) // 2 + MIN_POOL_CAP
        self.start = np.concatenate([[0], np.cumsum(self.cap)[:-1]]).astype(np.int64)
        ptr = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        off = np.arange(len(rows), dtype=np.int64) - ptr[pools]
        self.arena[self.start[pools] + off] = rows
        self.where[rows] = off.astype(np.int32)
        self.count = cnt
        self.tail = int(self.cap.sum())

    def _reserve(self, pools, extra):
        """pools: unique ids; extra: rows each is about to receive. A pool without room moves to the tail at twice its
        capacity, or the arena is compacted with every pool of this add at its need."""
        need = self.count[pools] + extra
        short = need > self.cap[pools]
        if not short.any():
            return
        ps, ns = pools[short], need[short]
        new_caps = np.maximum(2 * self.cap[ps], np.maximum(ns, MIN_POOL_CAP))
        if self.tail + int(new_caps.sum()) > self.arena.size:
            self._compact(pools, need)
            return
        k = self.count[ps]
        new_starts = self.tail + np.cumsum(new_caps) - new_caps
        self.arena[_ragged(new_starts, k)] = self.arena[_ragged(self.start[ps], k)]
        self.start[ps], self.cap[ps] = new_starts, new_caps
        self.tail += int(new_caps.sum())

    def _compact(self, pools=None, need=None):
        """Rewrite the arena with each pool at max(1.5 count, 8), and at its need for `pools`."""
        self.compactions += 1
        cap = np.maximum((self.count * 3) // 2, MIN_POOL_CAP)
        if pools is not None:
            cap[pools] = np.maximum(cap[pools], need)
        total = int(cap.sum())
        if total > self.arena.size:
            raise CrystalStoreError(f'index arena of {self.arena.size} entries cannot hold {total}')
        new_start = np.cumsum(cap) - cap
        live = np.flatnonzero(self.count)
        k = self.count[live]
        new = np.empty(self.arena.size, dtype=np.int32)
        new[_ragged(new_start[live], k)] = self.arena[_ragged(self.start[live], k)]
        self.arena, self.start, self.cap, self.tail = new, new_start, cap, total

    def add(self, rows, pools):
        """Insert rows; returns the pools that received rows."""
        order = np.argsort(pools, kind='stable')
        rows, pools = rows[order], pools[order]
        up, first, cnt = np.unique(pools, return_index=True, return_counts=True)
        self._reserve(up, cnt)
        off = np.arange(len(rows), dtype=np.int64) - np.repeat(first, cnt)
        at = self.count[pools] + off
        self.arena[self.start[pools] + at] = rows
        self.where[rows] = at.astype(np.int32)
        self.count[up] += cnt
        return up

    def remove(self, rows, pools):
        """Remove unique indexed rows; returns the pools left empty. Each pool's survivors in its last k positions fill
        its holes."""
        order = np.argsort(pools, kind='stable')
        rows, pools = rows[order], pools[order]
        up, first, k = np.unique(pools, return_index=True, return_counts=True)
        new_n = self.count[up] - k
        at = self.where[rows].astype(np.int64)
        is_hole = at < np.repeat(new_n, k)
        hole_start = self.start[pools[is_hole]]
        hole_abs = hole_start + at[is_hole]
        tail_abs = _ragged(self.start[up] + new_n, k)
        self.where[rows] = -1
        tail_rows = self.arena[tail_abs]
        movers = tail_rows[self.where[tail_rows] >= 0]
        if len(movers) != len(hole_abs):
            raise CrystalStoreError(f'index remove: {len(hole_abs)} holes but {len(movers)} movers')
        self.arena[hole_abs] = movers
        self.where[movers] = (hole_abs - hole_start).astype(np.int32)
        self.count[up] = new_n
        return up[new_n == 0]

    def uniform(self, pools, rng):
        """One uniform row of each entry of `pools` (each must be non-empty)."""
        return self.arena[self.start[pools] + rng.integers(0, self.count[pools])]

    def rows_of(self, pools):
        """The rows of `pools`, concatenated in pool order and in arena order within each."""
        return self.arena[_ragged(self.start[pools], self.count[pools])]

    def grow(self, k):
        """Append k empty pools and MIN_POOL_CAP * k arena entries."""
        for name in ('count', 'cap', 'start'):
            setattr(self, name, np.concatenate([getattr(self, name), np.zeros(k, dtype=np.int64)]))
        new = np.empty(self.arena.size + MIN_POOL_CAP * k, dtype=np.int32)
        new[:self.tail] = self.arena[:self.tail]
        self.arena = new

    def load(self, count, start, cap, tail, compactions, rows):
        """Restore the saved layout and each pool's rows, given in pool order and in arena order within each pool."""
        self.count, self.start, self.cap = count.copy(), start.copy(), cap.copy()
        self.tail, self.compactions = int(tail), int(compactions)
        live = np.flatnonzero(self.count)
        k = self.count[live]
        self.arena[_ragged(self.start[live], k)] = rows
        self.where[rows] = (np.arange(len(rows), dtype=np.int64) - np.repeat(np.cumsum(k) - k, k)).astype(np.int32)


class RolloutPool:
    """The trailing n rows of on-policy training rollouts, oldest first, as a list of chunks on the CPU. A chunk is
    (step, per-row tensors, crystal batch) and is dropped once the chunks after it hold n rows, so the pool holds fewer
    than n + one chunk's rows. Nothing here reads the store."""

    def __init__(self, n):
        self.n = int(n)
        if self.n < 1:
            raise CrystalStoreError(f'a rollout pool needs n >= 1, got {n}')
        self._chunks = []
        self.rows = 0

    def clear(self):
        self._chunks, self.rows = [], 0

    def _first_rows(self):
        return next(iter(self._chunks[0][1].values())).shape[0]

    def add(self, rows, batch, step):
        """rows: name -> [k, ...] tensor, detached to the CPU here; batch: the k crystals, row for row."""
        rows = {name: t.detach().cpu() for name, t in rows.items()}
        k = int(batch.num_graphs)
        if any(t.shape[0] != k for t in rows.values()):
            raise CrystalStoreError(f'pool chunk: {k} crystals but columns of {sorted({t.shape[0] for t in rows.values()})} '
                                    f'rows')
        if k == 0:
            return
        self._chunks.append((int(step), rows, batch))
        self.rows += k
        while self.rows - self._first_rows() >= self.n:
            self.rows -= self._first_rows()
            self._chunks.pop(0)

    def read(self, join):
        """(rows, batch, step of the oldest chunk read) for the newest min(n, held) rows, oldest first, or
        (None, None, None) when empty. join(list of batches) -> one batch. The batch returned is a copy."""
        if not self._chunks:
            return None, None, None
        drop = self.rows - self.n
        if drop > 0:
            step, rows, batch = self._chunks[0]
            keep = torch.arange(drop, self._first_rows())
            self._chunks[0] = (step, {name: t[keep] for name, t in rows.items()}, batch.subsample_new_batch(keep))
            self.rows = self.n
        names = list(self._chunks[0][1])
        out = {name: torch.cat([c[1][name] for c in self._chunks]) for name in names}
        return out, join([c[2] for c in self._chunks]).clone(), self._chunks[0][0]


class CrystalStore:
    """See the module docstring. `headroom` is H of I11, the largest single add."""

    def __init__(self, dim, n_cond, max_rows, headroom, seed=0):
        self.dim, self.n_cond, self.max_rows, self.headroom = int(dim), int(n_cond), int(max_rows), int(headroom)
        if self.dim < 1 or self.n_cond < 1 or self.headroom < 1:
            raise CrystalStoreError(f'dim, n_cond and headroom must be >= 1, got {dim}, {n_cond}, {headroom}')
        if self.max_rows < 2 * self.headroom:
            raise CrystalStoreError(f'max_rows {max_rows} is below 2 * headroom = {2 * self.headroom}')
        if self.max_rows > MAX_ROWS_LIMIT:
            raise CrystalStoreError(f'max_rows {max_rows} is above the int32 slot limit {MAX_ROWS_LIMIT}')
        self.base_seed = self._seed_word(seed)
        self.rng = np.random.default_rng(self.base_seed)
        for name, dtype in _COLUMNS:
            shape = (self.max_rows, self.dim) if name == 'x' else (self.max_rows,)
            setattr(self, name, torch.empty(shape, dtype=dtype, device='cpu'))
        n = self.n_cond
        self.tmpl_of = torch.full((n,), -1, dtype=torch.int32)
        self.cond_sg = torch.full((n,), -1, dtype=torch.int16)
        self.cond_radius = torch.full((n,), math.nan, dtype=torch.float32)
        self.cond_rg = torch.full((n,), math.nan, dtype=torch.float32)
        self.best_energy = torch.full((n,), math.inf, dtype=torch.float32)
        self.window_floor = torch.full((n,), math.inf, dtype=torch.float64)
        self._floor_written = self.window_floor.clone()
        self.cond_stats = {}
        self.templates = None
        self.index = PoolIndex(N_KINDS * n, self.max_rows)
        self.active = [_SparseSet(n) for _ in range(N_KINDS)]
        self._free = np.empty(self.max_rows, dtype=np.int32)
        self.n_free = self.high_water = 0
        self.epoch = self.generation = 0
        self.n_seed_given = self.n_seed_kept = self.seed_nonfinite = self.seed_orphans = 0
        self.thinned_total = self.nonfinite_rows = 0
        self.seeded = self.seed_intact = False
        self._seed_digest = None
        self._n_kind = [0] * N_KINDS
        self._n_drawable = [0] * N_KINDS
        self._n_undrawn = 0
        self._anchor_rows = None

    # ---- small helpers --------------------------------------------------------------

    @staticmethod
    def _seed_word(value):
        value = int(value)
        if value < 0:
            raise CrystalStoreError(f'a generator seed must be >= 0, got {value}')
        return value

    def reseed(self, seed, step):
        """Replace the generator by one keyed on (seed, step)."""
        self.rng = np.random.default_rng([self._seed_word(seed), self._seed_word(step)])

    @property
    def n_rows(self):
        return self.high_water - self.n_free

    @property
    def n_anchors(self):
        return self._n_kind[ANCHOR]

    def _vec(self, v, n, dtype, name):
        t = torch.as_tensor(v).detach().cpu().to(dtype)
        if t.dim() == 0:
            return t.expand(n).clone()
        t = t.reshape(-1)
        if t.numel() != n:
            raise CrystalStoreError(f'{name} has {t.numel()} entries, expected {n}')
        return t

    def _states(self, x):
        x = torch.as_tensor(x).detach().cpu().to(torch.float32)
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise CrystalStoreError(f'x has shape {tuple(x.shape)}, expected (n, {self.dim})')
        return x

    def _cids(self, cid, n=None):
        cid = torch.as_tensor(cid).detach().cpu().long()
        if n is not None and cid.dim() == 0:
            cid = cid.expand(n).clone()
        cid = cid.reshape(-1)
        if n is not None and cid.numel() != n:
            raise CrystalStoreError(f'cid has {cid.numel()} entries, expected {n}')
        if cid.numel() and (int(cid.min()) < 0 or int(cid.max()) >= self.n_cond):
            raise CrystalStoreError(f'condition id outside [0, {self.n_cond})')
        return cid

    def _rows(self, rows):
        return torch.as_tensor(rows).detach().cpu().long().reshape(-1)

    def _check_kind(self, kind):
        if int(kind) not in (ANCHOR, CHILD, ONPOLICY):
            raise CrystalStoreError(f'kind must be one of ANCHOR, CHILD, ONPOLICY, got {kind}')
        return int(kind)

    def _pools_of(self, rows):
        """Pool ids of `rows` (int64 numpy array of slots)."""
        return self.cid.numpy()[rows].astype(np.int64) * N_KINDS + self.kind.numpy()[rows].astype(np.int64)

    def _index_add(self, rows):
        """Index rows (int64 numpy array of drawable slots)."""
        if not len(rows):
            return
        up = self.index.add(rows.astype(np.int32), self._pools_of(rows))
        for k in range(N_KINDS):
            self.active[k].add((up[up % N_KINDS == k] // N_KINDS).astype(np.int32))
        kinds = self.kind.numpy()[rows]
        for k in range(N_KINDS):
            self._n_drawable[k] += int((kinds == k).sum())

    def _template_fields(self, templates):
        """(space group, radius, radius of gyration) of each graph, one entry per graph."""
        n = int(templates.num_graphs)
        return (self._vec(templates.sg_ind, n, torch.int16, 'sg_ind'),
                self._vec(templates.radius, n, torch.float32, 'radius'),
                self._vec(NP.radius_of_gyration(templates), n, torch.float32, 'radius of gyration'))

    def _set_templates(self, templates, template_cids):
        """Point condition template_cids[i] at graph i of `templates` and fill its tables."""
        n = int(templates.num_graphs)
        tc = self._cids(template_cids)
        if tc.numel() != n or torch.unique(tc).numel() != n:
            raise CrystalStoreError(f'{n} templates need {n} distinct condition ids, got {tc.numel()}')
        sg, radius, rg = self._template_fields(templates)
        self.tmpl_of[tc] = torch.arange(n, dtype=torch.int32)
        self.cond_sg[tc], self.cond_radius[tc], self.cond_rg[tc] = sg, radius, rg

    # ---- seeding --------------------------------------------------------------------

    def seed(self, x, cid, kind, log_r, energy, parent, undrawn, templates, template_cids, window_floor):
        """Load the data rows once, into an empty store, in the given order.

        kind: ANCHOR or CHILD per row. parent: for a child, the position of its anchor among these rows; ignored on an
        anchor. templates: one graph per entry of template_cids. window_floor: [n_cond], written once.

        best_energy is taken over every row with a finite x, log_r and energy. Rows with a non-finite value are dropped,
        and so are children of a dropped anchor. When more than max_rows - headroom rows remain, all anchors are kept
        and children are chosen uniformly to fill the rest, by a generator keyed on the construction seed alone."""
        if self.seeded:
            raise CrystalStoreError('seed() is only allowed once, on an empty store')
        x = self._states(x)
        n = x.shape[0]
        cid = self._cids(cid, n)
        kind = self._vec(kind, n, torch.int8, 'kind')
        log_r = self._vec(log_r, n, torch.float32, 'log_r')
        energy = self._vec(energy, n, torch.float32, 'energy')
        parent = self._vec(parent, n, torch.int64, 'parent')
        undrawn = self._vec(undrawn, n, torch.bool, 'undrawn')
        floor = torch.as_tensor(window_floor).detach().cpu().to(torch.float64).reshape(-1)
        if floor.numel() != self.n_cond:
            raise CrystalStoreError(f'window_floor has {floor.numel()} entries, expected n_cond = {self.n_cond}')
        is_anchor, is_child = kind == ANCHOR, kind == CHILD
        if not bool((is_anchor | is_child).all()):
            raise CrystalStoreError('seed() takes ANCHOR and CHILD rows only')
        p = parent[is_child]
        if p.numel():
            if int(p.min()) < 0 or int(p.max()) >= n:
                raise CrystalStoreError('a child parent is not a position among the seed rows')
            if not bool(is_anchor[p].all()):
                raise CrystalStoreError('a child parent is not an anchor')
            if not bool((cid[p] == cid[is_child]).all()):
                raise CrystalStoreError('a child and its parent anchor have different conditions')

        finite = torch.isfinite(x).all(1) & torch.isfinite(log_r) & torch.isfinite(energy)
        self.best_energy.scatter_reduce_(0, cid[finite], energy[finite], reduce='amin', include_self=True)
        keep = finite.clone()
        keep[is_child] &= finite[p]
        n_nonfinite = int((~finite).sum())
        n_orphans = int((finite & ~keep).sum())
        n_anchors = int((keep & is_anchor).sum())
        if n_anchors + 2 * self.headroom > self.max_rows:
            raise CrystalStoreError(f'{n_anchors} anchors + 2 * headroom {self.headroom} exceed max_rows {self.max_rows}')
        room = self.max_rows - self.headroom
        if int(keep.sum()) > room:
            kids = torch.nonzero(keep & is_child).flatten().numpy()
            gen = np.random.default_rng([self.base_seed, _SEED_THIN_STREAM])
            picked = gen.choice(kids, room - n_anchors, replace=False)
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[torch.from_numpy(picked)] = True
            keep &= is_anchor | chosen
        if not bool((keep & is_child & ~undrawn).any()):
            raise CrystalStoreError('seed() leaves no drawable child')

        rows = torch.nonzero(keep).flatten()
        m = rows.numel()
        slot_of = torch.cumsum(keep.long(), 0) - 1
        self.templates = templates
        self._set_templates(templates, template_cids)
        if bool((self.tmpl_of[cid[rows]] < 0).any()):
            bad = cid[rows][self.tmpl_of[cid[rows]] < 0].unique()[:8].tolist()
            raise CrystalStoreError(f'seed rows of conditions {bad} have no template')
        self.x[:m] = x[rows]
        self.cid[:m] = cid[rows].to(torch.int32)
        self.kind[:m] = kind[rows]
        self.flags[:m] = undrawn[rows].to(torch.uint8) * UNDRAWN
        self.origin[:m] = ORIGIN_SEED
        child_parent = torch.where(is_child[rows], parent[rows], torch.zeros(m, dtype=torch.int64))
        self.parent[:m] = torch.where(is_child[rows], slot_of[child_parent],
                                      torch.full((m,), -1, dtype=torch.int64)).to(torch.int32)
        self.log_r[:m] = log_r[rows]
        self.energy[:m] = energy[rows]
        for name in ('log_q', 'log_q_birth', 'resid'):
            getattr(self, name)[:m] = math.nan
        self.q_step[:m] = -1
        self.draws[:m] = 0
        self.birth_step[:m] = 0
        self.window_floor[:] = floor
        self._floor_written = floor.clone()
        self.high_water = m
        self.n_seed_given, self.n_seed_kept = n, m
        self.seed_nonfinite, self.seed_orphans = n_nonfinite, n_orphans
        kinds = self.kind[:m]
        self._n_kind = [int((kinds == k).sum()) for k in range(N_KINDS)]
        drawable = np.flatnonzero((self.flags[:m].numpy() & UNDRAWN) == 0)
        self._n_undrawn = m - len(drawable)
        self.index.build(drawable.astype(np.int32), self._pools_of(drawable))
        for k in range(N_KINDS):
            pool_ids = np.arange(self.n_cond, dtype=np.int64) * N_KINDS + k
            self.active[k].add(np.flatnonzero(self.index.count[pool_ids]).astype(np.int32))
            self._n_drawable[k] = int(self.index.count[pool_ids].sum())
        self._anchor_rows = None
        self.seeded = self.seed_intact = True
        self._seed_digest = None
        self.generation += 1

    # ---- growth ---------------------------------------------------------------------

    def _take_slots(self, k):
        take = min(k, self.n_free)
        reused = self._free[self.n_free - take:self.n_free].astype(np.int64)
        fresh = np.arange(self.high_water, self.high_water + k - take, dtype=np.int64)
        self.n_free -= take
        self.high_water += k - take
        return np.concatenate([reused, fresh])

    def add(self, x, cid, kind, origin, log_r, energy, step, parent=None, undrawn=None,
            log_q=None, resid=None, eval_row=False, durable=True):
        """Write rows; returns the slots written, in input order. Rows with a non-finite x, log_r or energy are dropped
        and counted (nonfinite_rows). kind and origin: one value or one per row. log_q: required on on-policy rows,
        refused elsewhere; an on-policy row with a non-finite log_q is stored UNDRAWN. A finite row of a condition with
        no template raises before any write. best_energy is updated before any thin. Thins first when the rows do not
        fit."""
        if not self.seeded:
            raise CrystalStoreError('add() before seed()')
        x = self._states(x)
        n = x.shape[0]
        if n > self.headroom:
            raise CrystalStoreError(f'an add of {n} rows exceeds headroom {self.headroom}')
        cid = self._cids(cid, n)
        kind = self._vec(kind, n, torch.int8, 'kind')
        origin = self._vec(origin, n, torch.int8, 'origin')
        log_r = self._vec(log_r, n, torch.float32, 'log_r')
        energy = self._vec(energy, n, torch.float32, 'energy')
        step = int(step)
        if step < 0:
            raise CrystalStoreError(f'step must be >= 0, got {step}')
        if n and not bool(((kind == ANCHOR) | (kind == CHILD) | (kind == ONPOLICY)).all()):
            raise CrystalStoreError('add() takes ANCHOR, CHILD and ONPOLICY rows only')
        is_child, is_on = kind == CHILD, kind == ONPOLICY
        par = torch.full((n,), -1, dtype=torch.int64) if parent is None else self._vec(parent, n, torch.int64, 'parent')
        self._check_parents(par, cid, is_child)
        und = (torch.zeros(n, dtype=torch.bool) if undrawn is None else self._vec(undrawn, n, torch.bool, 'undrawn'))
        if log_q is None and bool(is_on.any()):
            raise CrystalStoreError('on-policy rows need log_q')
        lq = (torch.full((n,), math.nan, dtype=torch.float32) if log_q is None
              else self._vec(log_q, n, torch.float32, 'log_q'))
        rs = (torch.full((n,), math.nan, dtype=torch.float32) if resid is None
              else self._vec(resid, n, torch.float32, 'resid'))
        if bool((torch.isfinite(lq) & ~is_on).any()) or bool((torch.isfinite(rs) & ~is_on).any()):
            raise CrystalStoreError('log_q and resid are written on on-policy rows only')

        finite = torch.isfinite(x).all(1) & torch.isfinite(log_r) & torch.isfinite(energy)
        at = torch.nonzero(finite).flatten()
        k = at.numel()
        no_template = self.tmpl_of[cid[at]] < 0
        if bool(no_template.any()):
            raise CrystalStoreError(f'add: conditions {cid[at][no_template].unique()[:8].tolist()} have no template')
        new_anchors = int((kind[at] == ANCHOR).sum())
        if self.n_anchors + new_anchors + 2 * self.headroom > self.max_rows:
            raise CrystalStoreError(f'{self.n_anchors} + {new_anchors} anchors + 2 * headroom {self.headroom} exceed '
                                    f'max_rows {self.max_rows}')
        self.nonfinite_rows += n - k
        self.best_energy.scatter_reduce_(0, cid[finite], energy[finite], reduce='amin', include_self=True)
        shortfall = self.n_rows + k - self.max_rows
        if shortfall > 0:
            self.thin(shortfall)
            if self.n_rows + k > self.max_rows:
                raise CrystalStoreError(f'a thin left {self.n_rows} rows, too many for an add of {k}')
        if k == 0:
            return torch.zeros(0, dtype=torch.long)

        lq_at, rs_at, on_at = lq[at], rs[at], is_on[at]
        good_q = on_at & torch.isfinite(lq_at)
        und_at = und[at] | (on_at & ~good_q)
        slots = self._take_slots(k)
        s = torch.from_numpy(slots)
        self.x[s] = x[at]
        self.cid[s] = cid[at].to(torch.int32)
        self.kind[s] = kind[at]
        self.flags[s] = und_at.to(torch.uint8) * UNDRAWN | (EVAL if eval_row else 0)
        self.origin[s] = origin[at]
        self.parent[s] = par[at].to(torch.int32)
        self.log_r[s] = log_r[at]
        self.energy[s] = energy[at]
        nan = torch.full((k,), math.nan, dtype=torch.float32)
        self.log_q[s] = torch.where(good_q, lq_at, nan)
        self.log_q_birth[s] = torch.where(good_q, lq_at, nan)
        self.resid[s] = torch.where(good_q, rs_at, nan)
        self.q_step[s] = torch.where(good_q, torch.full((k,), step, dtype=torch.int32),
                                     torch.full((k,), -1, dtype=torch.int32))
        self.draws[s] = 0
        self.birth_step[s] = step
        kinds_at = kind[at]
        for j in range(N_KINDS):
            self._n_kind[j] += int((kinds_at == j).sum())
        self._n_undrawn += int(und_at.sum())
        self._index_add(slots[(~und_at).numpy()])
        if new_anchors:
            self._anchor_rows = None
        if durable:
            self.generation += 1
        return s

    def _check_parents(self, par, cid, is_child):
        """I2 for rows about to be written: a child's parent is a live anchor of its condition; any other row has -1."""
        if bool((par[~is_child] != -1).any()):
            raise CrystalStoreError('only a child carries a parent')
        p = par[is_child]
        if not p.numel():
            return
        if int(p.min()) < 0 or int(p.max()) >= self.high_water:
            raise CrystalStoreError('a child parent is not a live slot')
        if not bool((self.kind[p] == ANCHOR).all()):
            raise CrystalStoreError('a child parent is not an anchor')
        if not bool((self.cid[p].long() == cid[is_child]).all()):
            raise CrystalStoreError('a child and its parent anchor have different conditions')

    # ---- shrinking ------------------------------------------------------------------

    def _free_rows(self, rows):
        """Free live rows (unique int64 numpy slots): index, counters, free stack, epoch."""
        if not len(rows):
            return
        kinds = self.kind.numpy()[rows]
        undrawn = (self.flags.numpy()[rows] & UNDRAWN) != 0
        drawable = rows[~undrawn]
        if len(drawable):
            emptied = self.index.remove(drawable.astype(np.int32), self._pools_of(drawable))
            for k in range(N_KINDS):
                self.active[k].remove((emptied[emptied % N_KINDS == k] // N_KINDS).astype(np.int32))
                self._n_drawable[k] -= int((kinds[~undrawn] == k).sum())
        for k in range(N_KINDS):
            self._n_kind[k] -= int((kinds == k).sum())
        self._n_undrawn -= int(undrawn.sum())
        if bool((kinds == ANCHOR).any()):
            self._anchor_rows = None
        self.kind[torch.from_numpy(rows)] = FREE
        self._free[self.n_free:self.n_free + len(rows)] = rows.astype(np.int32)
        self.n_free += len(rows)
        self.epoch += 1
        self.generation += 1
        self.seed_intact = False

    def thin(self, need):
        """Free at least `need` rows, 50:50 between children and on-policy rows, uniformly within each; returns the rows
        freed. A block of THIN_BLOCK * max_rows is freed when available. Children never fall below headroom and anchors
        are never freed; a pool that runs short passes its deficit to the other. Undrawn rows count."""
        need = int(need)
        if need <= 0:
            return 0
        non_anchor = self.n_rows - self.n_anchors
        target = max(need, min(math.ceil(THIN_BLOCK * self.max_rows), non_anchor // 2))
        kinds = self.kind[:self.high_water].numpy()
        kids, pol = np.flatnonzero(kinds == CHILD), np.flatnonzero(kinds == ONPOLICY)
        kid_room = max(0, len(kids) - self.headroom)
        if kid_room + len(pol) < need:
            raise CrystalStoreError(f'a thin of {need} rows finds {kid_room} children above the floor and {len(pol)} '
                                    f'on-policy rows')
        take_c = min(kid_room, target // 2)
        take_p = min(len(pol), target - take_c)
        take_c = min(kid_room, target - take_p)
        gone = np.concatenate([self.rng.choice(kids, take_c, replace=False),
                               self.rng.choice(pol, take_p, replace=False)]).astype(np.int64)
        self._free_rows(gone)
        self.thinned_total += len(gone)
        return len(gone)

    def remove(self, rows):
        """Free the given live rows. Refuses an anchor that keeps a live child outside `rows`."""
        rows = self._rows(rows)
        if not rows.numel():
            return
        if int(rows.min()) < 0 or int(rows.max()) >= self.high_water:
            raise CrystalStoreError('remove: a row is not a slot of this store')
        if torch.unique(rows).numel() != rows.numel():
            raise CrystalStoreError('remove: a row is given twice')
        if bool((self.kind[rows] == FREE).any()):
            raise CrystalStoreError('remove: a row is already free')
        if bool((self.kind[rows] == ANCHOR).any()):
            going = torch.zeros(self.high_water, dtype=torch.bool)
            going[rows] = True
            kids = torch.nonzero(self.kind[:self.high_water] == CHILD).flatten()
            left = kids[~going[kids]]
            kept = left[going[self.parent[left].long()]]
            if kept.numel():
                raise CrystalStoreError(f'remove: {kept.numel()} live children keep an anchor being removed '
                                        f'(e.g. child slot {int(kept[0])})')
        self._free_rows(rows.numpy().astype(np.int64))

    def add_conditions(self, k, templates=None):
        """Append k conditions with empty pools; returns their ids. templates: k graphs, in id order. Nothing is
        renumbered. A failure leaves the store unchanged."""
        k = int(k)
        if k < 1:
            raise CrystalStoreError(f'add_conditions takes k >= 1, got {k}')
        if templates is not None and int(templates.num_graphs) != k:
            raise CrystalStoreError(f'{int(templates.num_graphs)} templates for {k} new conditions')
        first = self.n_cond
        grown = {name: torch.cat([getattr(self, name), torch.full((k,), fill, dtype=getattr(self, name).dtype)])
                 for name, fill in (('tmpl_of', -1), ('cond_sg', -1), ('cond_radius', math.nan), ('cond_rg', math.nan),
                                    ('best_energy', math.inf), ('window_floor', math.inf), ('_floor_written', math.inf))}
        stats = {name: torch.cat([t, _blank_stat(t, k)]) for name, t in self.cond_stats.items()}
        joined = self.templates
        if templates is not None:
            old = 0 if self.templates is None else int(self.templates.num_graphs)
            sg, radius, rg = self._template_fields(templates)
            joined = templates if self.templates is None else self._join_templates(self.templates, templates)
            grown['tmpl_of'][first:] = torch.arange(old, old + k, dtype=torch.int32)
            grown['cond_sg'][first:], grown['cond_radius'][first:], grown['cond_rg'][first:] = sg, radius, rg
        self.index.grow(N_KINDS * k)
        for s in self.active:
            s.grow(k)
        for name, t in grown.items():
            setattr(self, name, t)
        self.cond_stats, self.templates = stats, joined
        self.n_cond += k
        self.generation += 1
        self.seed_intact = False
        return torch.arange(first, first + k, dtype=torch.long)

    @staticmethod
    def _join_templates(old, new):
        from mxtaltools.dataset_utils.utils import collate_data_list
        return collate_data_list(old.to_data_list() + new.to_data_list())

    # ---- draws ----------------------------------------------------------------------

    def n_drawable(self, kind):
        """Drawable rows of `kind`."""
        return self._n_drawable[self._check_kind(kind)]

    def _conditions(self, kind, B):
        act = self.active[kind]
        if act.n == 0:
            raise CrystalStoreError(f'no drawable {KIND_NAMES[kind]} rows')
        return act.items[self.rng.integers(0, act.n, B)].astype(np.int64)

    def draw(self, kind, B):
        """B rows of `kind`: condition uniform over those with a drawable row, then a uniform row of it, with replacement
        at both levels."""
        kind, B = self._check_kind(kind), int(B)
        if B < 1:
            raise CrystalStoreError(f'draw size must be >= 1, got {B}')
        conds = self._conditions(kind, B)
        rows = self.index.uniform(conds * N_KINDS + kind, self.rng)
        return torch.from_numpy(rows.astype(np.int64))

    def draw_anchors(self, B, by='anchor'):
        """B drawable anchors: 'anchor' uniform over anchors, 'condition' condition-first."""
        B = int(B)
        if by == 'condition':
            return self.draw(ANCHOR, B)
        if by != 'anchor':
            raise CrystalStoreError(f"draw_anchors: by must be 'anchor' or 'condition', got {by!r}")
        if B < 1:
            raise CrystalStoreError(f'draw size must be >= 1, got {B}')
        if self._anchor_rows is None:
            pools = np.arange(self.n_cond, dtype=np.int64) * N_KINDS + ANCHOR
            self._anchor_rows = self.index.rows_of(pools[self.index.count[pools] > 0])
        if not len(self._anchor_rows):
            raise CrystalStoreError('no drawable anchor rows')
        return torch.from_numpy(self._anchor_rows[self.rng.integers(0, len(self._anchor_rows), B)].astype(np.int64))

    def draw_weighted(self, kind, B, weight_fn, step, mask=None):
        """(rows [B], is_correction [B]): condition uniform over those with an eligible row, then a row by weight_fn
        within the condition, with replacement at both levels. mask(store, rows) -> bool array restricts eligible rows
        (a subtype draw) and reads every pool of `kind`. is_correction = 1 / (n_c p), the uniform-within-condition law
        over the draw law, with n_c the condition's eligible rows: exactly 1 under the uniform rule.

        A non-finite or negative weight on an eligible row raises, and so does a condition whose weight sum is not
        finite. A condition whose eligible weights are all zero draws uniformly among its eligible rows."""
        kind, B = self._check_kind(kind), int(B)
        if B < 1:
            raise CrystalStoreError(f'draw size must be >= 1, got {B}')
        if weight_fn is uniform and mask is None:
            return self.draw(kind, B), torch.ones(B, dtype=torch.float32)
        rule = getattr(weight_fn, '__name__', repr(weight_fn))
        if mask is None:
            conds = self._conditions(kind, B)
        else:
            act = self.active[kind].members().astype(np.int64)
            if not len(act):
                raise CrystalStoreError(f'no drawable {KIND_NAMES[kind]} rows')
            pools = act * N_KINDS + kind
            all_rows = self.index.rows_of(pools)
            ok = np.asarray(mask(self, torch.from_numpy(all_rows.astype(np.int64)))).astype(bool)
            hit = np.bincount(np.repeat(np.arange(len(act)), self.index.count[pools])[ok], minlength=len(act)) > 0
            if not hit.any():
                raise CrystalStoreError(f'no {KIND_NAMES[kind]} rows pass the mask')
            elig = act[hit]
            conds = elig[self.rng.integers(0, len(elig), B)]
        uc, inv = np.unique(conds, return_inverse=True)
        pools = uc * N_KINDS + kind
        lens = self.index.count[pools]
        flat = self.index.rows_of(pools).astype(np.int64)
        group = np.repeat(np.arange(len(uc)), lens)
        ok = (np.ones(len(flat), dtype=bool) if mask is None
              else np.asarray(mask(self, torch.from_numpy(flat))).astype(bool))
        got = weight_fn(self, torch.from_numpy(flat), step)
        if got is None:
            w = ok.astype(np.float64)
        else:
            w = torch.as_tensor(got).detach().cpu().double().reshape(-1).numpy().copy()
            if len(w) != len(flat):
                raise CrystalStoreError(f'weight rule {rule!r} returned {len(w)} weights for {len(flat)} rows')
            w[~ok] = 0.0
            bad = ~np.isfinite(w) | (w < 0)
            if bad.any():
                at = int(np.flatnonzero(bad)[0])
                raise CrystalStoreError(f'weight rule {rule!r} gave weight {w[at]} to slot {int(flat[at])}')
        n_c = np.bincount(group, weights=ok.astype(np.float64), minlength=len(uc))
        tot = np.bincount(group, weights=w, minlength=len(uc))
        if not np.isfinite(tot).all():
            at = int(np.flatnonzero(~np.isfinite(tot))[0])
            raise CrystalStoreError(f'weight rule {rule!r} gave condition {int(uc[at])} a weight sum that is not finite')
        zero = tot <= 0
        if zero.any():
            fill = zero[group] & ok
            w[fill] = 1.0
            tot = np.bincount(group, weights=w, minlength=len(uc))
        ends = np.cumsum(lens)
        firsts = ends - lens
        nw = w / tot[group]
        cum = np.cumsum(nw)
        base = np.concatenate([[0.0], cum[ends[:-1] - 1]])
        top = cum[ends - 1]
        last = np.maximum.reduceat(np.where(nw > 0, np.arange(len(flat)), -1), firsts)
        target = base[inv] + self.rng.random(B) * (top[inv] - base[inv])
        pick = np.minimum(np.searchsorted(cum, target, side='right'), last[inv])
        g = group[pick]
        correction = tot[g] / w[pick] / n_c[g]
        return torch.from_numpy(flat[pick]), torch.from_numpy(correction).to(torch.float32)

    def mol_batch(self, cids, repeats=1):
        """The template of each condition, tiled `repeats` times per entry; never oriented."""
        cids = self._cids(cids)
        t = self.tmpl_of[cids]
        if bool((t < 0).any()):
            raise CrystalStoreError(f'conditions {cids[t < 0].unique()[:8].tolist()} have no template')
        t = t.long()
        if int(repeats) > 1:
            t = t.repeat_interleave(int(repeats))
        return self.templates.subsample_new_batch(t)

    # ---- per-row writes -------------------------------------------------------------

    def touch(self, rows, epoch, log_q, resid, step):
        """After a training draw: draws += occurrences; log_q, resid and q_step take the last occurrence with a finite
        log_q and resid. Returns the number of rows whose log_q was written: 0 under another epoch, which writes nothing.
        Refuses anchors. Never writes log_q_birth."""
        if epoch != self.epoch:
            return 0
        rows = self._rows(rows)
        n = rows.numel()
        if n == 0:
            return 0
        log_q = self._vec(log_q, n, torch.float32, 'log_q')
        resid = self._vec(resid, n, torch.float32, 'resid')
        if int(rows.min()) < 0 or int(rows.max()) >= self.high_water:
            raise CrystalStoreError('touch: a row is not a slot of this store')
        kinds = self.kind[rows]
        if bool((kinds == FREE).any()):
            raise CrystalStoreError('touch: a row is free')
        if bool((kinds == ANCHOR).any()):
            raise CrystalStoreError('touch: an anchor is never written after its seed')
        u, c = torch.unique(rows, return_counts=True)
        self.draws[u] += c.to(torch.int32)
        ok = torch.nonzero(torch.isfinite(log_q) & torch.isfinite(resid)).flatten()
        written = 0
        if ok.numel():
            r = rows[ok].numpy()
            uq, first = np.unique(r[::-1], return_index=True)
            last = ok[torch.from_numpy(len(r) - 1 - first)]
            s = torch.from_numpy(uq)
            self.log_q[s] = log_q[last]
            self.resid[s] = resid[last]
            self.q_step[s] = int(step)
            written = len(uq)
        self.generation += 1
        return written

    def invalidate_q_after(self, step):
        """Undo log q writes after `step` on live rows. resid becomes NaN. A row with a finite log_q_birth returns to it
        with q_step = birth_step; any other row gets NaN and -1. Returns how many rows changed."""
        hw = self.high_water
        at_admission = (self.q_step[:hw] == self.birth_step[:hw]) & torch.isnan(self.resid[:hw])
        hit = torch.nonzero((self.q_step[:hw] > int(step)) & (self.kind[:hw] != FREE) & ~at_admission).flatten()
        if hit.numel():
            birth = self.log_q_birth[hit]
            self.log_q[hit] = birth
            self.resid[hit] = math.nan
            self.q_step[hit] = torch.where(torch.isfinite(birth), self.birth_step[hit],
                                           torch.full((hit.numel(),), -1, dtype=torch.int32))
            self.generation += 1
        return hit.numel()

    # ---- monitoring -----------------------------------------------------------------

    def bytes_per_row(self):
        """Bytes per live row of the occupied columns and of the index in use."""
        n = max(self.n_rows, 1)
        col = sum(getattr(self, name)[:0].element_size() * (self.dim if name == 'x' else 1) for name, _ in _COLUMNS)
        index = 4 * self.high_water + 4 * self.index.tail
        tables = 3 * 8 * len(self.index.count) + sum(4 * 2 * s.items.size for s in self.active)
        return col * self.high_water / n + (index + tables) / n

    def stats(self, prefix='store/'):
        """O(1) counters."""
        out = dict(rows=self.n_rows, anchors=self._n_kind[ANCHOR], children=self._n_kind[CHILD],
                   onpolicy=self._n_kind[ONPOLICY], undrawn=self._n_undrawn,
                   drawable_anchors=self._n_drawable[ANCHOR], drawable_children=self._n_drawable[CHILD],
                   drawable_onpolicy=self._n_drawable[ONPOLICY], thinned_total=self.thinned_total,
                   nonfinite_rows=self.nonfinite_rows, epoch=self.epoch, generation=self.generation,
                   index_compactions=self.index.compactions, conditions=self.n_cond)
        return {prefix + k: v for k, v in out.items()}

    def check(self):
        """Verify I1, I2, I4-I8, I11, I12 and the counters over the whole store, O(n). Raises on the first failure."""
        def need(ok, what):
            if not ok:
                raise CrystalStoreError(f'store check failed: {what}')

        hw = self.high_water
        for name in _TABLES + ('_floor_written',):
            need(len(getattr(self, name)) == self.n_cond, f'table {name} does not have n_cond entries')
        for name, t in self.cond_stats.items():
            need(t.dim() >= 1 and len(t) == self.n_cond, f'table {name} does not have n_cond entries')
        need(all(s.pos.size == self.n_cond for s in self.active), 'an active set does not have n_cond entries')
        need(0 <= self.n_free <= hw <= self.max_rows, 'I5: free stack or high water out of range')
        kinds = self.kind[:hw].numpy()
        flags = self.flags[:hw].numpy()
        cids = self.cid[:hw].numpy().astype(np.int64)
        parent = self.parent[:hw].numpy().astype(np.int64)
        live = kinds != FREE
        need(bool(np.isin(kinds, (ANCHOR, CHILD, ONPOLICY, FREE)).all()), 'I1: unknown kind')
        free = self._free[:self.n_free]
        need(len(np.unique(free)) == self.n_free, 'I5: free stack holds a slot twice')
        need(self.n_free == 0 or (int(free.min()) >= 0 and int(free.max()) < hw), 'I5: free slot out of range')
        need(bool((kinds[free] == FREE).all()), 'I1: a slot on the free stack is live')
        need(int((~live).sum()) == self.n_free, 'I5: a FREE slot is not on the free stack')
        need(self.n_rows == int(live.sum()), 'I5: n_rows')
        for k in range(N_KINDS):
            need(self._n_kind[k] == int((kinds == k).sum()), f'count of {KIND_NAMES[k]} rows')
        undrawn = (flags & UNDRAWN) != 0
        need(self._n_undrawn == int((live & undrawn).sum()), 'count of undrawn rows')
        need(bool(((cids[live] >= 0) & (cids[live] < self.n_cond)).all()), 'a live row has a condition id out of range')

        child = np.flatnonzero(kinds == CHILD)
        p = parent[child]
        need(not len(p) or (int(p.min()) >= 0 and int(p.max()) < hw), 'I2: a child parent is out of range')
        need(bool((kinds[p] == ANCHOR).all()), 'I2: a child parent is not an anchor')
        need(bool((cids[p] == cids[child]).all()), 'I2: a child and its parent differ in condition')
        need(bool((parent[live & (kinds != CHILD)] == -1).all()), 'I2: a non-child carries a parent')

        drawable = np.flatnonzero(live & ~undrawn)
        pools = cids[drawable] * N_KINDS + kinds[drawable].astype(np.int64)
        P = N_KINDS * self.n_cond
        need(len(self.index.count) == P, 'I4: pool count')
        need(bool((np.bincount(pools, minlength=P) == self.index.count).all()), 'I4: pool membership')
        need(bool((self.index.where[drawable] < self.index.count[pools]).all())
             and bool((self.index.where[drawable] >= 0).all()), 'I4: a position outside its pool')
        need(bool((self.index.arena[self.index.start[pools] + self.index.where[drawable]] == drawable).all()),
             'I4: where does not invert the arena')
        need(bool((self.index.start + self.index.cap <= self.index.tail).all())
             and self.index.tail <= self.index.arena.size, 'I4: a pool beyond the arena tail')
        for k in range(N_KINDS):
            pc = self.index.count[np.arange(self.n_cond, dtype=np.int64) * N_KINDS + k]
            need(self._n_drawable[k] == int(pc.sum()), f'count of drawable {KIND_NAMES[k]} rows')
            mem = self.active[k].members()
            need(len(np.unique(mem)) == len(mem) and bool((pc[mem] > 0).all()) and len(mem) == int((pc > 0).sum()),
                 f'active set of {KIND_NAMES[k]}')
            need(bool((self.active[k].pos[mem] == np.arange(len(mem))).all()), f'active positions of {KIND_NAMES[k]}')

        e = self.energy[:hw]
        be = self.best_energy[self.cid[:hw].long().clamp_min(0)]
        lv = torch.from_numpy(live)
        need(bool((be[lv] <= e[lv]).all()), 'I6: a live row is below its condition minimum')
        same = (self.window_floor == self._floor_written) | (torch.isnan(self.window_floor)
                                                             & torch.isnan(self._floor_written))
        need(bool(same.all()), 'I7: window_floor changed after seed')
        need(bool((self.tmpl_of[self.cid[:hw][lv].long()] >= 0).all()), 'I8: a live row has no template')
        need(self.n_anchors + 2 * self.headroom <= self.max_rows, 'I11: capacity')
        not_on = torch.from_numpy(live & (kinds != ONPOLICY))
        need(bool(torch.isnan(self.log_q_birth[:hw][not_on]).all()), 'I12: log_q_birth on a non on-policy row')
        drawn_on = torch.from_numpy(live & (kinds == ONPOLICY) & ~undrawn)
        need(bool(torch.isfinite(self.log_q_birth[:hw][drawn_on]).all()),
             'I12: a drawable on-policy row has no finite log_q_birth')
        need(bool(torch.isfinite(self.log_q[:hw][drawn_on]).all()), 'I12: a drawable on-policy row has no finite log_q')

    # ---- persistence ----------------------------------------------------------------

    def seed_digest(self):
        """What identifies the seed rows: 'exact', a sha1 of the discrete seed columns (_SEED_EXACT), the template map
        and the template atom types; 'close', [n_cond, dim + 2 + 3] float64, each condition's mean x, log_r and energy
        over its seed rows (NaN without one) and its _SEED_CLOSE_TABLES entries. x, log_r and energy are recomputed at
        every launch on that launch's device, so they are compared within SEED_CLOSE_RTOL and SEED_CLOSE_ATOL. Cached:
        none of this is written while seed_intact holds."""
        if self._seed_digest is None:
            n = self.n_seed_kept
            h = hashlib.sha1(repr((self.dim, self.n_cond, n)).encode())
            for name in _SEED_EXACT:
                h.update(getattr(self, name)[:n].contiguous().numpy())
            for name in ('tmpl_of', 'cond_sg'):
                h.update(getattr(self, name).contiguous().numpy())
            z = getattr(self.templates, 'z', None)
            if z is not None:
                h.update(z.detach().cpu().contiguous().numpy())
            sums = torch.zeros((self.n_cond, self.dim + 2), dtype=torch.float64)
            counts = torch.zeros(self.n_cond, dtype=torch.float64)
            for lo in range(0, n, _DIGEST_CHUNK):
                hi = min(lo + _DIGEST_CHUNK, n)
                c = self.cid[lo:hi].long()
                sums.index_add_(0, c, torch.cat([self.x[lo:hi], self.log_r[lo:hi, None], self.energy[lo:hi, None]],
                                                1).double())
                counts.index_add_(0, c, torch.ones(hi - lo, dtype=torch.float64))
            tables = torch.stack([getattr(self, name).double() for name in _SEED_CLOSE_TABLES], 1)
            self._seed_digest = {'exact': h.hexdigest()[:16], 'close': torch.cat([sums / counts[:, None], tables], 1)}
        return self._seed_digest

    def _digest_differs(self, digest):
        """How `digest` differs from this store's seed_digest(), or None."""
        mine = self.seed_digest()
        if digest['exact'] != mine['exact']:
            return f"seed rows: state digest {digest['exact']}, this store {mine['exact']}"
        a, b = digest['close'], mine['close']
        if a.shape != b.shape:
            return f'seed digest shape {tuple(a.shape)}, this store {tuple(b.shape)}'
        far = ~torch.isclose(a, b, rtol=SEED_CLOSE_RTOL, atol=SEED_CLOSE_ATOL, equal_nan=True)
        if bool(far.any()):
            return (f'seed digest: the seed values of {int(far.any(1).sum())} conditions moved, largest change '
                    f'{float((a - b)[far].abs().nan_to_num(math.inf).max()):.4g}')
        return None

    def state_dict(self, stamp=None, compact=False):
        """Clones of the occupied column prefix, the tables, the free stack, the index order and the generator.

        compact=True on a store with seed_intact leaves out what a relaunch's seed() rebuilds: `columns` then holds only
        the rows appended after the seed, `seed_columns` the seed rows' _SEED_MUTABLE columns, `seed_digest` identifies
        the rest, and `templates` is None. On any other store it has no effect."""
        hw = self.high_water
        compact = bool(compact) and self.seed_intact
        first = self.n_seed_kept if compact else 0
        pools = np.arange(len(self.index.count), dtype=np.int64)
        state = {'format_version': FORMAT_VERSION, 'stamp': stamp, 'dim': self.dim, 'max_rows': self.max_rows,
                 'headroom': self.headroom, 'n_cond': self.n_cond, 'base_seed': self.base_seed,
                 'columns': {name: getattr(self, name)[first:hw].clone() for name, _ in _COLUMNS},
                 'tables': {name: getattr(self, name).clone() for name in _TABLES},
                 'cond_stats': {k: v.clone() for k, v in self.cond_stats.items()},
                 'free': torch.from_numpy(self._free[:self.n_free].copy()),
                 'index_count': torch.from_numpy(self.index.count.copy()),
                 'index_start': torch.from_numpy(self.index.start.copy()),
                 'index_cap': torch.from_numpy(self.index.cap.copy()),
                 'index_tail': self.index.tail, 'index_compactions': self.index.compactions,
                 'index_rows': torch.from_numpy(self.index.rows_of(pools)),
                 'active': [torch.from_numpy(s.members().copy()) for s in self.active],
                 'counters': {name: getattr(self, name) for name in _COUNTERS},
                 'n_kind': list(self._n_kind), 'n_drawable': list(self._n_drawable), 'n_undrawn': self._n_undrawn,
                 'rng_state': self.rng.bit_generator.state, 'templates': None if compact else self.templates,
                 'seed_digest': self.seed_digest() if compact else None}
        if compact:
            state['seed_columns'] = {name: getattr(self, name)[:first].clone() for name in _SEED_MUTABLE}
        return state

    def _refusal(self, state):
        """Why state_dict() output `state` cannot be applied to this store, or None."""
        fv = state.get('format_version')
        if fv != FORMAT_VERSION:
            return f'crystal store format_version {fv}, this code reads {FORMAT_VERSION}'
        for name in ('dim', 'max_rows', 'headroom'):
            if state.get(name) != getattr(self, name):
                return f'crystal store {name}: state {state.get(name)}, this store {getattr(self, name)}'
        if state['n_cond'] < self.n_cond:
            return f"crystal store n_cond: state {state['n_cond']}, this store {self.n_cond}"
        hw = int(state['counters']['high_water'])
        if hw > self.max_rows:
            return f'crystal store high_water {hw} above max_rows {self.max_rows}'
        digest = state.get('seed_digest')
        if digest is not None:
            if not (self.seeded and self.seed_intact and self.high_water == self.n_seed_kept):
                return 'a compact crystal store state applies only to a freshly seeded store'
            differs = self._digest_differs(digest)
            if differs is not None:
                return f'crystal store {differs} (the prior file, its load window, the seed or the row limit changed)'
        return None

    def state_mismatch(self, state, stamp):
        """Why `state` cannot be restored into this store under `stamp`, or None."""
        diff = stamp_diff(state.get('stamp'), stamp)
        if diff:
            return f'crystal store stamp differs on {diff}'
        return self._refusal(state)

    def apply_state(self, state):
        """Restore state_dict() into this store. dim, max_rows and headroom must match. A compact state needs a store
        freshly seeded to the same seed digest."""
        why = self._refusal(state)
        if why is not None:
            raise CrystalStoreError(why)
        if state['n_cond'] > self.n_cond:
            self.add_conditions(state['n_cond'] - self.n_cond)
        hw = int(state['counters']['high_water'])
        compact = state.get('seed_digest') is not None
        first = int(state['counters']['n_seed_kept']) if compact else 0
        for name, _ in _COLUMNS:
            getattr(self, name)[first:hw] = state['columns'][name]
        if compact:
            for name in _SEED_MUTABLE:
                getattr(self, name)[:first] = state['seed_columns'][name]
        seed_best = self.best_energy
        for name in _TABLES:
            setattr(self, name, state['tables'][name].clone())
        if compact:     # I6 over the seed energies this launch recomputed
            self.best_energy = torch.minimum(self.best_energy, seed_best)
        self._floor_written = self.window_floor.clone()
        self.cond_stats = {k: v.clone() for k, v in state['cond_stats'].items()}
        self.base_seed = int(state['base_seed'])
        for name in _COUNTERS:
            setattr(self, name, state['counters'][name])
        free = state['free'].numpy()
        self._free[:len(free)] = free
        self.index.load(state['index_count'].numpy(), state['index_start'].numpy(), state['index_cap'].numpy(),
                        state['index_tail'], state['index_compactions'], state['index_rows'].numpy())
        for s, members in zip(self.active, state['active']):
            s.load(members.numpy())
        self._n_kind, self._n_drawable = list(state['n_kind']), list(state['n_drawable'])
        self._n_undrawn = state['n_undrawn']
        self._anchor_rows = None
        if not compact:
            self.templates = state['templates']
            self._seed_digest = None
        self.rng = np.random.Generator(np.random.PCG64())
        self.rng.bit_generator.state = state['rng_state']
