"""One flat row store for the NF route (docs/design/unified_store.md).

Rows are plain tensors of equal length n, rebuilt by `index_select` / `cat`:

    cid        int64        condition id
    x          f32 [n,dim]  chart state the policy scores
    log_r      f32          log R at admission, run currency, referenced
    log_q      f32          log q at the last refresh or touch; NaN = never scored
    q_step     int64        training step of that write; -1 = never scored
    tag        int8         TAG_ANCHOR / TAG_CHILD / TAG_ONPOLICY / TAG_GENERATED
    protected  bool         never evicted (seed rows; records until demoted)
    held_out   bool         never drawn for training
    born_eval  int32        eval index at admission

Row order: [0, n_mle) data rows not held out; [n_mle, n_seed) held-out seed rows;
[n_seed, n) admitted rows. Seed rows are protected and never demoted, and
compaction preserves order, so the layout holds for the store's life.

The residual is r = log_r - log_q - L(c). L is a [n_cond] level table supplied
live by the caller on every draw, admission, refresh and stats call; it is never
stored per row. NaN in the table means "no level": rows of that condition have a
non-finite r, are not drawable for TB and are not admitted.

This module imports torch, math and time only.
"""
import math
import time

import torch

TAG_ANCHOR, TAG_CHILD, TAG_ONPOLICY, TAG_GENERATED = 0, 1, 2, 3
TAG_NAMES = ('anchor', 'child', 'onpolicy', 'generated')
DATA_TAGS = (TAG_ANCHOR, TAG_CHILD)
FORMAT_VERSION = 1

_SETTINGS = ('bar', 'eps', 'cap_per_condition', 'max_rows', 'records_kept', 'kappa',
             'floor_frac', 'uniform_frac', 'admit_reward_min')
_COLUMNS = ('cid', 'x', 'log_r', 'log_q', 'q_step', 'tag', 'protected', 'held_out', 'born_eval')
_COND_TOL = 1e-6
_FLOOR_MIN = 1e-12


class RowStoreError(ValueError):
    """A call that would break a store invariant."""


def _within_group_rank(sorted_keys):
    """Position of each element inside its run of equal keys (keys already grouped)."""
    n = sorted_keys.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.int64, device=sorted_keys.device)
    change = torch.ones(n, dtype=torch.bool, device=sorted_keys.device)
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    pos = torch.arange(n, device=sorted_keys.device)
    starts = pos[change]
    return pos - starts[torch.cumsum(change.long(), 0) - 1]


def _median(t):
    """Lower median as a float; NaN when empty."""
    return float(t.median()) if t.numel() else math.nan


def _real(v):
    """True for a finite real number (bool, str and None excluded)."""
    if v is None or isinstance(v, (bool, str, bytes)):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _whole(v):
    return _real(v) and int(v) == v


def _check_settings(s):
    if not (_real(s['kappa']) and s['kappa'] >= 0):
        raise RowStoreError(f"kappa must be >= 0, got {s['kappa']}")
    for k in ('floor_frac', 'uniform_frac'):
        if not (_real(s[k]) and 0.0 <= s[k] <= 1.0):
            raise RowStoreError(f'{k} must lie in [0, 1], got {s[k]}')
    if not (_real(s['bar']) and s['bar'] >= 0):
        raise RowStoreError(f"bar must be >= 0, got {s['bar']}")
    if not (_real(s['eps']) and s['eps'] > 0):
        raise RowStoreError(f"eps must be > 0, got {s['eps']}")
    for k in ('cap_per_condition', 'max_rows'):
        if not (_whole(s[k]) and s[k] >= 0):
            raise RowStoreError(f'{k} must be an integer >= 0, got {s[k]}')
    rk = s['records_kept']
    if rk is not None and not (_whole(rk) and rk >= 1):
        raise RowStoreError(f'records_kept must be an integer >= 1 or None, got {rk}')
    arm = s['admit_reward_min']
    if arm is not None and not _real(arm):
        raise RowStoreError(f'admit_reward_min must be a finite number or None, got {arm}')


class RowStore:
    def __init__(self, dim, n_cond, cond_dim, device, *, bar, eps, cap_per_condition, max_rows,
                 records_kept, kappa, floor_frac, uniform_frac, admit_reward_min=None):
        self.dim, self.n_cond, self.cond_dim = int(dim), int(n_cond), int(cond_dim)
        if self.n_cond < 1:
            raise RowStoreError(f'n_cond must be >= 1, got {n_cond}')
        self.device = torch.device(device)
        d = self.device
        self.cid = torch.zeros(0, dtype=torch.int64, device=d)
        self.x = torch.zeros(0, self.dim, dtype=torch.float32, device=d)
        self.log_r = torch.zeros(0, dtype=torch.float32, device=d)
        self.log_q = torch.zeros(0, dtype=torch.float32, device=d)
        self.q_step = torch.zeros(0, dtype=torch.int64, device=d)
        self.tag = torch.zeros(0, dtype=torch.int8, device=d)
        self.protected = torch.zeros(0, dtype=torch.bool, device=d)
        self.held_out = torch.zeros(0, dtype=torch.bool, device=d)
        self.born_eval = torch.zeros(0, dtype=torch.int32, device=d)
        self.cond_table = torch.full((self.n_cond, self.cond_dim), math.nan, dtype=torch.float32, device=d)
        self.best_log_r = torch.full((self.n_cond,), -math.inf, dtype=torch.float32, device=d)
        self.n_mle = 0
        self.n_seed = 0
        self.eval_idx = 0
        self.version = 0
        self.last_refresh = None
        self._reset_window()
        for k in _SETTINGS:
            setattr(self, k, None)
        self.configure(bar=bar, eps=eps, cap_per_condition=cap_per_condition, max_rows=max_rows,
                       records_kept=records_kept, kappa=kappa, floor_frac=floor_frac,
                       uniform_frac=uniform_frac, admit_reward_min=admit_reward_min)

    # ---- settings and shape helpers -------------------------------------------------

    def configure(self, **settings):
        """Set any of the store settings; the merged set is validated before anything is written."""
        unknown = sorted(set(settings) - set(_SETTINGS))
        if unknown:
            raise RowStoreError(f'unknown row store settings {unknown}; known: {list(_SETTINGS)}')
        merged = {k: getattr(self, k) for k in _SETTINGS}
        merged.update(settings)
        _check_settings(merged)
        for k, v in settings.items():
            if k in ('cap_per_condition', 'max_rows') or (k == 'records_kept' and v is not None):
                v = int(v)
            setattr(self, k, v)

    def __len__(self):
        return self.cid.numel()

    def _reset_window(self):
        self._win = {'admitted': 0, 'admitted_records': 0, 'rejected_floor': 0,
                     'evicted_cap': 0, 'evicted_global': 0, 'demoted': 0}
        self._win_admitted_r = []
        self._win_evicted_r = []
        self._win_evicted_age = []

    def _vec(self, v, n, dtype, name):
        t = torch.as_tensor(v, device=self.device)
        t = t.detach().to(dtype)
        if t.dim() == 0:
            t = t.expand(n).clone()
        t = t.reshape(-1)
        if t.numel() != n:
            raise RowStoreError(f'{name} has {t.numel()} entries, expected {n}')
        return t

    def _cids(self, cid):
        cid = torch.as_tensor(cid, device=self.device).detach().to(torch.int64).reshape(-1)
        if cid.numel() and (int(cid.min()) < 0 or int(cid.max()) >= self.n_cond):
            raise RowStoreError(f'condition id outside [0, {self.n_cond})')
        return cid

    def _states(self, x, n):
        x = torch.as_tensor(x, device=self.device).detach().to(torch.float32)
        if x.dim() != 2 or x.shape[0] != n or x.shape[1] != self.dim:
            raise RowStoreError(f'x has shape {tuple(x.shape)}, expected ({n}, {self.dim})')
        return x

    def _level(self, level_table):
        lv = torch.as_tensor(level_table, device=self.device).detach().to(torch.float32).reshape(-1)
        if lv.numel() != self.n_cond:
            raise RowStoreError(f'level table has {lv.numel()} entries, expected n_cond={self.n_cond}')
        return lv

    def _set_columns(self, cols):
        for c in _COLUMNS:
            setattr(self, c, cols[c])

    # ---- seeding and conditions -----------------------------------------------------

    def seed(self, cid, x, log_r, tag, held_out):
        """Load the data rows once, into an empty store. Every row is protected; held-out rows go last."""
        if len(self):
            raise RowStoreError('seed() is only allowed on an empty store')
        cid = self._cids(cid)
        n = cid.numel()
        x = self._states(x, n)
        log_r = self._vec(log_r, n, torch.float32, 'log_r')
        tag = self._vec(tag, n, torch.int8, 'tag')
        held_out = self._vec(held_out, n, torch.bool, 'held_out')
        is_data = (tag == TAG_ANCHOR) | (tag == TAG_CHILD)
        if not bool(is_data.all()):
            raise RowStoreError('seed() takes data tags only (anchor, child)')
        order = torch.argsort(held_out.to(torch.int8), stable=True)
        self._set_columns({
            'cid': cid[order], 'x': x[order], 'log_r': log_r[order],
            'log_q': torch.full((n,), math.nan, dtype=torch.float32, device=self.device),
            'q_step': torch.full((n,), -1, dtype=torch.int64, device=self.device),
            'tag': tag[order],
            'protected': torch.ones(n, dtype=torch.bool, device=self.device),
            'held_out': held_out[order],
            'born_eval': torch.zeros(n, dtype=torch.int32, device=self.device)})
        self.n_mle = int((~held_out).sum())
        self.n_seed = n
        keep = (~self.held_out) & torch.isfinite(self.log_r)
        self.best_log_r = self.best_log_r.scatter_reduce(
            0, self.cid[keep], self.log_r[keep], reduce='amax', include_self=True)

    def set_conditions(self, cid, cond):
        """Fill unset rows of cond_table; raise when a condition id maps to two different conditions."""
        if cond is None:
            return
        cid = self._cids(cid)
        n = cid.numel()
        if n == 0:
            return
        cond = torch.as_tensor(cond, device=self.device).detach().to(torch.float32).reshape(n, self.cond_dim)
        if not bool(torch.isfinite(cond).all()):
            raise RowStoreError('non-finite condition passed to set_conditions')
        u, inv = torch.unique(cid, return_inverse=True)
        first = torch.full((u.numel(),), n, dtype=torch.int64, device=self.device).scatter_reduce(
            0, inv, torch.arange(n, device=self.device), reduce='amin', include_self=True)
        rep = cond[first]
        if self.cond_dim == 0:
            return
        if float((cond - rep[inv]).abs().max()) > _COND_TOL:
            raise RowStoreError('one condition id carries two different conditions within one call')
        cur = self.cond_table[u]
        unset = torch.isnan(cur).any(1)
        is_set = ~unset
        if bool(is_set.any()) and float((cur[is_set] - rep[is_set]).abs().max()) > _COND_TOL:
            bad = u[is_set][(cur[is_set] - rep[is_set]).abs().max(1).values > _COND_TOL]
            raise RowStoreError(f'condition ids {bad.tolist()[:8]} differ from the stored condition')
        self.cond_table[u[unset]] = rep[unset]

    # ---- residuals and draws --------------------------------------------------------

    def resid(self, level_table, idx=None):
        """r = log_r - log_q - L(cid), against the supplied live level table."""
        lv = self._level(level_table)
        if idx is None:
            return self.log_r - self.log_q - lv[self.cid]
        idx = torch.as_tensor(idx, device=self.device).to(torch.int64)
        return self.log_r[idx] - self.log_q[idx] - lv[self.cid[idx]]

    def _drawable(self, level_table):
        """(drawable mask E, |r| for every row, non-held-out mask)."""
        r = self.resid(level_table)
        elig = ~self.held_out
        return elig & torch.isfinite(r), r.abs(), elig

    def n_drawable(self, level_table):
        """n_E: rows not held out with a finite residual."""
        return int(self._drawable(level_table)[0].sum())

    def draw_probs(self, level_table):
        """(rows of E, q over those rows in float64), or None when E is empty."""
        E, s_all, _ = self._drawable(level_table)
        rows = torch.nonzero(E).squeeze(1)
        n_E = rows.numel()
        if n_E == 0:
            return None
        s = s_all[rows].double()
        f = max(self.floor_frac * float(s.median()), _FLOOR_MIN)
        sp = s.clamp_min(f)
        P = (sp / sp.max()).pow(float(self.kappa))
        P = P / P.sum()
        u = float(self.uniform_frac)
        return rows, (1.0 - u) * P + u / n_E

    def draw_tb(self, B, level_table, generator=None):
        """B i.i.d. rows by priority with IS weights w = (1/n_E)/q; None when nothing is drawable."""
        B = int(B)
        if B < 1:
            raise RowStoreError(f'draw size must be >= 1, got {B}')
        got = self.draw_probs(level_table)
        if got is None:
            return None
        rows, q = got
        n_E = rows.numel()
        dev = generator.device if generator is not None else q.device
        pick = torch.multinomial(q.to(dev), B, replacement=True, generator=generator).to(self.device)
        w = (1.0 / n_E) / q[pick]
        n_elig = int((~self.held_out).sum())
        stats = {'replay/is_ess_frac': float(w.sum() ** 2 / (B * (w * w).sum())),
                 'replay/is_w_max_ratio': float(w.max() / w.mean()),
                 'replay/is_elig_frac': n_E / n_elig}
        return rows[pick], w, stats

    def draw_mle(self, B, generator=None):
        """B i.i.d. rows uniform over [0, n_mle): data tags, not held out."""
        if self.n_mle == 0:
            raise RowStoreError('draw_mle on a store with no drawable data rows')
        B = int(B)
        if B < 1:
            raise RowStoreError(f'draw size must be >= 1, got {B}')
        dev = generator.device if generator is not None else self.device
        return torch.randint(self.n_mle, (B,), generator=generator, device=dev).to(self.device)

    def touch(self, idx, version, log_q, step):
        """Write log q of drawn rows; refuse indices from another store version."""
        if version != self.version:
            raise RowStoreError(f'touch with indices from store version {version}, store is at {self.version}')
        idx = torch.as_tensor(idx, device=self.device).to(torch.int64).reshape(-1)
        log_q = self._vec(log_q, idx.numel(), torch.float32, 'log_q')
        self.log_q[idx] = log_q
        self.q_step[idx] = int(step)

    def refresh(self, log_q_fn, level_table=None, step=0, chunk=65536):
        """Recompute log q of every row, `chunk` rows per call of log_q_fn(x, cid); increments eval_idx."""
        chunk = int(chunk)
        if chunk < 1:
            raise RowStoreError(f'refresh chunk must be >= 1, got {chunk}')
        t0 = time.perf_counter()
        n = len(self)
        with torch.no_grad():
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                lq = log_q_fn(self.x[s:e], self.cid[s:e])
                self.log_q[s:e] = self._vec(lq, e - s, torch.float32, 'refreshed log_q')
        self.q_step.fill_(int(step))
        self.eval_idx += 1
        nonfinite = int((~torch.isfinite(self.log_q)).sum())
        out = {'rows': n, 'seconds': time.perf_counter() - t0, 'nonfinite': nonfinite}
        if level_table is not None:
            E, s_all, _ = self._drawable(level_table)
            out['resid_med'] = _median(s_all[E])
        self.last_refresh = out
        return out

    # ---- admission, demotion, eviction ----------------------------------------------

    def admit(self, cid, x, log_r, log_q, ok, tag, step, level_table, cond, generator=None):
        """Admit candidates with |r| >= bar or a strict per-condition record, then demote and evict.

        ok: bool per candidate (the admit_reward_min floor, applied by the caller) or None.
        generator: optional torch.Generator for the eviction keys.
        """
        cid = self._cids(cid)
        B = cid.numel()
        x = self._states(x, B)
        log_r = self._vec(log_r, B, torch.float32, 'log_r')
        log_q = self._vec(log_q, B, torch.float32, 'log_q')
        ok = (torch.ones(B, dtype=torch.bool, device=self.device) if ok is None
              else self._vec(ok, B, torch.bool, 'ok'))
        tag = self._vec(tag, B, torch.int8, 'tag')
        if bool(((tag == TAG_ANCHOR) | (tag == TAG_CHILD)).any()):
            raise RowStoreError('admit() refuses data tags; data rows enter through seed() only')
        lv = self._level(level_table)
        self.set_conditions(cid, cond)

        L = lv[cid]
        r = log_r - log_q - L
        finite = torch.isfinite(log_r) & torch.isfinite(log_q) & torch.isfinite(L)
        fin = finite & ok
        ar = torch.arange(B, device=self.device)
        M = torch.full((self.n_cond,), -math.inf, dtype=torch.float32, device=self.device).scatter_reduce(
            0, cid, torch.where(fin, log_r, torch.full_like(log_r, -math.inf)), reduce='amax', include_self=True)
        at_max = fin & (log_r == M[cid]) & (M[cid] > self.best_log_r[cid])
        first = torch.full((self.n_cond,), B, dtype=torch.int64, device=self.device).scatter_reduce(
            0, cid[at_max], ar[at_max], reduce='amin', include_self=True)
        record = at_max & (ar == first[cid])
        take = fin & ((r.abs() >= self.bar) | record)
        self.best_log_r[cid[record]] = log_r[record]

        a = torch.nonzero(take).squeeze(1)
        n_old = len(self)
        na = a.numel()
        self._set_columns({
            'cid': torch.cat([self.cid, cid[a]]),
            'x': torch.cat([self.x, x[a]]),
            'log_r': torch.cat([self.log_r, log_r[a]]),
            'log_q': torch.cat([self.log_q, log_q[a]]),
            'q_step': torch.cat([self.q_step, torch.full((na,), int(step), dtype=torch.int64, device=self.device)]),
            'tag': torch.cat([self.tag, tag[a]]),
            'protected': torch.cat([self.protected, record[a]]),
            'held_out': torch.cat([self.held_out, torch.zeros(na, dtype=torch.bool, device=self.device)]),
            'born_eval': torch.cat([self.born_eval,
                                    torch.full((na,), self.eval_idx, dtype=torch.int32, device=self.device)])})
        n_records = int(record.sum())
        n_floor = int((finite & ~ok).sum())
        self._win['admitted'] += na
        self._win['admitted_records'] += n_records
        self._win['rejected_floor'] += n_floor
        self._win_admitted_r.append(r.abs()[a].detach().cpu())

        n_dem = self._demote()
        evict, n_cap, n_glob = self._evict(lv, generator)
        if evict.numel():
            r_ev = self.resid(lv, evict).abs()
            self._win_evicted_r.append(torch.where(torch.isfinite(r_ev), r_ev, torch.zeros_like(r_ev)).cpu())
            self._win_evicted_age.append((self.eval_idx - self.born_eval[evict]).cpu())
            keep = torch.ones(len(self), dtype=torch.bool, device=self.device)
            keep[evict] = False
            moved = bool((~keep[:n_old]).any())
            k = torch.nonzero(keep).squeeze(1)
            self._set_columns({c: getattr(self, c).index_select(0, k) for c in _COLUMNS})
            if moved:
                self.version += 1
        return {'admitted': na, 'records': n_records, 'rejected_floor': n_floor,
                'evicted_cap': n_cap, 'evicted_global': n_glob, 'demoted': n_dem, 'rows': len(self)}

    def _demote(self):
        """Per condition, protected non-data rows ranked by log R (highest first, ties by position):
        ranks >= records_kept lose protection."""
        if self.records_kept is None:
            return 0
        is_data = (self.tag == TAG_ANCHOR) | (self.tag == TAG_CHILD)
        cand = torch.nonzero(self.protected & ~is_data).squeeze(1)
        if cand.numel() <= self.records_kept:
            return 0
        c1 = cand[torch.argsort(self.log_r[cand], descending=True, stable=True)]
        c2 = c1[torch.argsort(self.cid[c1], stable=True)]
        dem = c2[_within_group_rank(self.cid[c2]) >= self.records_kept]
        self.protected[dem] = False
        self._win['demoted'] += dem.numel()
        return dem.numel()

    def _evict(self, lv, generator):
        """Rows to evict (store indices), and the per-condition and global counts.

        Pool: unprotected rows not held out. Keys -log(|r| + eps) + Gumbel(0, 1); non-finite r
        counts as |r| = 0. Per condition the excess over cap_per_condition with the highest keys
        goes; then the excess of the survivors over max_rows with the highest keys goes.
        """
        empty = torch.zeros(0, dtype=torch.int64, device=self.device)
        pool = torch.nonzero((~self.protected) & (~self.held_out)).squeeze(1)
        n_pool = pool.numel()
        if n_pool == 0:
            return empty, 0, 0
        cid = self.cid[pool]
        n_c = torch.bincount(cid, minlength=self.n_cond)
        excess_c = (n_c - self.cap_per_condition).clamp_min(0)
        excess_g = max(0, n_pool - int(excess_c.sum()) - self.max_rows)
        if int(excess_c.sum()) == 0 and excess_g == 0:
            return empty, 0, 0
        r = self.resid(lv, pool).abs().double()
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        dev = generator.device if generator is not None else self.device
        U = torch.rand(n_pool, generator=generator, dtype=torch.float64, device=dev).to(self.device)
        tiny = torch.finfo(torch.float64).tiny
        G = -torch.log(-torch.log(U.clamp(tiny, 1.0 - 1e-16)))
        key = -torch.log(r + float(self.eps)) + G

        o = torch.argsort(key, descending=True, stable=True)
        perm = o[torch.argsort(cid[o], stable=True)]
        ev_cap = torch.zeros(n_pool, dtype=torch.bool, device=self.device)
        ev_cap[perm] = _within_group_rank(cid[perm]) < excess_c[cid[perm]]
        ev_glob = torch.zeros(n_pool, dtype=torch.bool, device=self.device)
        if excess_g > 0:
            surv = torch.nonzero(~ev_cap).squeeze(1)
            top = torch.argsort(key[surv], descending=True, stable=True)[:excess_g]
            ev_glob[surv[top]] = True
        n_cap, n_glob = int(ev_cap.sum()), int(ev_glob.sum())
        self._win['evicted_cap'] += n_cap
        self._win['evicted_global'] += n_glob
        return pool[ev_cap | ev_glob], n_cap, n_glob

    # ---- monitoring -----------------------------------------------------------------

    def stats(self, level_table, prefix='store/'):
        """Store monitors against the live level; drains the admission/eviction window.

        Per-condition occupancy is over conditions with at least one non-held-out row;
        `conds` counts conditions holding any row; protected_above_bar excludes held-out rows.
        """
        out = {}
        out['rows'] = len(self)
        for t, name in enumerate(TAG_NAMES):
            out[f'rows_{name}'] = int((self.tag == t).sum())
        is_data = (self.tag == TAG_ANCHOR) | (self.tag == TAG_CHILD)
        nh = ~self.held_out
        out['rows_heldout'] = int(self.held_out.sum())
        out['rows_protected'] = int(self.protected.sum())
        out['rows_records'] = int((self.protected & ~is_data).sum())

        per = torch.bincount(self.cid[nh], minlength=self.n_cond)
        present = per > 0
        pc = per[present].double()
        out['cond_rows_min'] = float(pc.min()) if pc.numel() else math.nan
        out['cond_rows_median'] = _median(pc)
        out['cond_rows_max'] = float(pc.max()) if pc.numel() else math.nan
        held = torch.bincount(self.cid, minlength=self.n_cond) > 0
        out['conds'] = int(held.sum())
        unprot = torch.bincount(self.cid[(~self.protected) & nh], minlength=self.n_cond)
        out['conds_at_cap'] = int((held & (unprot >= self.cap_per_condition)).sum())

        out.update(self._win)
        ar = torch.cat(self._win_admitted_r) if self._win_admitted_r else torch.zeros(0)
        er = torch.cat(self._win_evicted_r) if self._win_evicted_r else torch.zeros(0)
        ea = torch.cat(self._win_evicted_age).double() if self._win_evicted_age else torch.zeros(0)
        out['admitted_resid_med'] = _median(ar)
        out['evicted_resid_med'] = _median(er)
        out['evicted_age_med'] = _median(ea)
        out['evicted_age_max'] = float(ea.max()) if ea.numel() else math.nan

        E, s, _ = self._drawable(level_table)
        above = self.protected & nh & torch.isfinite(s) & (s >= self.bar)
        per_above = torch.bincount(self.cid[above], minlength=self.n_cond)
        out['protected_above_bar'] = int(above.sum())
        out['protected_above_bar_conds'] = int((per_above > 0).sum())
        out['protected_above_bar_cond_max'] = int(per_above.max())
        sE = s[E].double()
        out['resid_med'] = _median(sE)
        out['resid_p90'] = float(torch.quantile(sE, 0.9)) if sE.numel() else math.nan
        lr = self.last_refresh
        out['refresh_seconds'] = float(lr['seconds']) if lr else math.nan
        out['refresh_nonfinite'] = int(lr['nonfinite']) if lr else math.nan
        self._reset_window()
        return {prefix + k: v for k, v in out.items()}

    # ---- persistence ----------------------------------------------------------------

    def state_dict(self, stamp=None):
        """Every column and table on CPU, the counters and the caller's stamp; settings are not saved."""
        return {'format_version': FORMAT_VERSION, 'stamp': stamp,
                'dim': self.dim, 'n_cond': self.n_cond, 'cond_dim': self.cond_dim,
                'columns': {c: getattr(self, c).detach().cpu().clone() for c in _COLUMNS},
                'cond_table': self.cond_table.detach().cpu().clone(),
                'best_log_r': self.best_log_r.detach().cpu().clone(),
                'n_mle': self.n_mle, 'n_seed': self.n_seed,
                'eval_idx': self.eval_idx, 'version': self.version}

    @classmethod
    def from_state_dict(cls, state, device, **settings):
        """Rebuild a store from state_dict(); settings come from the config, not the state."""
        fv = state.get('format_version')
        if fv != FORMAT_VERSION:
            raise RowStoreError(f'row store format_version {fv}, this code reads {FORMAT_VERSION}')
        st = cls(state['dim'], state['n_cond'], state['cond_dim'], device, **settings)
        cols = state['columns']
        n = cols['cid'].numel()
        for c in _COLUMNS:
            if cols[c].shape[0] != n:
                raise RowStoreError(f'row store column {c} has {cols[c].shape[0]} rows, cid has {n}')
        if cols['x'].dim() != 2 or cols['x'].shape[1] != st.dim:
            raise RowStoreError(f"row store column x has shape {tuple(cols['x'].shape)}, expected (n, {st.dim})")
        st._set_columns({c: cols[c].to(st.device) for c in _COLUMNS})
        st.cond_table = state['cond_table'].to(st.device, torch.float32)
        st.best_log_r = state['best_log_r'].to(st.device, torch.float32)
        st.n_mle, st.n_seed = int(state['n_mle']), int(state['n_seed'])
        st.eval_idx, st.version = int(state['eval_idx']), int(state['version'])
        return st
