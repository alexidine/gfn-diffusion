"""Relax FD starts or the campaign's latent random starts with the campaign random stream's stages; chunked output.

usage, from gfn_diffusion/energy_sampling (CPU):
    python -m eval.fd_search.run_fd_search --starts fd|latent --n N --seed S --run-name NAME
        [--virtual-batch 2000] [--chunk 32] [--threads 4] [--check-emulation] [--resume] [--fd-refold]
        [--convergence grad|params]

``--fd-refold`` adds ``fd_refold: True`` to every stage, passed to MXtalTools' ``gradient_descent_optimization``
(crystal_fd re-descriptions of rows at the angle clamp or near / past F = 0 during the walk; the end of a stage folded
by crystal_fd, ``fd_fold_aunit``, which the emulation's rebuild then calls instead of ``canonicalize_aunit``).

Starts.
- ``fd``: ``fd_starts.FDStartSampler`` (shape and pose uniform on the FD, size from the campaign's packing-coefficient
  draw), a torch Generator seeded with (seed, virtual-batch index); then every cell length is floored at the latent
  initializer's floor (3 A, ``length_floor``) as that initializer floors it (``fd_start_rows`` says why).
- ``latent``: the campaign random stream's own initializer, ``crystal_search.utils.init_samples_to_optim`` and
  ``get_initial_state`` with the stream's settings (init_sample_method random, init_reduced, init_target_cp std),
  ``opt_seed = --seed`` and ``batch_idx`` = the virtual-batch index, drawn as a block of the campaign's batch size
  (``--virtual-batch``) so that its batch-level rescale (every row rescaled when any row's packing coefficient is
  more than 0.1 off its target) acts as in the campaign; the global torch seed is set before the call, so the target
  packing coefficients it draws first are reproducible and recorded.

Relaxation. The stages are the campaign stream's ``opt`` list (streams/random.yaml) run through MXtalTools HEAD's
``optimize_crystal_parameters`` (eLJ, rprop, centroid_boundary wrap, the stage-2 early-stop cascade). The cascade's
reference is by default the fixed campaign reference ``CAMPAIGN_REF`` for every virtual batch (``--early-stop-ref
campaign``), the same for both arms; run_search used each job's running minimum, None for its first batch, and the
campaign random stream's four jobs had found a minimum within 0.008 kT of ``CAMPAIGN_REF`` by the end of their first
batch, so the two rules differ only on those first batches (8,000 of 149,494 rows, which ran without the cascade).
``--early-stop-ref running`` is run_search's rule on this run's own results. The campaign ran batches of about
2,000 rows under the stopping rule of that time (MXtalTools ``convergence='params'``): a stage stopped when more than
95% of a batch's active rows had converged (``check_convergence``), so how long a row was relaxed depended on its
batch; it is still MXtalTools' default. With ``--convergence grad`` (MXtalTools' opt-in rule since 2026-10-27) each
row stops on its own gradient and leaves the batch: such a stage is simply run in chunks, with no emulation, and each
row reports its ``converged_step``. Under ``convergence='params'`` (the default) the batch's stopping rule is emulated: each chunk of ``--chunk``
rows is run to the stage's step cap (the stage's convergence check disabled by ``convergence_eps = -1``), then the
stop step T of a batch of ``--virtual-batch`` rows is found by calling MXtalTools' own ``check_convergence`` on the
batch's assembled parameter record, step by step, as the stage loop calls it, the stage-2 cascade is replayed from
the recorded losses, and each row's output is rebuilt as ``gradient_descent_optimization`` rebuilds it (its
lowest-loss state among the first T steps, set, wrapped, ``canonicalize_aunit``, analysed). Row-evaluations are
counted as run_search counts them (non-inf loss entries among the first T steps; the evaluations run past T are
recorded apart). ``--check-emulation`` compares, on the first chunk, a normal stage run (its own quorum) with the
emulation of that chunk alone: stop step, evaluations, output parameters and energies must agree (they did, bit for
bit, 2026-10-26, both stages, the cascade included).

What the emulation does not reproduce: a row's TRAJECTORY depends on the other rows of the batch it is relaxed in.
rprop steps by gradient sign, so neither the batch-mean loss nor the joint gradient clip changes a row's step, but the
eLJ GRADIENT of some crystals depends on the batch (measured 2026-10-26 on 32 FD starts: eLJ values bit-identical,
gradients up to 81% different in 5 of 32 rows, all cells with one axis of 73-148 A and one of 1.4-5.4 A, between a
32-row batch and 8-row batches; 9 of 32 loss trajectories differ by step 30). So the rows of a virtual batch are
relaxed in chunks, not in one 2,000-row batch as the campaign's were: the stop step follows the campaign's rule, the
float trajectories are another realization.

Output: ``OUT_ROOT/<run_name>/`` (a new directory; an existing one is refused unless ``--resume`` continues the
same run): ``meta.json``, ``run.log``, one ``vb_<k>.pt`` per virtual batch (starts, endpoints, eLJ, row-evaluations,
the cascade reference of each stage, the FD canonical description of every start and endpoint), ``summary.json``.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

from eval.fd_search import campaign, fd_starts

SG = campaign.SG
#: the campaign's energy reference (registry minimum; stats.md), for --early-stop-ref campaign
CAMPAIGN_REF = campaign.CAMPAIGN_REF


# ---------------------------------------------------------------------------
# starts
# ---------------------------------------------------------------------------

def latent_starts(config, block: int, batch_idx: int, seed: int):
    """``block`` rows from the campaign random stream's initializer at ``batch_idx``; returns (batch, target_cp)."""
    from mxtaltools.crystal_search.utils import get_initial_state, init_samples_to_optim
    from mxtaltools.dataset_utils.utils import collate_data_list
    if not (config.init_sample_method == 'random' and config.init_reduced and config.init_target_cp == 'std'):
        raise ValueError('the campaign stream is not the random/reduced/std stream this driver reproduces')
    config.num_samples = block
    config.opt_seed = int(seed)
    samples = init_samples_to_optim(config)  # placeholders: the conformer, sg 14, Z' 1 (shuffled; all identical)
    batch = collate_data_list(samples)
    gseed = int(seed) * 7919 + int(batch_idx)
    torch.manual_seed(gseed)  # get_initial_state draws its target packing coefficients before it seeds
    target_cp = (torch.randn(block) * 0.0447 + 0.6226).clip(min=0.45, max=0.95)
    torch.manual_seed(gseed)
    batch = get_initial_state(config, batch, 'cpu', batch_idx)
    return batch, target_cp.double()


def length_floor() -> float:
    """The campaign initializer's length floor: ``get_initial_state`` ends with ``clean_cell_parameters(mode='hard')``,
    whose ``length_pad`` default (3 A) clips every cell length from below."""
    import inspect
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    return float(inspect.signature(MolCrystalData.clean_cell_parameters).parameters['length_pad'].default)


def fd_start_rows(sampler, n: int, seed: int, vb: int, floor: float):
    """n FD starts of virtual batch ``vb``: the draw is checked (``fd_starts.check_rows``) as drawn, then every cell
    length is floored at ``floor`` as the latent arm's initializer floors it.

    Why the floor is applied here. The optimizer cleans lengths each step with ``clip(min=3)``, whose gradient is zero
    below 3: a raw length under 3 never moves (rprop with a zero gradient), so an FD axis drawn under 3 A would stay
    frozen at 3 A for the whole stage, while the latent initializer hands such an axis to the optimizer at exactly 3,
    where the gradient flows (2026-10-26 probe: six FD starts with an axis of 1.9-2.9 A, as drawn: axis 3.000 at
    every recorded step; floored first: the axis grew to 3.75-4.68 A and the stage ended 77-211 eLJ lower). Only the
    lengths are floored: ``clean_cell_parameters`` itself would also clamp each centroid to MXtalTools' asymmetric-unit
    box and so move every start with canonical t_y > 1/4 onto that box's face. The floor can take a start out of the
    reduced set (it changes a/c); the start record keeps the draw (``draw_params``) and the floored mask (``clipped``).

    Returns (batch, cp_target, desc, info, draw_params [n, 12] float32, clipped [n, 3] bool)."""
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(vb))
    desc, cp, k, info = sampler.draw(n, g)
    batch = sampler.to_search_rows(desc)
    info = dict(info, check=fd_starts.check_rows(SG, batch, desc, cp, n_builder=4))  # the draw, before the floor
    draw = batch.full_cell_parameters().detach().float().clone()
    clipped = batch.cell_lengths.detach() < floor
    batch.cell_lengths = batch.cell_lengths.clip(min=floor)
    batch.box_analysis()
    return batch, cp, desc, info, draw, clipped


def canonical_record(batch):
    """FD canonical form of MolCrystalData rows (crystal_fd ``canon.canonicalize``, +1 sheet): float64 params
    [n, 12] in MolCrystalData's layout, handedness, status codes, n_auto, near_face."""
    from mxtaltools.crystal_fd import canon
    d = fd_starts.rows_to_desc(SG, batch)
    c = canon.canonicalize(d)
    L, A, t, rv, h = c.desc.crystal_params()
    return dict(params=torch.cat([L, A, t, rv], 1), hand=h.reshape(-1).to(torch.int8),
                status=c.status.clone(), n_auto=c.n_auto.clone(), near_face=c.near_face.clone())


# ---------------------------------------------------------------------------
# one stage over a virtual batch
# ---------------------------------------------------------------------------

def _energy_computes(optim_target):
    target_map = {'silu': 'silu', 'qlj': 'qlj', 'ellipsoid': 'ellipsoid', 'reduce': 'reduction_en', 'uma': 'uma',
                  'mace': 'mace', 'rdf_dist': 'rdf', 'elj': 'elj', 'exp6': 'exp6', 'lj': 'lj'}
    return list(set(['lj', 'elj', target_map[optim_target.lower()]]))


def _stage_config(stage, config):
    from mxtaltools.crystal_search.utils import parse_opt_config
    oc = parse_opt_config(dict(stage), config, 'cpu', None)
    if oc.pop('keep_lowest_fraction', None) is not None:
        raise NotImplementedError('keep_lowest_fraction is not emulated')
    if oc.get('enforce_reduced'):
        raise NotImplementedError('enforce_reduced is not emulated')
    if str(oc['optim_target']).lower() in ('uma', 'mace', 'rdf_dist'):
        raise NotImplementedError('only force-field targets are emulated')
    return oc


def _run_chunk(rows, oc, early_stop_ref, intermediates, capped: bool):
    from mxtaltools.dataset_utils.utils import collate_data_list
    cb = collate_data_list(rows)
    run = dict(oc, intermediates_path=intermediates)
    if oc.get('early_stop') is not None:
        run['early_stop_ref'] = early_stop_ref
    if capped:
        run['convergence_eps'] = -1.0  # never converges: runs to the stage's step cap
    return cb, run, cb.optimize_crystal_parameters(return_record=True, **run)


def _rebuild(rows, best, oc, early_stopped):
    """gradient_descent_optimization's output rows for the chosen states (its final block, verbatim in effect)."""
    from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit, fd_fold_aunit, wrap_centroid
    from mxtaltools.dataset_utils.utils import collate_data_list
    cb = collate_data_list(rows)
    init = cb.clone().to('cpu')
    init.orient_molecule(mode='standardized', target_handedness=cb.aunit_handedness)
    x = init.clone().detach()
    x.set_cell_parameters(best.to(x.device), skip_box_analysis=False)
    if oc.get('centroid_boundary', 'clamp') == 'wrap':
        x.aunit_centroid = wrap_centroid(x.aunit_centroid)
        (fd_fold_aunit if oc.get('fd_refold') else canonicalize_aunit)(x)
    else:
        x.canonicalize_orientation()
    cutoff = oc.get('cutoff') or 10
    with torch.no_grad():
        x.analyze(computes=_energy_computes(oc['optim_target']), cutoff=cutoff, rdf_cutoff=cutoff,
                  supercell_size=oc.get('supercell_size', 10), return_cluster=False, repulsion=oc.get('repulsion', 1.0),
                  surface_padding=0, predictor=None, assign_outputs=True, elementwise=False, atomwise=True, bins=100)
    out = x.batch_to_list()
    if early_stopped is not None:
        for i, elem in enumerate(out):
            elem.early_stopped = torch.tensor([bool(early_stopped[i])])
    return out


def emulate_stop(loss, params, oc, early_stop_ref):
    """The stop step T of one batch whose per-step loss [S, n] and parameter record [S, n, p] (capped runs) are
    given, and the step each row was retired at (n, a large value if never), exactly as the stage loop decides."""
    from mxtaltools.crystal_search.crystal_opt_utils import check_convergence
    S, n = loss.shape
    max_steps = int(oc['max_num_steps'])
    if S != max_steps - 1:
        raise ValueError(f'records hold {S} steps, expected {max_steps - 1}')
    cascade = {int(st): float(m) for st, m in (oc.get('early_stop') or [])} if early_stop_ref is not None else {}
    kT = float(oc.get('early_stop_kT', 1.0))
    never = 10 ** 9
    retired_at = torch.full((n,), never, dtype=torch.long)
    active = torch.ones(n, dtype=torch.bool)
    for k in sorted(cascade):
        if k >= S:
            continue
        r = active & (loss[k] > early_stop_ref + cascade[k] * kT)
        retired_at[r] = k
        active &= ~r
    # consistency with the capped runs, which ran the same cascade: a row retired at k has no loss after k
    for i in torch.nonzero(retired_at < never).flatten().tolist():
        k = int(retired_at[i])
        if not bool(torch.isinf(loss[k + 1:, i]).all()):
            raise RuntimeError(f'row {i}: retired at step {k} by the emulation but evaluated after it')
    min_steps = min(max_steps, 50)
    eps = float(oc['convergence_eps'])
    T = S
    for s in range(1, S):
        act = retired_at >= s
        if not bool(act.any()):
            T = s
            break
        if s >= min_steps and bool(check_convergence(params, s, eps, None, None, active=act).all()):
            T = s
            break
    return T, retired_at


def run_stage(rows, stage, config, chunk, early_stop_ref, intermediates, log=print):
    """One campaign stage over the virtual batch ``rows`` (list of MolCrystalData). Returns (output rows, info)."""
    oc = _stage_config(stage, config)
    n = len(rows)
    S = int(oc['max_num_steps']) - 1
    if oc.get('convergence', 'params') == 'grad':  # per-row stopping: no batch rule to emulate
        return _run_stage_by_grad(rows, oc, chunk, early_stop_ref, intermediates, S, log)
    loss = torch.full((S, n), float('inf'))
    params = None
    t0 = time.time()
    run_evals = 0
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        _, _, (out, rec) = _run_chunk(rows[lo:hi], oc, early_stop_ref, intermediates, capped=True)
        sc = rec['loss'].shape[0]
        if params is None:
            params = torch.zeros(S, n, rec['params'].shape[-1])
        loss[:sc, lo:hi] = rec['loss']
        params[:sc, lo:hi] = rec['params']
        if sc < S:  # every row of the chunk retired early: frozen rows, no more evaluations
            params[sc:, lo:hi] = rec['params'][-1]
        run_evals += int(rec['loss'].numel() - torch.isinf(rec['loss']).sum())
        del out, rec
    t_run = time.time() - t0
    T, retired_at = emulate_stop(loss, params, oc, early_stop_ref)
    head = loss[:T]
    best_ind = torch.argmin(head, dim=0)
    best = params[best_ind, torch.arange(n)]
    evals = (~torch.isinf(head)).sum(0)
    cascade_on = oc.get('early_stop') is not None and early_stop_ref is not None
    early_stopped = (retired_at < T) if cascade_on else None
    t1 = time.time()
    outs = []
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        outs += _rebuild(rows[lo:hi], best[lo:hi], oc,
                         None if early_stopped is None else early_stopped[lo:hi])
    info = dict(T=int(T), steps_cap=S, evals=evals, evals_run=run_evals, wall_run=t_run, wall_rebuild=time.time() - t1,
                retired=int((retired_at < T).sum()) if cascade_on else 0, early_stop_ref=early_stop_ref)
    log(f'  stage {oc["optim_target"]} T={T}/{S} retired={info["retired"]} evals/row={float(evals.float().mean()):.1f}'
        f' run {t_run:.0f}s rebuild {info["wall_rebuild"]:.0f}s')
    return outs, info


def _run_stage_by_grad(rows, oc, chunk, early_stop_ref, intermediates, S, log):
    """A ``convergence='grad'`` stage: each chunk run as MXtalTools runs it (rows stop on their own gradients)."""
    t0 = time.time()
    outs, evals, T = [], [], 0
    for lo in range(0, len(rows), chunk):
        _, _, (out, rec) = _run_chunk(rows[lo:lo + chunk], oc, early_stop_ref, intermediates, capped=False)
        outs += out
        evals.append((~torch.isinf(rec['loss'])).sum(0))
        T = max(T, int(rec['loss'].shape[0]))
    evals = torch.cat(evals)
    retired = sum(bool(getattr(r, 'early_stopped', torch.tensor([False]))[0]) for r in outs)
    converged = sum(int(r.converged_step) >= 0 for r in outs)
    t_run = time.time() - t0
    info = dict(T=T, steps_cap=S, evals=evals, evals_run=int(evals.sum()), wall_run=t_run, wall_rebuild=0.0,
                retired=retired, converged=converged, early_stop_ref=early_stop_ref)
    log(f'  stage {oc["optim_target"]} (grad) T={T}/{S} converged={converged} retired={retired} '
        f'evals/row={float(evals.float().mean()):.1f} run {t_run:.0f}s')
    return outs, info


def check_emulation(rows, stage, config, early_stop_ref, intermediates):
    """A normal stage run of ``rows`` (its own quorum) against the emulation of the same rows: dict of differences."""
    oc = _stage_config(stage, config)
    _, _, (ref_out, ref_rec) = _run_chunk(rows, oc, early_stop_ref, intermediates, capped=False)
    emu_out, info = run_stage(rows, stage, config, len(rows), early_stop_ref, intermediates, log=lambda *a: None)
    from mxtaltools.dataset_utils.utils import collate_data_list
    a, b = collate_data_list(ref_out), collate_data_list(emu_out)
    T_ref = int(ref_rec['loss'].shape[0])
    ev_ref = int(ref_rec['loss'].numel() - torch.isinf(ref_rec['loss']).sum())
    return dict(T_normal=T_ref, T_emulated=info['T'], evals_normal=ev_ref, evals_emulated=int(info['evals'].sum()),
                max_param_diff=float((a.full_cell_parameters() - b.full_cell_parameters()).abs().max()),
                max_elj_diff=float((a.elj - b.elj).abs().max()),
                handedness_equal=bool(torch.equal(a.aunit_handedness, b.aunit_handedness)),
                retired_normal=int(sum(bool(getattr(r, 'early_stopped', torch.tensor([False]))[0]) for r in ref_out)),
                retired_emulated=info['retired']), emu_out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _git_head(path):
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--starts', choices=('fd', 'latent'), required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--virtual-batch', type=int, default=2000, help="the campaign's batch size (stop quorum)")
    ap.add_argument('--chunk', type=int, default=32, help='rows relaxed together (memory)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--check-emulation', action='store_true')
    ap.add_argument('--early-stop-ref', choices=('campaign', 'running'), default='campaign',
                    help="stage-2 cascade reference: 'campaign' (default), the fixed campaign reference CAMPAIGN_REF "
                         "for every virtual batch, the first included, so both arms run the same stopping rule; "
                         "'running', run_search's rule, this run's own running minimum final eLJ (None, no cascade, "
                         "before the first virtual batch completes), which couples the rule to the arm's results")
    ap.add_argument('--resume', action='store_true', help='continue this run in its own existing directory')
    ap.add_argument('--fd-refold', action='store_true',
                    help="add fd_refold: True to every stage (MXtalTools' optimiser re-describes rows by crystal_fd)")
    ap.add_argument('--convergence', choices=('grad', 'params'), default='params',
                    help="MXtalTools' stopping rule for every stage: 'params' (its default and the campaign's rule, "
                         "emulated on virtual batches) or 'grad' (opt-in, per row)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        print('warning: CUDA_VISIBLE_DEVICES is not -1; this driver runs on the CPU regardless')

    config = campaign.stream_config('random', device='cpu')
    if args.fd_refold:
        config.opt = [dict(dict(st), fd_refold=True) for st in config.opt]
    config.opt = [dict(dict(st), convergence=args.convergence) for st in config.opt]
    cc = campaign.coord_config()
    if int(cc['sg']) != SG or int(cc['z_prime']) != 1 or list(config.sgs_to_search) != [SG]:
        raise ValueError(f"the campaign's coord.yaml (sg {cc['sg']}, Z' {cc['z_prime']}) disagrees with "
                         f"campaign.CAMPAIGNS[{campaign.CAMPAIGN!r}] (sg {SG}, Z' 1)")
    keys = ('starts', 'n', 'seed', 'virtual_batch', 'early_stop_ref')
    if args.resume:
        out_dir = campaign.run_dir(args.run_name)
        with open(os.path.join(out_dir, 'meta.json')) as fh:
            meta = json.load(fh)
        if meta.get('campaign') != campaign.CAMPAIGN_DIR:
            raise ValueError(f"--resume under campaign {campaign.CAMPAIGN_DIR}, but the run was started under {meta.get('campaign')}")
        if (any(meta['args'][k] != getattr(args, k) for k in keys)
                or meta['args'].get('fd_refold', False) != args.fd_refold
                or meta['args'].get('convergence', 'params') != args.convergence):
            raise ValueError(f'--resume with other arguments than the run was started with: {meta["args"]}')
    else:
        out_dir = campaign.new_run_dir(args.run_name)
        meta = dict(args=vars(args), campaign=campaign.CAMPAIGN_DIR, campaign_commit=campaign.CAMPAIGN_COMMIT,
                    mxt_head=_git_head(os.path.dirname(os.path.dirname(os.path.abspath(
                        __import__('mxtaltools').__file__)))),
                    gfn_head=_git_head(os.path.dirname(os.path.abspath(__file__))),
                    stages=[dict(s) for s in config.opt], mol=campaign.LOCAL_MOL,
                    started=time.strftime('%Y-%m-%d %H:%M:%S'), python=sys.version)
        with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=1, default=str)
    campaign.check_not_campaign(out_dir)
    intermediates = os.path.join(out_dir, '_opt_intermediates.pt')  # written by MXT only on a CUDA OOM

    log_path = os.path.join(out_dir, 'run.log')

    def log(*a):
        print(*a, flush=True)
        with open(log_path, 'a') as fh:  # the run's own log, inside its own directory
            print(*a, file=fh)

    log(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {"resumed" if args.resume else "started"}: {vars(args)}')
    mol = campaign.load_molecule()
    sampler = fd_starts.FDStartSampler(SG, mol) if args.starts == 'fd' else None
    floor = length_floor()

    def stage_ref(st, running_min):
        """The cascade reference of stage ``st`` (None: the stage has no cascade, or 'running' before any result)."""
        if dict(st).get('early_stop') is None:
            return None
        return CAMPAIGN_REF if args.early_stop_ref == 'campaign' else running_min
    n_vb = math.ceil(args.n / args.virtual_batch)
    running_min = None
    done = set()
    for k in range(n_vb):
        p = os.path.join(out_dir, f'vb_{k:05d}.pt')
        if os.path.exists(p):
            d = torch.load(p, weights_only=False)
            done.add(k)
            m = float(d['elj'].min())
            running_min = m if running_min is None else min(running_min, m)
    import psutil
    proc = psutil.Process(os.getpid())
    rss_peak = 0.0
    for vb in range(n_vb):
        if vb in done:
            continue
        m = min(args.virtual_batch, args.n - vb * args.virtual_batch)
        t0 = time.time()
        start_info = {}
        draw, clipped = None, None
        if args.starts == 'fd':
            batch, cp_target, desc, start_info, draw, clipped = fd_start_rows(sampler, m, args.seed, vb, floor)
        else:
            block, cp_all = latent_starts(config, args.virtual_batch, vb, args.seed)
            from mxtaltools.dataset_utils.utils import collate_data_list
            batch = collate_data_list(block.batch_to_list()[:m])
            cp_target = cp_all[:m]
        lengths = batch.cell_lengths.detach()
        below = int((lengths < floor).any(1).sum())
        if below and args.starts == 'fd':
            raise RuntimeError(f'{below} FD starts have a cell length below the floor {floor} A: frozen in stage 1')
        if below:  # the latent initializer floors only when its batch rescale fires; reproduced as the campaign ran
            log(f'  note: {below} latent starts have a cell length below {floor} A (no batch rescale in this block)')
        start = dict(params=batch.full_cell_parameters().detach().float().clone(),
                     hand=batch.aunit_handedness.detach().reshape(-1).float().clone(),
                     cp=batch.packing_coeff.detach().double().clone(), cp_target=cp_target.clone(),
                     fd=canonical_record(batch), length_floor=floor,
                     at_floor=(lengths - floor).abs() < 1e-5,  # [m, 3]: lengths the initializer set to the floor
                     draw_params=draw, clipped=clipped)  # FD arm: the draw before the floor, and the floored mask
        log(f'[{args.run_name}] vb {vb}: {m} {args.starts} starts in {time.time() - t0:.1f}s; '
            f'rows with a length at the {floor:g} A floor {int(start["at_floor"].any(1).sum())}'
            + ('' if clipped is None else f' (floored here {int(clipped.any(1).sum())})')
            + f'; FD statuses {dict(zip(*np.unique([int(s) for s in start["fd"]["status"]], return_counts=True)))}')
        rows = batch.batch_to_list()
        emu_check = None
        if args.check_emulation and vb == min(set(range(n_vb)) - done):
            c = rows[:args.chunk]
            d1, out1 = check_emulation(c, config.opt[0], config, stage_ref(config.opt[0], running_min), intermediates)
            ref2 = stage_ref(config.opt[1], running_min)
            ref2 = CAMPAIGN_REF if ref2 is None else ref2  # exercise the cascade
            d2, _ = check_emulation(out1, config.opt[1], config, ref2, intermediates)
            emu_check = dict(stage0=d1, stage1=d2, stage1_ref=ref2)
            log(f'  emulation check: {emu_check}')
        stage_info = []
        for st in config.opt:
            rows, info = run_stage(rows, st, config, args.chunk, stage_ref(st, running_min), intermediates, log)
            stage_info.append(info)
            mi = proc.memory_info()
            rss_peak = max(rss_peak, getattr(mi, 'peak_wset', mi.rss) / 1e9)  # Windows: the process's peak
        from mxtaltools.dataset_utils.utils import collate_data_list
        final = collate_data_list(rows)
        final.box_analysis()
        elj = final.elj.detach().double().reshape(-1)
        rec = dict(arm=args.starts, vb=vb, seed=args.seed, start=start, start_info=start_info,
                   params=final.full_cell_parameters().detach().float().clone(),
                   hand=final.aunit_handedness.detach().reshape(-1).float().clone(),
                   elj=elj, lj=final.lj.detach().double().reshape(-1),
                   cp=final.packing_coeff.detach().double().reshape(-1),
                   early_stopped=torch.tensor([bool(getattr(r, 'early_stopped', torch.tensor([False]))[0])
                                               for r in rows]),
                   row_evals=sum(si['evals'] for si in stage_info).long(),
                   row_evals_stage=torch.stack([si['evals'] for si in stage_info], 1).long(),
                   T=[si['T'] for si in stage_info], evals_run=[si['evals_run'] for si in stage_info],
                   wall=time.time() - t0, wall_stage=[si['wall_run'] + si['wall_rebuild'] for si in stage_info],
                   early_stop_ref=[si['early_stop_ref'] for si in stage_info],  # per stage, as used (None: none)
                   early_stop_mode=args.early_stop_ref, running_min_before=running_min,
                   end_fd=canonical_record(final), emulation_check=emu_check,
                   rss_peak_gb=rss_peak)
        _atomic_save(rec, os.path.join(out_dir, f'vb_{vb:05d}.pt'))
        running_min = float(elj.min()) if running_min is None else min(running_min, float(elj.min()))
        log(f'[{args.run_name}] vb {vb} done: {m} relaxations, {rec["wall"]:.0f}s ({rec["wall"] / m:.2f} s each), '
            f'row-evals/relaxation {float(rec["row_evals"].double().mean()):.1f}, min eLJ {float(elj.min()):.2f}, '
            f'RSS peak {rss_peak:.2f} GB')
    # summary over every virtual batch
    recs = [torch.load(os.path.join(out_dir, f'vb_{k:05d}.pt'), weights_only=False) for k in range(n_vb)]
    n = sum(len(r['elj']) for r in recs)
    summ = dict(run=args.run_name, arm=args.starts, n=n, wall_s=sum(r['wall'] for r in recs),
                s_per_relaxation=sum(r['wall'] for r in recs) / n,
                row_evals_per_relaxation=float(torch.cat([r['row_evals'] for r in recs]).double().mean()),
                evals_run_per_relaxation=sum(sum(r['evals_run']) for r in recs) / n,
                # the wall time had each row run only to its batch's stop step (the campaign's cost), by evaluations
                s_per_relaxation_campaign_equivalent=sum(r['wall'] * float(r['row_evals'].sum()) / sum(r['evals_run'])
                                                         for r in recs) / n,
                T=[r['T'] for r in recs], rss_peak_gb=max(r['rss_peak_gb'] for r in recs),
                min_elj=float(min(float(r['elj'].min()) for r in recs)))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    log(json.dumps(summ))


if __name__ == '__main__':
    main()
