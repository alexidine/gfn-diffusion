"""
Offline draws from a conditional crystal checkpoint, for a chosen condition.

The trainer's eval pools every condition it draws into one batch (2,500 train and 1,000
held-out samples over thousands of molecules, about 1.7 draws per held-out molecule), so it
cannot show what any one molecule's distribution looks like. This module rebuilds the policy
and the reward from an archive and draws through the function the trainer's eval calls,
eval/utils.py::sample_eval_fwd_trajs, with the eval's own settings: the checkpoint's eval
weights, a uniform grid of cfg:eval_T steps, T from the run's energy_config, no grad.

THREE THINGS THE EXISTING LOADERS GET WRONG ON THIS ROUTE, each handled here:

  P_B. A stage declaring freeze_pb scores log P_B on a snapshot the checkpoint carries under
  'pb_frozen' (GFN.freeze_backward_policy), not on the live backward head. The snapshot is
  not in the model state_dict. build_prior_flow.py::load_policy and
  new_analysis.py::load_gfn never install it, so every log P_B, log w and xcond value they
  would compute is scored under a backward kernel training never used.

  The condition. new_analysis.py::sample_latents passes a zeros condition. The QM9 route
  conditions on the 192-wide encoder embedding carried on each conditions-file row, turned
  into the policy input by MolecularCrystal.condition_samples, as in training.

  The device. Checkpointer._gfn_config_from keeps gfn_config['device'] ('cuda' on the
  cluster and laptop runs alike), and GFN builds its index tensors on that device as plain
  attributes that .to() does not move. The device is overridden before construction.

THE REGISTRY. condition_id keys every tracker lookup, and it is the rank of the molecule's
identifier in the sorted union over the conditions, prior and held-out files
(train.py::Modeller.init_identifiers). It is rebuilt the same way here and checked against
the tracker's library size, so an id can never silently point at another molecule's slot.

Constructing a GFN resets torch's global RNG (scalarMLP init), so seeds are set after it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import torch
import yaml

from energies.molecular_crystal import MolecularCrystal
from energy_sampling.eval.utils import sample_eval_fwd_trajs
from models import GFN
from utils import get_gfn_init_state, logmeanexp, uniform_discretizer

# per-sample fields copied off the scored crystal batch when present; which were absent
# is reported on the draw rather than guessed at
SAMPLE_FIELDS = ('mol_energy', 'elj', 'density_energy', 'pressure_energy', 'reduction_en',
                 'reduction_energy', 'bounding_energy', 'jacobian_energy',
                 'rot_r_jacobian_energy', 'rot_theta_jacobian_energy', 'physical_energy',
                 'gfn_energy', 'packing_coeff')


@dataclass
class Run:
    """One checkpoint rebuilt for offline sampling."""
    gfn: GFN
    energy_function: MolecularCrystal
    conditions: object          # collated crystal batch, one row per training molecule
    test_conditions: object     # the held-out rows, or None
    registry: dict              # identifier -> mol_id, as init_identifiers builds it
    tracker: dict               # ck['condition_log_z'] tensors, indexed by condition_id
    step: int
    stage: str
    temperature: float
    eval_T: int
    device: str
    config: dict = field(repr=False, default_factory=dict)

    def rows(self, identifier: str, n: int):
        """`n` copies of one molecule's conditions row, train or held-out."""
        for batch in (self.conditions, self.test_conditions):
            if batch is None:
                continue
            hits = [i for i, ident in enumerate(batch.identifier) if ident == identifier]
            if hits:
                if len(hits) > 1:
                    raise ValueError(f'{identifier} has {len(hits)} rows in one conditions file')
                return batch.subsample_new_batch(torch.full((n,), hits[0], dtype=torch.long))
        raise KeyError(f'{identifier} is in neither conditions file')

    def random_rows(self, batch, n: int, generator: torch.Generator):
        """`n` rows drawn uniformly with replacement: the eval's unweighted draw over rows."""
        idx = torch.randint(0, batch.num_graphs, (n,), generator=generator)
        return batch.subsample_new_batch(idx)

    def is_held_out(self, identifier: str) -> bool:
        return (self.test_conditions is not None
                and identifier in set(self.test_conditions.identifier))


def _load_condition_file(path):
    from train import Modeller
    return Modeller._load_condition_file(path)


def reasonable_mask(run: Run, sample_batch):
    """train.py::Modeller._reasonable_sample_mask, called rather than copied: the gate
    compares this fraction with the logged one, so the two must be one function."""
    from train import Modeller
    return Modeller._reasonable_sample_mask(SimpleNamespace(energy_function=run.energy_function),
                                            sample_batch)


def _attach_mol_id(batch, registry):
    mol_id = torch.tensor([registry[ident] for ident in batch.identifier], dtype=torch.long)
    batch.add_graph_attr(mol_id, 'mol_id')


def load_run(checkpoint_path: str, config_path: str, device: str = 'cpu',
             prior_path: Optional[str] = None) -> Run:
    """Rebuild policy, reward, conditions and registry for one archive.

    config_path is the run's yaml: the energy coefficients are not in problem_def (only
    its identity keys are), so they are read from the config and the identity keys are
    asserted equal between the two.
    """
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    pdef = ck['problem_def']
    ec = dict(cfg['energy_config'])
    for key, want in (('energy_function', cfg['energy_function']),
                      ('space_groups', cfg['space_groups']),
                      ('z_primes', cfg['z_primes']),
                      ('emb_cond', bool(cfg.get('embedding_conditioning', False)))):
        if pdef.get(key) != want:
            raise ValueError(f'checkpoint problem_def {key}={pdef.get(key)!r}, config has {want!r}')
    if float(pdef['energy_config']['temperature']) != float(ec['temperature']):
        raise ValueError(f"checkpoint T {pdef['energy_config']['temperature']} != config T {ec['temperature']}")

    gcfg = dict(ck['gfn_config'])
    gcfg['device'] = device
    gfn = GFN(**gcfg).to(device)
    gfn.load_state_dict(ck['model_eval'], strict=True)
    gfn.eval()
    if ck.get('pb_frozen') is not None:
        gfn.freeze_backward_policy(source_state=ck['pb_frozen'])

    energy_config = {
        'device': device,
        'energy_function': cfg['energy_function'],
        'mlip_path': cfg.get('mlip_path'),
        'space_groups': cfg['space_groups'],
        'z_primes': cfg['z_primes'],
        'sg_conditioning': cfg.get('sg_conditioning', False),
        'temperature_conditioning': cfg.get('temperature_conditioning', False),
        'zp_conditioning': cfg.get('zp_conditioning', False),
        'vector_conditioning': cfg.get('vector_conditioning', False),
        'vector_conditioning_dim': cfg.get('vector_conditioning_dim'),
        'embedding_conditioning': cfg.get('embedding_conditioning', False),
        'embedding_conditioning_dim': cfg.get('embedding_conditioning_dim'),
    }
    energy_config.update(ec)
    energy_config['internal_oom_recovery'] = False
    ef = MolecularCrystal(**energy_config)

    conditions = _load_condition_file(cfg['molecules_path'])
    test = (_load_condition_file(cfg['test_molecules_path'])
            if cfg.get('test_molecules_path') else None)

    # the prior contributes identifiers to the registry even where it adds no molecule,
    # so it is read, not assumed a subset (init_identifiers does not assume it either)
    prior = torch.load(prior_path or cfg['prior_path'], map_location='cpu', weights_only=False)
    prior_batch = prior['equalized_prior'] if isinstance(prior, dict) else prior
    if isinstance(prior, dict) and 'thermal_scaling_factor' in prior:
        # init_prior_dataset writes it over lj_coeff for the whole run
        ef.lj_coeff = float(prior['thermal_scaling_factor'])
    identifiers = set(conditions.identifier) | set(prior_batch.identifier)
    if test is not None:
        identifiers |= set(test.identifier)
    registry = {ident: i for i, ident in enumerate(sorted(identifiers))}
    del prior, prior_batch

    tracker = ck.get('condition_log_z') or {}
    if 'ema_logw' in tracker and tracker['ema_logw'].numel() != len(registry):
        raise ValueError(f"registry has {len(registry)} conditions but the tracker holds "
                         f"{tracker['ema_logw'].numel()} slots: condition ids would not line up")
    for batch in (conditions, test):
        if batch is not None:
            _attach_mol_id(batch, registry)
    ef.set_n_molecules(max(len(registry), 1))

    # energy_config.energy_reference: log R is scored against a per-condition constant the
    # trainer computes at init (Modeller.init_energy_reference) and the checkpoint stores.
    # The stored table is installed, never recomputed: it is the one training scored against.
    if ef.energy_reference_mode is not None:
        ref = ck.get('energy_reference') or {}
        if ref.get('mode') != ef.energy_reference_mode or ref.get('table') is None:
            raise ValueError(f"config energy_reference is {ef.energy_reference_mode!r} but the checkpoint "
                             f"stores {ref.get('mode')!r}: log R would be in another currency")
        ef.set_energy_reference(ref['table'])

    return Run(gfn=gfn, energy_function=ef, conditions=conditions, test_conditions=test,
               registry=registry, tracker=tracker, step=int(ck['modeller_state']['step_ind']),
               stage=str(ck['modeller_state'].get('stage')),
               temperature=float(ec['temperature']), eval_T=int(cfg['eval_T']),
               device=device, config=cfg)


@torch.no_grad()
def head_log_z(run: Run, rows) -> torch.Tensor:
    """The learned Z head's log Z(c) for each row; no rollout needed."""
    rows = rows.to(run.device)
    _, _, cond, _ = run.energy_function.condition_samples(rows)
    emb = run.gfn.get_condition_embedding(cond.to(run.device), rows)
    return run.gfn._condition_flow(emb).flatten().cpu()


@torch.no_grad()
def draw_latents(run: Run, rows, batch_size: int = 2048, seed: Optional[int] = None) -> torch.Tensor:
    """One forward draw per row of `rows` with no energy call: the raw terminal states,
    [n, data_ndim] on the CPU. The rollout is the eval's (GFN.get_traj_fwd on a uniform grid
    of cfg:eval_T steps); nothing is scored. `crystals_from` builds the crystals."""
    if seed is not None:
        torch.manual_seed(seed)
    ef = run.energy_function
    out = []
    for start in range(0, rows.num_graphs, batch_size):
        mol_batch = rows.subsample_new_batch(torch.arange(start, min(start + batch_size, rows.num_graphs))).to(run.device)
        mol_batch.orient_molecule(mode='standard')
        bsz = mol_batch.num_graphs
        temperatures = run.temperature * torch.ones(bsz, dtype=torch.float32, device=run.device)
        mol_batch, _, condition, _ = ef.condition_samples(mol_batch, temperature=temperatures)
        states, *_ = run.gfn.get_traj_fwd(get_gfn_init_state(bsz, ef.data_ndim, run.device),
                                          lambda b: uniform_discretizer(b, run.eval_T), None, condition.to(run.gfn.device), mol_batch)
        out.append(states[:, -1].detach().cpu())
    return torch.cat(out)


@torch.no_grad()
def crystals_from(run: Run, rows, terminals, scored: bool = True, batch_size: int = 1024):
    """The crystals the trainer builds from `terminals`, one row of `rows` each (CPU batch).
    With `scored` they come out of the reward function, as `draw` returns them, at one energy
    call each. Without, they come from MolecularCrystal.instantiate_crystals alone: cell and
    pose set, nothing analysed, no energy call."""
    ef = run.energy_function
    out = None
    for start in range(0, rows.num_graphs, batch_size):
        mol_batch = rows.subsample_new_batch(torch.arange(start, min(start + batch_size, rows.num_graphs))).to(run.device)
        mol_batch.orient_molecule(mode='standard')
        temperatures = run.temperature * torch.ones(mol_batch.num_graphs, dtype=torch.float32, device=run.device)
        mol_batch, log_T, _, _ = ef.condition_samples(mol_batch, temperature=temperatures)
        x = terminals[start:start + batch_size].float().to(run.device)
        if scored:
            _, part = ef.log_reward(x, mol_batch=mol_batch, log_temperature=log_T, return_exp=True)
        else:
            part = ef.instantiate_crystals(x, mol_batch)
        part = part.cpu().detach()
        out = part if out is None else out.append_batch(part)
    return out


@torch.no_grad()
def draw(run: Run, rows, batch_size: int = 1000, seed: Optional[int] = None) -> dict:
    """One forward draw per row of `rows`, through the eval's own sampling function.

    Returns CPU tensors, one entry per draw: log_pf, log_pb, log_r and log_w = log_r + log_pb
    - log_pf (the per-trajectory log weight the eval's Z estimates pool), the head's log Z
    for the row's condition (head_log_z), condition_id, the raw terminal state, the fields
    in SAMPLE_FIELDS found on the scored batch, and the scored crystals as `sample_batch`.
    """
    if seed is not None:
        torch.manual_seed(seed)
    disc = lambda bsz: uniform_discretizer(bsz, run.eval_T)
    parts, batches = [], []
    for start in range(0, rows.num_graphs, batch_size):
        idx = torch.arange(start, min(start + batch_size, rows.num_graphs))
        mol_batch = rows.subsample_new_batch(idx).to(run.device)
        mol_batch.orient_molecule(mode='standard')
        bsz = mol_batch.num_graphs
        init_state = get_gfn_init_state(bsz, run.energy_function.data_ndim, run.device)
        temperatures = run.temperature * torch.ones(bsz, dtype=torch.float32, device=run.device)
        out = sample_eval_fwd_trajs(init_state, run.gfn, disc, run.energy_function, mol_batch,
                                    no_conditioning=False, temperatures=temperatures)
        batches.append(out.pop('sample_batch'))
        parts.append(out)

    sample_batch = batches[0]
    for b in batches[1:]:
        sample_batch = sample_batch.append_batch(b)
    cat = lambda k: torch.cat([p[k] for p in parts])
    res = {
        'log_pf': cat('log_pf'),
        'log_pb': cat('log_pb'),
        'log_r': cat('log_r').flatten(),
        'head_log_z': cat('log_flow')[:, 0],
        'condition_id': cat('condition_id'),
        'terminal_raw': cat('flow_states')[:, -1],
    }
    res['log_w'] = res['log_r'] + res['log_pb'] - res['log_pf']
    absent = []
    for name in SAMPLE_FIELDS:
        value = getattr(sample_batch, name, None)
        if torch.is_tensor(value) and value.numel() == sample_batch.num_graphs:
            res[name] = value.detach().flatten().cpu()
        else:
            absent.append(name)
    res['absent_fields'] = absent
    res['identifier'] = list(sample_batch.identifier)
    res['sample_batch'] = sample_batch
    return res


def pooled_levels(log_w: torch.Tensor) -> dict:
    """The eval's pooled Z readings over finite rows (fwd_eval_sampling): jensen_z is the
    mean log w (log_Z_lb), emp_z its log-mean-exp (log_Z), with the Jensen mean's standard
    error and the importance sampler's effective-sample fraction beside them."""
    lw = log_w.double()
    finite = torch.isfinite(lw)
    lw = lw[finite]
    n = lw.numel()
    w = torch.softmax(lw, 0)
    return {
        'n': n,
        'nonfinite': int((~finite).sum()),
        'jensen_z': float(lw.mean()),
        'jensen_z_se': float(lw.std() / math.sqrt(n)) if n > 1 else float('nan'),
        'emp_z': float(logmeanexp(lw)),
        'ess_frac': float(1.0 / (w.pow(2).sum() * n)) if n else float('nan'),
        'logw_std': float(lw.std()) if n > 1 else float('nan'),
    }
