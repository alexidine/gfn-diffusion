"""
Paper-1 sampling analysis against the CURRENT single-file checkpoints.

Differences from analysis.py, all in the load/sample/decode front end:
  * one checkpoint file (checkpointing.py) instead of *_model_config.npy + bare
    state dict: GFN is rebuilt from ck['gfn_config'], weights from ck['model_eval']
  * rollout length is ck['train_T'], never a yaml number
  * energy function, space group and Z' come from ck['problem_def']
  * get_traj_fwd takes mol_batch (None for an unconditional model)
  * the prior file's batch is 'equalized_prior' (what training consumes); the
    MOLECULE comes from its 'prior' key, which is what train.py conditions on
  * eLJ is scaled INSIDE compute_eLJ_energy by the batch's own per-graph
    lj_coeff, so the thermal scaling factor is stamped on the batch before
    analyze() and never multiplied on afterwards
The analysis after decoding (RDF distance matrix, local density, mean-shift
basins, combo figure) is unchanged and imported from utils / figures.
"""
import gc
import os
from time import sleep

import numpy as np
import torch
from tqdm import tqdm

from energy_sampling.eval.paper1_results.figures import combo_fig
from energy_sampling.eval.paper1_results.utils import (
    dmat_local_analysis, combo_fig_analysis, clustering)
from energy_sampling.models import GFN
from energy_sampling.utils import (load_yaml, dict2namespace, uniform_discretizer,
                                   get_gfn_init_state, is_cuda_oom)
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance, compute_rdf_distmat
# which triclinic walls this process scores with: sg 1/2 use tri_niggli_reduction_penalty
# unless MXT_LEGACY_TRICLINIC_WALLS=1 was set BEFORE mxtaltools was imported. It must match
# the walls the arm trained under, so it is recorded with the results and checked on reload.
from mxtaltools.common.sym_utils import LEGACY_TRICLINIC_WALLS
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

# The MLIP the run was TRAINED against is named in the arm config's mlip_path
# (a cluster path); these are the local copies of the same files. The checkpoint
# does not record it, so a mismatch here is silent -- see MLIP_PATHS use below.
UMA_PATH = r"D:\crystal_datasets\esen_s.pt"
MACE_PATH = r"D:\crystal_datasets\acr_112025_mh1_stagetwo.model"


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def load_gfn(checkpoint_path, device):
    """Rebuild the EMA model exactly as train.py does for prior_model_name."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gfn_config = {**ck['gfn_config'], 'device': device}
    model = GFN(**gfn_config).to(device)
    model.load_state_dict(ck['model_eval'])
    # Under freeze_pb:full, training evaluates P_B on the checkpoint's frozen snapshot, not on the live head in
    # model_eval (which kept drifting with the shared trunk). Without this, every log P_B read here -- the TB parity
    # figure's x axis -- is off by 1-2.4 nats with ~2 nats of per-path scatter (measured 2026-09-28 on p23 mipu/nehu/acr).
    pb_frozen = ck.get('pb_frozen') is not None
    if pb_frozen:
        model.freeze_backward_policy(source_state=ck['pb_frozen'])
    model.eval()
    train_T = ck.get('train_T')
    assert train_T is not None, (
        f"{checkpoint_path} carries no train_T; refusing to guess a rollout length")
    ms = ck.get('modeller_state', {})
    # the stored `tag` is whatever tag wrote those bytes (best is a hardlink to
    # running), so print the FILENAME's tag too or a correct load reads as wrong
    file_tag = os.path.basename(checkpoint_path).rsplit('_', 1)[-1].replace('.pt', '')
    prov = {'checkpoint_file': os.path.basename(checkpoint_path),
            'file_tag': file_tag, 'stored_tag': ck.get('tag'),
            'run_name': ck.get('run_name'), 'step_ind': ms.get('step_ind'),
            'stage': ms.get('stage'), 'train_T': int(train_T), 'pb_frozen_installed': pb_frozen,
            # anchor of a stage coeff_schedule ramp (protocol.py energy_coeffs); None if the stage has none
            'coeff_sched_entry': (ms.get('stage_ctrl') or {}).get('coeff_sched_entry')}
    print(f"loaded {prov['checkpoint_file']}: file_tag={file_tag} "
          f"(stored tag={ck['tag']}) run={ck['run_name']} step={prov['step_ind']} "
          f"stage={prov['stage']} train_T={train_T}")
    return model, ck['problem_def'], int(train_T), prov


def load_predictor(energy_function, device):
    if energy_function == 'uma':
        return init_uma_crystal_predictor(UMA_PATH, device=device)
    if energy_function == 'mace':
        return load_mace_model(MACE_PATH, device=device, dtype=torch.float32)
    return None


# ----------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------
@torch.no_grad()
def sample_latents(model, num_samples, batch_size, n_steps, device):
    ndim = model.dim
    discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)
    states, pfs, pbs = [], [], []
    with tqdm(total=num_samples, desc='sampling') as pbar:
        n = 0
        while n < num_samples:
            bsz = min(batch_size, num_samples - n)
            init_state = get_gfn_init_state(bsz, ndim, device)
            # condition is ignored by an unconditional model; mol_batch is only
            # read by the molecule-conditioned route
            condition = torch.zeros((bsz, 1), device=device)
            x, log_pf, log_pb, _ = model.get_traj_fwd(
                init_state, discretizer, None, condition, None,
                return_gauss_params=False)
            states.append(x[:, -1].cpu())
            pfs.append(log_pf.sum(-1).cpu())
            pbs.append(log_pb.sum(-1).cpu())
            n += bsz
            pbar.update(bsz)
    # wrapped dims are already on [-1, 1]; the clip only touches the box dims.
    # Structurally dead rows are pinned by the model and must survive this.
    out = torch.cat(states)
    dead = tuple(getattr(model, 'dead_rows', ()) or ())
    for r in dead:
        spread = float(out[:, r].max() - out[:, r].min())
        assert spread < 1e-6, (
            f"dead latent row {r} varies by {spread:.3e} across samples -- the "
            f"pinning that the checkpoint's gfn_config declares did not survive "
            f"the reload, so these crystals are not in the trained subspace")
    if dead:
        print(f"dead latent rows {dead} confirmed pinned across {len(out)} samples")
    # the RAW terminal is kept too: the training reward's bounding term is scored on
    # what the policy emitted, before the clamp
    return out.clip(-1, 1), torch.cat(pfs), torch.cat(pbs), out


def stamp_lj_coeff(batch, lj_coeff):
    batch.lj_coeff = torch.full((batch.num_graphs,), float(lj_coeff),
                                dtype=torch.float32, device=batch.device)


def _computes_for(energy_function):
    return sorted(set(['lj', 'qlj', 'elj', 'silu', 'rdf', 'reduction_en', energy_function]))


# An aunit's stored rotvec rotates the molecule FROM its standard orientation, so
# posing without align_mol_batch_to_standard_axes is only valid when the molecule
# is ALREADY aligned. Train time gets away with std_orientation=False because its
# conditioning molecule is pre-standardized; experimental conformers are not.
# MEASURED 2026-09-07 on this acridine checkpoint (MACE, kJ/mol):
#   conditioning molecule, inertia off-diag 0.014 -> True vs False differs by 0.006
#   ACRDIN04/ACRDIN12,     inertia off-diag 70.4  -> True vs False differs by 3826
#                                                    (-60.06 becomes +3765.75)
# So True is correct for both legs, and False is a silent catastrophe on any
# structure carrying its own conformer frame. This guard makes that measurable
# rather than a comment.
ORIENTATION_TOL = 1.0


def _assert_orientation_safe(crystals, std_orientation, tag):
    if std_orientation:
        return
    worst, worst_id = 0.0, None
    for c in crystals:
        p = c.pos - c.pos.mean(0)
        inertia = p.T @ p
        off = float((inertia - torch.diag(torch.diag(inertia))).abs().max())
        if off > worst:
            worst, worst_id = off, getattr(c, 'identifier', '?')
    print(f"  {tag}: worst inertia off-diagonal {worst:.4f} (std_orientation off)")
    if worst > ORIENTATION_TOL:
        raise ValueError(
            f"{tag} structure {worst_id} has inertia off-diagonal {worst:.3f}, so it is "
            f"NOT in standard orientation, and std_orientation=False would pose its "
            f"asymmetric unit from the wrong frame -- measured error up to 3826 kJ/mol. "
            f"Set std_orientation: true.")


#: max |d| between sorted heavy-atom distance fingerprints for "same conformer".
#: The two as-deposited acridine files differ by 0.199 A, so this cannot confuse them.
CONFORMER_TOL = 1e-3


def _conformer_fingerprint(crystal):
    """Sorted intramolecular heavy-atom distances of ONE molecule: invariant to
    rotation, translation and atom order, so it identifies the conformer alone."""
    n_mol = int((crystal.z > 1).sum()) // int(crystal.z_prime)
    p = crystal.pos[crystal.z > 1][:n_mol]
    iu = torch.triu_indices(n_mol, n_mol, 1)
    return torch.cdist(p, p)[iu[0], iu[1]].sort().values


def score_embedding_references(config, rdf_mode, std_orientation, sample_crystal, rdf_cutoff=10):
    """Structures placed on the UMAP ONLY -- never in the table, densities or basin
    assignment. Unlike score_experimental there is no (space group, Z') filter:
    forms the model cannot generate are exactly what these are for.

    `embedding_references` is a list of {path, tag}; every form in each file is
    used. The conformer must match the SAMPLES', because the RDF moves with molecule
    size: std_opt_acridine_polymorphs.pt and std_acridine_polymorphs.pt share cells
    but carry conformers 0.199 A apart, and only one of them is this model's."""
    ref_fp = _conformer_fingerprint(sample_crystal)
    # atomwise channels are atom PAIRS of the asymmetric unit, so a Z'=2 form has 1081 channels against Z'=1's 276
    # and is not comparable at all; envwise channels do not depend on Z'. Such forms are skipped, by name.
    want_shape = tuple(sample_crystal.rdf.shape[-2:])
    rdfs, labels, tags, sampleable, skipped = [], [], [], [], []
    for spec in config.embedding_references:
        crystals = torch.load(spec['path'], map_location='cpu', weights_only=False)
        crystals = crystals.batch_to_list() if getattr(crystals, 'is_batch', False) else crystals
        _assert_orientation_safe(crystals, std_orientation, f"embedding references ({spec['tag']})")
        for c in crystals:
            gap = float((_conformer_fingerprint(c) - ref_fp).abs().max())
            if gap > CONFORMER_TOL:
                raise ValueError(
                    f"{c.identifier} in {spec['path']} carries a different conformer from the "
                    f"samples (max fingerprint gap {gap:.4f} A > {CONFORMER_TOL}); its RDF is not "
                    f"comparable. Use the file whose molecule matches the model's.")
            # one structure per call: the forms mix space groups and Z', which do not collate
            b = collate_data_list([c]).to(config.device)
            b.analyze(['rdf'], cutoff=rdf_cutoff, rdf_cutoff=rdf_cutoff, std_orientation=std_orientation,
                      assign_outputs=True, rdf_mode=rdf_mode)
            if tuple(b.rdf[0].shape) != want_shape:
                skipped.append(f"{c.identifier} (Z'={int(c.z_prime)}, {rdf_mode} rdf {tuple(b.rdf[0].shape)})")
                continue
            rdfs.append(b.rdf[0].cpu())
            labels.append(str(c.identifier))
            tags.append(spec['tag'])
            sampleable.append((int(c.sg_ind), int(c.z_prime)))
        print(f"  embedding references ({spec['tag']}): {len(crystals)} from "
              f"{os.path.basename(spec['path'])}, conformer matches the samples")
    if skipped:
        print(f"  embedding references SKIPPED, rdf not comparable with the samples' {want_shape}: "
              + ', '.join(skipped))
    if not rdfs:
        return None
    return {'rdf': torch.stack(rdfs), 'labels': labels, 'tags': tags, 'cells': sampleable}


def _drop_nonfinite(crystals, energy_function, tag):
    """A non-OOM MLIP failure substitutes NaN by contract (AL_mace_utils
    _crashed_energy). Without this filter those rows flow into the energy
    filter, get dropped by the `<` comparison, and the run reports a bare
    'Data list is empty!' assertion from collate -- a swallowed diagnostic that
    reads as a plotting bug rather than 'MACE crashed on N structures'."""
    keep, keep_idx, bad = [], [], 0
    for i, c in enumerate(crystals):
        v = getattr(c, energy_function, None)
        if v is None or not bool(torch.isfinite(torch.as_tensor(v)).all()):
            bad += 1
        else:
            keep.append(c)
            keep_idx.append(i)
    if bad:
        print(f"  !! {bad}/{len(crystals)} {tag} rows had non-finite {energy_function} "
              f"(MLIP crash) and were dropped")
    assert keep, f"every {tag} row had a non-finite {energy_function}; the MLIP failed wholesale"
    return keep, torch.tensor(keep_idx, dtype=torch.long)


def decode_and_score(latents, molecule, max_z_prime, sg_ind, zp, energy_function,
                     lj_coeff, predictor, batch_size, device, std_orientation,
                     max_batch_size, rdf_mode):
    """latents -> crystals -> energies/RDF, batched with OOM back-off."""
    num_samples = len(latents)
    computes = _computes_for(energy_function)
    _assert_orientation_safe([molecule], std_orientation, 'conditioning molecule')
    samples = []
    cursor = 0
    already_oomed = False
    with tqdm(total=num_samples, desc='scoring') as pbar, torch.no_grad():
        while cursor < num_samples:
            batch = None
            try:
                inds = np.arange(cursor, min(num_samples, cursor + batch_size))
                mols = [molecule.clone() for _ in inds]
                for m in mols:
                    m.z_prime = zp
                batch = collate_data_list(mols, max_z_prime=max_z_prime).to(device)
                batch.reset_sg_info(sg_ind)
                batch.latent_to_cell_params(latents[inds].to(device))
                stamp_lj_coeff(batch, lj_coeff)
                batch.analyze(computes, cutoff=10, std_orientation=std_orientation,
                              assign_outputs=True, predictor=predictor, rdf_mode=rdf_mode)
                batch = batch.to('cpu')
                samples.extend(batch.batch_to_list())
                del batch
                batch = None
                cursor += len(inds)
                pbar.update(len(inds))
                if batch_size < min(num_samples, max_batch_size) and not already_oomed:
                    batch_size = min(batch_size + max(int(batch_size * 0.05), 1), max_batch_size)
            except (RuntimeError, ValueError) as e:
                if not is_cuda_oom(e):
                    raise
                assert batch_size > 1, "cascading OOM in crystal scoring"
                # release the batch that just OOMed BEFORE emptying the cache,
                # otherwise the retry starts with it still resident
                del batch
                batch = None
                batch_size = max(int(batch_size * 0.65), 1)
                print(f"OOM in scoring: batch size -> {batch_size}")
                already_oomed = True
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sleep(0.1)
    # returns (kept crystals, their indices into `latents`) so per-sample tensors
    # can be filtered in step
    return _drop_nonfinite(samples, energy_function, 'sample')


def score_experimental(config, energy_function, lj_coeff, predictor, device,
                       std_orientation, sg_ind, zp, batch_size, rdf_mode):
    """Experimental references, scored on the SAME footing as the samples.

    Both legs are computed fresh here, so any constant MLIP reference-energy
    offset cancels in the sample-vs-experiment comparison."""
    exp = torch.load(config.exp_sample_path, map_location='cpu', weights_only=False)
    if getattr(exp, 'is_batch', False):
        exp = exp.batch_to_list()
    elif not isinstance(exp, list):
        exp = [exp]

    identifiers = getattr(config, 'identifiers', None)
    if identifiers:
        found = [e for e in exp if e.identifier in identifiers]
        got = [e.identifier for e in found]
        missing = [i for i in identifiers if i not in got]
        assert not missing, (
            f"identifiers {missing} not present in {config.exp_sample_path}; "
            f"it holds {sorted({e.identifier for e in exp})}")
        exp = found
    else:
        print(f"  no `identifiers` filter: using all {len(exp)} references in the file")

    # A reference the model structurally cannot produce is not a reference. The
    # acridine polymorph file mixes sg 9/14/19 and Z' 1/2/3 against an sg14 Z'=1
    # model; scoring those as "experimental" silently compares different problems.
    bad = [(e.identifier, int(e.sg_ind), int(e.z_prime)) for e in exp
           if int(e.sg_ind) != sg_ind or int(e.z_prime) != zp]
    assert not bad, (
        f"experimental references {bad} do not match the checkpoint's cell "
        f"(sg {sg_ind}, Z' {zp}). Filter them out with `identifiers`, or analyse "
        f"a checkpoint trained on their cell.")
    print(f"  experimental references: {[e.identifier for e in exp]} "
          f"(sg {sg_ind}, Z'={zp})")

    _assert_orientation_safe(exp, std_orientation, 'experimental references')
    computes = _computes_for(energy_function)
    scored = []
    with torch.no_grad():
        for i in range(0, len(exp), batch_size):
            eb = collate_data_list(exp[i:i + batch_size]).to(device)
            stamp_lj_coeff(eb, lj_coeff)
            eb.analyze(computes, cutoff=10, std_orientation=std_orientation,
                       assign_outputs=True, predictor=predictor, rdf_mode=rdf_mode)
            scored.extend(eb.to('cpu').batch_to_list())
            del eb
            torch.cuda.empty_cache()
    scored, _ = _drop_nonfinite(scored, energy_function, 'experimental')
    return collate_data_list(scored)


def reference_proximity(ebatch, sample_batch, d_cut, bins, poly_labels, row_to_sample):
    """How close does the model actually GET to each experimental form?

    The pipeline's only statement about experimental structures is
    polymorph_basin_index -- a thresholdless argmin (utils.py) that assigns and
    colours a reference into a basin however far away it is. This measures the
    thing the assignment assumes: the RDF distance from each reference to its
    NEAREST sample, in units of the pipeline's own neighbourhood radius d_cut.
    A ratio above 1 means no sample is within the radius that defines a
    neighbourhood, i.e. the model did not reach that form at this sample count.

    sample_batch is the ENERGY-FILTERED set, so argmin is a row of it; row_to_sample
    (good_ens) maps that row back to the index in results['samples']. Before this the
    row was stored as 'nearest_sample' -- on p12 acridine, row 3847 is sample 4200.
    """
    out = []
    print(f"  reference proximity (d_cut = {float(d_cut):.4f}):")
    for i in range(ebatch.num_graphs):
        d = compute_rdf_distance(ebatch.rdf[i], sample_batch.rdf, bins)
        dmin = float(d.min())
        row = int(d.argmin())
        j = int(row_to_sample[row])
        ratio = dmin / float(d_cut)
        label = poly_labels[i] if poly_labels else str(i)
        verdict = 'INSIDE d_cut' if ratio <= 1 else 'OUTSIDE d_cut -- not reached'
        print(f"    {label}: nearest sample {dmin:.4f} = {ratio:.2f} x d_cut "
              f"(sample {j}, filtered row {row})  <== {verdict}")
        out.append({'label': label, 'min_rdf_distance': dmin, 'd_cut': float(d_cut),
                    'ratio_to_d_cut': ratio, 'nearest_sample': j, 'nearest_filtered_row': row,
                    'reached': ratio <= 1})
    return out


def export_structures(basin_min_batch, ebatch, results_dir, run_name, poly_labels):
    """Write the per-basin energy minima and the experimental references as CIFs.

    Without this the pipeline emits a picture and nothing anyone can open,
    overlay against the CSD entry, or deposit. Exports basin_min_batch (the
    per-basin ENERGY minima that combo_fig_analysis already selected) rather
    than re-indexing p_maxima into a sample list, which is where the old
    generate_figs export went wrong."""
    from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
    out_dir = os.path.join(results_dir, f"{run_name}_structures")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for batch, names in ((basin_min_batch, [f"basin{i + 1}" for i in range(basin_min_batch.num_graphs)]),
                         (ebatch, [f"ref_{l}" for l in (poly_labels or
                                                        [str(i) for i in range(ebatch.num_graphs)])])):
        try:
            cluster = batch.clone()
            cluster.mol2ucell()
        except Exception as e:
            print(f"  cif export skipped for one batch: {e}")
            continue
        for i, name in enumerate(names):
            try:
                mol = ase_mol_from_crystaldata(cluster, i, mode='unit cell')
                path = os.path.join(out_dir, f"{name}.cif")
                mol.write(path)
                written.append(path)
            except Exception as e:
                print(f"  cif export failed for {name}: {e}")
    print(f"  wrote {len(written)} cif files to {out_dir}")
    return written


def _as_dict(node):
    """A yaml mapping read through the config loader may arrive as a dict or a namespace."""
    if node is None or isinstance(node, dict):
        return node
    return dict(vars(node))


def live_energy_coeffs(cfg, stage, step, entry):
    """The energy_config coefficients the checkpoint was TRAINING at: the stage's coeff_schedule
    ramp evaluated at its step, with the same formula as protocol.py energy_coeffs. The yaml's
    energy_config holds only the ramp's BASE, so without this a ramped arm is scored at the wrong
    penalty. {} when the stage has no schedule."""
    stages = cfg['protocols'][cfg['protocol']]['stages']
    spec_stage = [s for s in stages if s['name'] == stage]
    sched = (spec_stage[0].get('coeff_schedule') or {}) if spec_stage else {}
    if not sched:
        return {}
    if step is None or entry is None:
        raise ValueError(f"stage {stage!r} ramps {sorted(sched)} but the checkpoint carries no "
                         f"step ({step}) or coeff_sched_entry ({entry}); refusing to score at the base value")
    out = {}
    for name, spec in sched.items():
        base = float(cfg['energy_config'][name])
        p = min(1.0, max(0.0, (int(step) - int(entry)) / float(spec['steps'])))
        out[name] = (base * (spec['target'] / base) ** p if spec.get('kind', 'geometric') == 'geometric'
                     else base + (spec['target'] - base) * p)
    return out


def exact_log_rewards(samples, raw_latents, train_config_path, energy_function, lj_coeff,
                      prior_file, device, chunk=500, stage=None, step=None, sched_entry=None):
    """The TRAINING reward log R for every sample, from the arm's own MolecularCrystal.

    Replaces utils.generator_reward, a frozen copy of an old molecular_crystal
    reward (no Jacobian, and other terms since changed). prebuilt_sample_to_reward
    reads the energy already stored on each crystal, so this makes NO MLIP calls --
    the Jacobian, bounding, reduction and pressure terms come from geometry.

    The reward clip. Training sets it from the COMPOSITE energy of its re-analysed
    prior (set_reward_clip: clip = min(E) + reward_range * T). That pass would need
    the MLIP over the whole prior, so it is approximated here as
        min(prior lattice energy) + median(composite - lattice) over these samples,
    and the number of samples anywhere near the clip is reported. The clip is
    ~reward_range*T = 625 kJ/mol above the minimum, so it binds only on junk.
    """
    import yaml
    from energy_sampling.energies.molecular_crystal import MolecularCrystal

    cfg = yaml.safe_load(open(train_config_path))
    assert cfg['energy_function'] == energy_function, (
        f"{train_config_path} trains {cfg['energy_function']!r}, checkpoint is {energy_function!r}")
    ec = dict(cfg['energy_config'])
    live = live_energy_coeffs(cfg, stage, step, sched_entry)
    if live:
        print(f"  coeff_schedule live at step {step} (anchor {sched_entry}): "
              + ', '.join(f"{k} {ec[k]:g} -> {v:g}" for k, v in live.items()))
        ec.update(live)
    ec['internal_oom_recovery'] = False
    ef = MolecularCrystal(device=device, energy_function=energy_function,
                          mlip_path={'uma': UMA_PATH, 'mace': MACE_PATH}.get(energy_function),
                          space_groups=cfg['space_groups'], z_primes=cfg['z_primes'],
                          sg_conditioning=cfg['sg_conditioning'],
                          temperature_conditioning=cfg['temperature_conditioning'],
                          zp_conditioning=cfg['zp_conditioning'],
                          vector_conditioning=cfg.get('vector_conditioning', False),
                          vector_conditioning_dim=cfg.get('vector_conditioning_dim'),
                          embedding_conditioning=cfg.get('embedding_conditioning', False),
                          embedding_conditioning_dim=cfg.get('embedding_conditioning_dim'),
                          **ec)
    ef.lj_coeff = lj_coeff
    T = float(ec['temperature'])

    def score():
        out = []
        for i in range(0, len(samples), chunk):
            b = collate_data_list([s.clone() for s in samples[i:i + chunk]]).to(device)
            # the stored .elj is already scaled; the stamp only certifies its currency
            stamp_lj_coeff(b, lj_coeff)
            with torch.no_grad():
                r = ef.prebuilt_sample_to_reward(b, T, raw_latents=raw_latents[i:i + chunk].to(device))
            out.append(r.detach().flatten().double().cpu())
            del b
        return torch.cat(out)

    lattice = torch.tensor([float(getattr(s, energy_function)) for s in samples], dtype=torch.float64)
    log_r_unclipped = score()
    composite = -log_r_unclipped * T
    offset = float((composite - lattice)[torch.isfinite(composite)].median())

    prior_e = getattr(prior_file['equalized_prior'], energy_function).double()
    prior_e = prior_e[torch.isfinite(prior_e)]
    # stored eLJ prior energies are RAW sums (measured 2026-09-15: mipcas min -387.8,
    # x0.3636 = -141.0); MLIP priors are stored on the working scale
    prior_min = float(prior_e.min()) * (lj_coeff if energy_function == 'elj' else 1.0)
    ef.set_reward_clip([-(prior_min + offset) / T])
    log_r = score()

    clip = float(ef.energy_clip)
    near = int((composite > clip - 50).sum())
    print(f"  exact training reward: composite - lattice energy median {offset:+.2f}; "
          f"clip {clip:.1f} (prior min {prior_min:.2f}); "
          f"{near}/{len(samples)} samples within 50 kJ/mol of the clip or above it; "
          f"max composite {float(composite.max()):.1f}")
    del ef
    torch.cuda.empty_cache()
    info = {'train_config': os.path.basename(train_config_path), 'temperature': T, 'live_coeffs': live,
            'energy_clip': clip, 'composite_minus_lattice_median': offset,
            'n_near_or_above_clip': near}
    return log_r, info


def resolve_molecule(config, prior_file):
    """The molecule the model was CONDITIONED on.

    train.py points molecules_path at the prior dataset and unwraps its 'prior'
    key (_load_condition_file), so that is the geometry the policy was fit
    against. A stand-alone conformer file on disk is a DIFFERENT geometry, and
    swapping it changes every crystal built here while nothing errors.
    `molecule_path` overrides, loudly."""
    override = getattr(config, 'molecule_path', None)
    ref = prior_file['prior'].batch_to_list()[0]
    if not override:
        print(f"  molecule: prior['prior'][0] (identifier {ref.identifier}), "
              f"matching train.py's conditioning source")
        return ref
    mol = torch.load(override, map_location='cpu', weights_only=False)
    mol = mol[0] if isinstance(mol, list) else mol
    print(f"  molecule: OVERRIDE {os.path.basename(override)} "
          f"(identifier {getattr(mol, 'identifier', '?')})")
    same_species = (mol.z.shape == ref.z.shape) and bool((mol.z == ref.z).all())
    if not same_species:
        raise ValueError(
            f"molecule_path {override} has a different atom list from the prior's "
            f"conditioning molecule -- it is not the same species/ordering, so the "
            f"model was never trained to place it")
    drift = float((mol.pos - ref.pos).abs().max())
    print(f"    max |pos| drift from the training molecule: {drift:.4f} A"
          + ("  <== GEOMETRY DIFFERS" if drift > 1e-3 else ""))
    return mol


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run_analysis(config):
    device = config.device
    results_path = os.path.join(config.results_dir, f"{config.run_name}.pt")
    checkpoint_path = os.path.join(config.checkpoints_dir, config.checkpoint)

    "problem identity comes from the checkpoint"
    model, problem_def, train_T, provenance = load_gfn(checkpoint_path, device)
    energy_function = problem_def['energy_function']
    space_groups, z_primes = problem_def['space_groups'], problem_def['z_primes']
    assert len(space_groups) == 1 and len(z_primes) == 1, (
        f"this analysis samples one (sg, Z') cell; checkpoint trained on {space_groups} x {z_primes}")
    sg_ind, zp = int(space_groups[0]), int(z_primes[0])
    max_z_prime = model.max_z_prime
    std_orientation = bool(getattr(config, 'std_orientation', True))
    # No default: the right mode is a property of the MOLECULE. atomwise for
    # mipcas/nehzor; envwise for acridine, whose C2 symmetry makes atomwise
    # non-monotone. For a 13-atom molecule both modes give 91 channels, so a
    # mismatch between runs raises nothing -- it just measures the wrong thing.
    rdf_mode = getattr(config, 'rdf_mode', None)
    assert rdf_mode in ('atomwise', 'envwise'), (
        f"run {config.run_name}: set rdf_mode to 'atomwise' or 'envwise' (got {rdf_mode!r})")
    print(f"problem: {energy_function} sg={sg_ind} Z'={zp} T_train={train_T} "
          f"std_orientation={std_orientation} rdf_mode={rdf_mode}")

    "the prior the model was trained against: molecule + eLJ unit factor"
    prior_file = torch.load(config.dataset_path, map_location='cpu', weights_only=False)
    lj_coeff = float(prior_file.get('thermal_scaling_factor', 1.0))
    if energy_function != 'elj' and abs(lj_coeff - 1.0) > 1e-9:
        raise ValueError(f"prior carries thermal_scaling_factor={lj_coeff} on a "
                         f"{energy_function} problem (mirrors train.py's check)")
    print(f"lj_coeff (thermal_scaling_factor) = {lj_coeff}")
    molecule = resolve_molecule(config, prior_file)

    predictor = load_predictor(energy_function, device)
    max_batch = int(getattr(config, 'max_analysis_batch_size', config.analysis_batch_size))

    "sample, or reload"
    reloaded = False
    # source_results: reload ANOTHER run's samples read-only; outputs still go to results_path
    source_path = getattr(config, 'source_results', None)
    load_path = source_path or results_path
    if source_path and not os.path.exists(source_path):
        raise FileNotFoundError(f"source_results {source_path} does not exist; refusing to resample under "
                                f"{config.run_name}")
    if (config.reload_results or source_path) and os.path.exists(load_path):
        cached = torch.load(load_path, map_location='cpu', weights_only=False)
        # provenance, not just presence: a cache scored under different settings
        # is a different measurement wearing the same filename. An ABSENT key
        # counts as stale -- the cache predates the setting, so it cannot be
        # assumed to match it.
        cached_step = (cached.get('provenance') or {}).get('step_ind')
        now = {'checkpoint': config.checkpoint, 'std_orientation': std_orientation,
               'lj_coeff': lj_coeff, 'train_T': train_T, 'rdf_mode': rdf_mode,
               'legacy_triclinic_walls': LEGACY_TRICLINIC_WALLS}
        stale = [f"{k}: cached {cached.get(k, '<absent>')!r} vs now {v!r}"
                 for k, v in now.items() if cached.get(k, object()) != v]
        # a re-downloaded checkpoint keeps its FILENAME, so the name alone would
        # silently reload last round's samples from different weights
        if cached_step != provenance['step_ind']:
            stale.append(f"checkpoint step: cached {cached_step!r} vs now {provenance['step_ind']!r}")
        if stale:
            if source_path:
                raise ValueError(f"source_results {source_path} does not match this run's configuration: {stale}")
            print("cache does not match this configuration, resampling:")
            for s in stale:
                print(f"    {s}")
        else:
            results_dict = cached
            samples = results_dict['samples']
            reloaded = True
            print(f"reloaded {len(samples)} samples from {load_path}")
    if not reloaded:
        latents, log_pf, log_pb, latents_raw = sample_latents(
            model, config.num_samples, config.sample_batch_size, train_T, device)
        samples, kept = decode_and_score(latents, molecule, max_z_prime, sg_ind, zp, energy_function,
                                         lj_coeff, predictor, config.analysis_batch_size, device,
                                         std_orientation, max_batch, rdf_mode)
        # keep every per-sample tensor aligned with `samples` if any MLIP call crashed
        latents, latents_raw, log_pf, log_pb = (latents[kept], latents_raw[kept],
                                                log_pf[kept], log_pb[kept])
        results_dict = {
            'checkpoint': config.checkpoint,
            'problem_def': problem_def,
            'train_T': train_T,
            'lj_coeff': lj_coeff,
            'std_orientation': std_orientation,
            'rdf_mode': rdf_mode,
            'legacy_triclinic_walls': LEGACY_TRICLINIC_WALLS,
            'sample_latents': latents,
            'sample_latents_raw': latents_raw,
            'log_pfs': log_pf,
            'log_pbs': log_pb,
            'samples': samples,
            'learned_log_z': model.flow_model().item(),
        }
        assert len(samples) == len(log_pf) == len(latents), "per-sample tensors out of step with samples"
    del model
    torch.cuda.empty_cache()

    # The old analysis.py saved the collated batch alongside the list, and
    # generate_figs reads it. Restored here. NB `.elj` on these is already scaled by
    # lj_coeff (the old files stored the raw sum); generate_figs checks for the
    # 'lj_coeff' key to tell the two apart.
    full_batch = collate_data_list(samples)
    results_dict['sample_batch'] = full_batch
    results_dict['sample_energy'] = full_batch[energy_function].detach().float()
    results_dict['sample_cp'] = full_batch.packing_coeff.detach().float()

    train_config = getattr(config, 'train_config', None)
    if train_config and not os.path.isabs(train_config):
        # relative to energy_sampling/, where configs/ lives
        train_config = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), train_config)
    if 'log_r' not in results_dict:
        if train_config:
            raw = results_dict.get('sample_latents_raw', results_dict['sample_latents'])
            if 'sample_latents_raw' not in results_dict:
                print("  cache has no raw latents; bounding term scored on clipped latents")
            results_dict['log_r'], results_dict['reward_info'] = exact_log_rewards(
                samples, raw, train_config, energy_function, lj_coeff, prior_file, device,
                stage=provenance['stage'], step=provenance['step_ind'],
                sched_entry=provenance.get('coeff_sched_entry'))
        else:
            print("  !! no train_config for this run: 'log_r' NOT written, "
                  "generate_figs will refuse the TB parity figure")
    del prior_file

    "experimental reference structures"
    ebatch = score_experimental(config, energy_function, lj_coeff, predictor, device,
                                std_orientation, sg_ind, zp, max_batch, rdf_mode)
    num_polymorphs = ebatch.num_graphs
    poly_labels = [str(i) for i in ebatch.identifier] if hasattr(ebatch, 'identifier') else None

    "energy filter, as before: keep everything within 15 kT of the best sample"
    sample_batch = collate_data_list(samples)
    sample_energy = sample_batch[energy_function]
    print(f"sampled {len(samples)}: E min {sample_energy.amin():.2f}, median "
          f"{sample_energy.median():.2f}; experimental "
          f"{[round(float(v), 2) for v in ebatch[energy_function]]}")
    threshold = float(sample_energy.amin() + 15 * config.kT)
    keep = sample_energy < threshold
    max_red = getattr(config, 'max_reduction_en', None)
    if max_red is not None:
        in_window = int(keep.sum())
        keep = keep & (sample_batch.reduction_en.flatten() <= float(max_red))
        print(f"reduction filter (reduction_en <= {max_red}): kept {int(keep.sum())} of {in_window} "
              f"samples in the 15 kT window")
    good_ens = keep.argwhere().flatten()
    sample_batch = collate_data_list([samples[i] for i in good_ens])
    sample_energy = sample_batch[energy_function]
    print(f"{len(good_ens)} samples within 15 kT of the minimum")

    "RDF distance matrix -- cached WITH the index set that produced it"
    # the cached dmat is reusable whenever the new rows are a SUBSET of the rows it was built on
    sub_idx = None
    if (reloaded and 'dmat' in results_dict and results_dict.get('good_ens') is not None
            and abs(results_dict.get('energy_threshold', 1e9) - threshold) < 1e-6):
        cached_rows = {int(g): i for i, g in enumerate(results_dict['good_ens'])}
        if all(int(g) in cached_rows for g in good_ens):
            sub_idx = torch.tensor([cached_rows[int(g)] for g in good_ens], dtype=torch.long)
    cached_ok = sub_idx is not None
    if cached_ok:
        dmat = results_dict['dmat']
        if len(sub_idx) != dmat.shape[0] or not bool((sub_idx == torch.arange(dmat.shape[0])).all()):
            dmat = dmat[sub_idx][:, sub_idx]
            # written back: clustering() and every saved results file read results_dict['dmat'],
            # which must describe the SAME rows as good_ens
            results_dict['dmat'] = dmat
            print(f"reusing cached dmat, subset to {len(sub_idx)} rows")
        else:
            print("reusing cached dmat")
    else:
        if reloaded and 'dmat' in results_dict:
            print("cached dmat does not match the current energy filter; recomputing")
        rdf_bins = torch.linspace(0, 10, sample_batch.rdf.shape[-1])
        with torch.no_grad():
            dmat = compute_rdf_distmat(sample_batch.rdf.to(device), rdf_bins.to(device)).cpu()
        results_dict['dmat'] = dmat
    assert results_dict['dmat'].shape[0] == len(good_ens), (
        f"dmat has {results_dict['dmat'].shape[0]} rows but the filters kept {len(good_ens)}")
    results_dict['provenance'] = provenance
    results_dict['mlip_path'] = {'mace': MACE_PATH, 'uma': UMA_PATH}.get(energy_function)
    results_dict['n_good'] = len(good_ens)
    results_dict['energy_threshold'] = threshold
    results_dict['good_ens'] = good_ens
    results_dict['max_reduction_en'] = max_red

    "neighbourhood analysis"
    n_probe = min(50, sample_batch.num_graphs)
    bins = torch.linspace(0, 10, sample_batch.rdf.shape[-1])
    # one probe row materialises [n_samples, channels, bins] floats: on NEHZOR atomwise
    # (~300 channels) that is 1.2 GB per probe, which killed this on the CPU allocator
    # 2026-09-23. Score on `device` in sample chunks and keep only the distances.
    with torch.no_grad():
        bins_dev = bins.to(device)
        rdf_dev = sample_batch.rdf.to(device)
        probe_chunk = max(1, min(len(rdf_dev), int(2e8 // max(1, rdf_dev[0].numel()))))
        d = torch.cat([compute_rdf_distance(rdf_dev[ii], rdf_dev[c:c + probe_chunk], bins_dev).cpu()
                       for ii in range(n_probe)
                       for c in range(0, len(rdf_dev), probe_chunk)])
        del rdf_dev
        torch.cuda.empty_cache()
    d_cuts = [d.quantile(0.15)]
    with torch.no_grad():
        d_metrics = [dmat_local_analysis(sample_energy, dmat, d_cut=d_cut, d_kernel=d_cut / 3, e_cut=None)
                     for d_cut in d_cuts]
    results_dict.update({'metrics': d_metrics, 'd_cuts': d_cuts})

    "did the model actually reach the experimental forms?"
    results_dict['reference_proximity'] = reference_proximity(
        ebatch, sample_batch, d_cuts[0], bins, poly_labels, good_ens)

    if config.save_results and (config.overwrite_results or not os.path.exists(results_path)):
        torch.save(results_dict, results_path)
        print(f"saved {results_path}")

    if not getattr(config, 'make_figures', True):
        # data only: the paper figures come from generate_figs.py
        return results_dict

    "basins + figure"
    thermos = results_dict['metrics'][0]
    cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering(
        results_dict, thermos, max_n_clusters=config.max_n_clusters,
        min_basin_size=config.min_basin_size)
    print(f"{n_basins} basins")

    extra_refs = None
    if getattr(config, 'embedding_references', None):
        extra_refs = score_embedding_references(config, rdf_mode, std_orientation, samples[0])
        extra_refs['model_cell'] = (int(sg_ind), int(zp))

    # The table's LJ row is a density in the eLJ RUN's cloud (its samples, its dmat). Passing this
    # run in both slots -- which is what happened until 2026-09-23 -- made that row a copy of the
    # UMA density under an LJ label. `elj_results_path` names the partner eLJ results file, exactly
    # the pairing generate_figs.py does; without it the figure carries one density row.
    elj_path = getattr(config, 'elj_results_path', None)
    if elj_path:
        elj_results = torch.load(elj_path, map_location='cpu', weights_only=False)
        assert elj_results.get('rdf_mode') == rdf_mode, (
            f"partner {os.path.basename(elj_path)} has rdf_mode {elj_results.get('rdf_mode')!r}, "
            f"this run is {rdf_mode!r}: the two clouds would be measured with different distances")
        assert elj_results['problem_def']['energy_function'] == 'elj', (
            f"partner {os.path.basename(elj_path)} is "
            f"{elj_results['problem_def']['energy_function']!r}, not an eLJ run")
        for k in ('space_groups', 'z_primes'):
            assert list(elj_results['problem_def'][k]) == list(problem_def[k]), (
                f"partner {os.path.basename(elj_path)} {k} {elj_results['problem_def'][k]} != {problem_def[k]}")
        assert 'dmat' in elj_results, f"partner {os.path.basename(elj_path)} carries no dmat"
        assert elj_results['dmat'].shape[0] == len(elj_results['good_ens']), (
            f"partner {os.path.basename(elj_path)}: dmat has {elj_results['dmat'].shape[0]} rows but good_ens "
            f"{len(elj_results['good_ens'])} -- the LJ density would be normalised over the wrong rows")
        elj_batch = elj_results['sample_batch']
        print(f"  LJ density row paired with {os.path.basename(elj_path)} "
              f"(step {(elj_results.get('provenance') or {}).get('step_ind')}, "
              f"{elj_batch.num_graphs} samples)")
    else:
        elj_results, elj_batch = results_dict, sample_batch
        print("  no elj_results_path: the figure carries ONE density row, labelled by this run's "
              "own energy function")
    (basin_colorscale, basin_min_batch, indexed_cluster_labels,
     n_basins, new_min_inds, p_maxima, packing_coeffs, polymorph_basin_index,
     polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
     sample_energy, sample_inds, stats) = combo_fig_analysis(
        ebatch, elj_batch, elj_results, num_polymorphs, sample_batch, results_dict,
        energy_function, cluster_labels, p_maxima, n_basins, indexed_cluster_labels,
        extra_refs=extra_refs, umap_kwargs=_as_dict(getattr(config, 'umap_kwargs', None)),
        kT=float(config.kT))
    if elj_path:
        del elj_results, elj_batch

    fig = combo_fig(num_polymorphs, n_basins, packing_coeffs, stats, thermos, sample_inds,
                    sample_embedding, p_maxima, polymorph_inds, new_min_inds, sample_energy,
                    polymorph_colorscale, sample_colors, indexed_cluster_labels,
                    basin_colorscale, basin_min_batch, polymorph_basin_index, energy_function,
                    poly_labels=poly_labels)
    if getattr(config, 'export_structures', True):
        export_structures(basin_min_batch, ebatch, config.results_dir,
                          config.run_name, poly_labels)

    # plotly.js from CDN, not inlined: the self-contained file is ~4.8 MB and
    # several viewers refuse to render it. This one is ~50 kB and stays
    # interactive; the PNG below is the offline-safe copy.
    fig_path = os.path.join(config.results_dir, f"{config.run_name}_combo.html")
    fig.write_html(fig_path, include_plotlyjs='cdn')
    print(f"wrote {fig_path} ({os.path.getsize(fig_path) // 1024} kB)")
    try:  # static copy for reading outside a browser; needs kaleido
        png_path = fig_path.replace('.html', '.png')
        fig.write_image(png_path, width=1600, height=1000, scale=2)
        print(f"wrote {png_path}")
    except Exception as e:
        print(f"png export skipped: {e}")
    if config.show_fig:
        # the default renderer needs nbformat, which this venv lacks; 'browser'
        # opens the html in the system browser and needs nothing
        fig.show(renderer='browser')
    return results_dict


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    raw = load_yaml(os.path.join(here, 'new_analysis.yaml'))
    base, runs = raw['base'], raw['runs']
    # comma-separated run names; unset runs every block in the yaml
    only = {r.strip() for r in os.environ.get('ONLY_RUN', '').split(',') if r.strip()}
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get('GPU_MEM_FRACTION', 0.9)), device=0)
    for run in runs:
        config = dict2namespace({**base, **run})
        if only and config.run_name not in only:
            continue
        print(f"\n=== {config.run_name} ===")
        run_analysis(config)
