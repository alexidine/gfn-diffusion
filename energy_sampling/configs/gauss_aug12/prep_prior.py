"""
gauss_aug12 -- build the per-space-group priors for the latent_gaussian battery.

The prior for this toy IS THE TARGET: latents drawn from N(c, T*w^2) on live rows,
pinned at the canonical 0.0 on dead rows. Backward training therefore starts on
distribution and the run's only remaining job is to make the forward policy agree
-- which is exactly what makes the analytic log Z a usable correctness gate rather
than a convergence study.

WHY THE MOLECULE DOES NOT MATTER. `latent_gaussian` registers False in
COMPUTES_REQUIRE_CLUSTER and is absent from COMPUTES_REQUIRE_UNIT_CELL, so
analyze() builds neither a supercell nor a unit cell. The reward reads only
latent_params(). So one real molecule replicated N times is enough, and it is
PREFERABLE: one identifier means one mol_id, one condition, one log Z, and no
per-condition bookkeeping between the measurement and the analytic number.

WHY NO ENERGY IS BAKED. init_prior_dataset re-analyzes unconditionally
(`if True:` at train.py:1607) from prior.latent_params(), so a baked value would be
overwritten anyway. Not baking it removes the chance of shipping a prior whose
stored energy disagrees with the live one.

    python configs/gauss_aug12/prep_prior.py            # all space groups
    python configs/gauss_aug12/prep_prior.py 14 19      # just these
"""
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for p in (r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion',
          r'C:\Users\mikem\Projects\mxt_gfn\mxtaltools'):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(HERE))

import spec  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

MOL_SOURCE = r'C:\Users\mikem\Projects\mxt_gfn\mxtaltools\mini_datasets\mini_new_csd.pt'
SEED = 20260812


def pick_molecule():
    """
    One well-defined Z'=1 molecule, smallest first. Size is irrelevant to the energy
    (nothing is built) but it is the whole memory cost of a 20000-row prior, so the
    smallest well-defined entry is strictly better.
    """
    data = torch.load(MOL_SOURCE, weights_only=False)
    cands = [e for e in data
             if int(e.z_prime) == 1 and bool(e.is_well_defined) and not bool(e.cocrystal)]
    if not cands:
        raise RuntimeError(f'no well-defined Z\'=1 molecule in {MOL_SOURCE}')
    cands.sort(key=lambda e: int(e.num_nodes))
    return cands[0]


def draw_latents(sg, n, generator):
    """
    N(MODE, T*w^2) on live rows, exactly 0.0 on dead rows.

    The dead rows are set, not sampled: 0.0 is what latent_params() reads back for
    them, so this makes the prior a FIXED POINT of the crystal round-trip. Sampling
    them instead would put every prior row at a value the build then discards, which
    is the original D33 defect re-introduced on the data side.
    """
    dead = spec.dead_rows(sg)
    std = spec.WIDTH * math.sqrt(spec.T)
    x = spec.MODE + std * torch.randn(n, spec.DIM, generator=generator)
    for r in dead:
        x[:, r] = 0.0
    return x


def build(sg, n=spec.N_PRIOR, seed=SEED):
    mol = pick_molecule()
    ident = mol.identifier
    ident = ident[0] if isinstance(ident, (list, tuple)) else ident
    print(f"  molecule {ident}  ({int(mol.num_nodes)} atoms)")

    g = torch.Generator().manual_seed(seed + int(sg))
    reps = []
    for _ in range(n):
        m = mol.clone()
        # PLAIN STRING, not the 1-element list mini_new_csd stores. collate_data_list
        # concatenates per-graph identifiers, so a list-valued one yields a list OF
        # LISTS and train.py's init_identifiers dies on `set.update` with
        # "unhashable type: 'list'" -- after the dead-row probe has already printed,
        # which makes it look like a D33 failure when it is a prep-side one. The
        # reference prior (mipcas) stores flat strings; match it.
        m.identifier = ident
        reps.append(m)
    batch = collate_data_list(reps)
    batch.reset_sg_info(int(sg))

    latents = draw_latents(sg, n, g)
    batch.latent_to_cell_params(latents.clone())

    # ---- the check that matters. If the round trip is not a fixed point then the
    # prior does not lie in the space the SDE can reach, and every backward step
    # trains toward a point the forward policy can never produce. On dead rows the
    # discrepancy is EXPECTED to be zero because we wrote the canonical value; on
    # live rows it must be zero because the transform is a bijection there.
    back = batch.latent_params()
    dead = spec.dead_rows(sg)
    live = [r for r in range(spec.DIM) if r not in dead]
    err_live = (back[:, live] - latents[:, live]).abs().max().item()
    err_dead = back[:, dead].abs().max().item() if dead else 0.0
    print(f"  round-trip: live |err| {err_live:.2e}   dead |value| {err_dead:.2e}")
    if err_live > 1e-4:
        raise RuntimeError(f'sg {sg}: latent round trip is not a fixed point on live rows '
                           f'({err_live:.3e}) -- the prior is unreachable, not merely noisy')
    if err_dead > 1e-6:
        raise RuntimeError(f'sg {sg}: dead rows read back as {err_dead:.3e}, not the '
                           f'canonical 0.0 the SDE pins them to -- fwd and bwd terminals '
                           f'would disagree on every step')

    # init_identifiers does `set().update(batch.identifier)`, so every element must be
    # hashable and there must be exactly one per graph. Assert it here rather than
    # discovering it several minutes into a launch.
    ids = batch.identifier
    if not isinstance(ids, list) or len(ids) != n or not all(isinstance(v, str) for v in ids):
        raise RuntimeError(
            f'sg {sg}: batch.identifier must be a flat list of {n} strings, got '
            f'{type(ids).__name__} of len {len(ids)} with element type '
            f'{type(ids[0]).__name__ if len(ids) else "?"} -- train.py\'s '
            f'init_identifiers will raise "unhashable type"')

    cl = batch.cell_lengths
    ca = batch.cell_angles * 180 / math.pi
    print(f"  cell lengths {cl.mean(0).tolist()}")
    print(f"  cell angles  {ca.mean(0).tolist()}")
    print(f"  packing_coeff mean {float(batch.packing_coeff.mean()):.3f}")

    # Drop the bulky per-sample analysis artifacts, same as data_processing's
    # collate_prior.py and generate_toy_prior.py already do. `fingerprint` alone is
    # 164 MB of a 186 MB file at 20000 rows, train.py excludes it from every buffer
    # (BULKY_ATTR_EXCLUDE_KEYS) and nothing reads it at runtime.
    for key in ('fingerprint', 'rdf'):
        if key in batch.keys():
            delattr(batch, key)

    # Both keys: prior_path reads 'equalized_prior', molecules_path reads 'prior' via
    # _load_condition_file. One file serves both, as mk_dev does.
    return {'prior': batch, 'equalized_prior': batch}


def main(argv):
    sgs = [int(a) for a in argv] or list(spec.SPACE_GROUPS)
    os.makedirs(spec.PRIOR_DIR, exist_ok=True)
    print(f"T={spec.T} width={spec.WIDTH} mode={spec.MODE} n={spec.N_PRIOR}\n")
    for sg in sgs:
        out = spec.prior_path(sg)
        print(f"SG{sg}  dead={spec.dead_rows(sg)}  -> {out}")
        payload = build(sg)
        torch.save(payload, out)
        print(f"  saved {os.path.getsize(out) / 1e6:.1f} MB\n")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
