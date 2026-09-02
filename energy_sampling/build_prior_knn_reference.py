r"""
Freeze a draw of prior samples into a reference file for the `latent_knn` energy.

    python build_prior_knn_reference.py --source dataset \
        --prior-path D:\crystal_datasets\...\prior.pt \
        --space-group 2 --max-z-prime 1 --periodic-centroids \
        --k 32 --out prior_knn_sg2.pt

The file it writes is the ENERGY. Everything about the target distribution that
is not a knob lives in these coordinates, so the two things this script exists to
get right are:

  GAUGE. Latents are extracted through latent_params(gauge_fix_free_axes=True) --
  the same call generator_energy scores. A reference built in any other gauge
  differs from the scored batch by a free-axis translation, which is invisible:
  every distance is simply wrong, no error is raised, and the resulting energy
  landscape is a plausible one belonging to no distribution in particular.

  GEOMETRY. The wrap mask and dead rows are properties of (space group,
  max_z_prime, periodic_centroids) and are reconstructed here from the same
  helpers the policy uses. They are STORED in the file, and PriorKNN.
  verify_against_policy re-checks them against the live model's ang_mask at
  startup -- so this reconstruction is guarded rather than trusted.

Sources:
  --source dataset   a prior_path file, i.e. {'equalized_prior': CrystalBatch}
  --source latents   a raw [N, D] tensor of ALREADY gauge-fixed latents

The prior MODEL's own marginal -- what the policy is actually sitting at when
phase 1 exits -- is a third source that needs a live modeller to sample from;
that route belongs in a protocol on_exit action beside snapshot_prior, not here.
A dataset-built reference is the prior's SUPPORT, not the prior model's density,
and the difference matters for a lambda=0 null test.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here),
           os.path.join(os.path.dirname(_here), 'mxtaltools')):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from energies.prior_knn import PriorKNN, reference_digest, LATENT_PERIOD  # noqa: E402
from models.aunit_periodicity import sg_periodic_centroid_axes  # noqa: E402
from models.dead_latent_rows import resolve_dead_rows  # noqa: E402


def build_wrap_mask(max_z_prime: int, periodic_centroid_axes=()) -> list:
    """The policy's ang_mask, reconstructed from the crystal layout.

    Mirrors GFN.get_periodic_dimensions: 6 box params, then max_z_prime centroid
    triples, then max_z_prime (theta, phi, r) triples of which phi and r wrap.
    Centroid axes wrap only where the aunit spans the full cell.
    """
    angs = [False] * 6
    angs.extend([False, False, False] * max_z_prime)      # centroids
    for _ in range(max_z_prime):
        angs.extend([False, True, True])                  # theta (no), phi, r
    for zp in range(max_z_prime):
        for axis in periodic_centroid_axes or ():
            if axis not in (0, 1, 2):
                raise ValueError(f"centroid axes must be in (0, 1, 2), got {axis}")
            angs[6 + 3 * zp + axis] = True
    return angs


def latents_from_dataset(prior_path: str, limit: int = 0) -> torch.Tensor:
    from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: F401

    blob = torch.load(prior_path, map_location='cpu', weights_only=False)
    if isinstance(blob, dict):
        for key in ('equalized_prior', 'prior', 'batch'):
            if key in blob:
                batch = blob[key]
                break
        else:
            raise ValueError(
                f"{prior_path} is a dict with keys {sorted(blob)[:8]} and none of them "
                f"is a stored batch (expected 'equalized_prior')")
    else:
        batch = blob

    if limit and getattr(batch, 'num_graphs', 0) > limit:
        idx = torch.randperm(batch.num_graphs)[:limit]
        batch = batch.subsample_new_batch(idx)

    # gauge_fix_free_axes=True: the crystal route. This MUST match the
    # `gauge_fix_free_axes=self.is_crystal` call in generator_energy.
    return batch.latent_params(gauge_fix_free_axes=True).detach().to(torch.float32)


def audit(latents: torch.Tensor, wrap_mask, dead_rows, k: int) -> None:
    """Report the draw, and refuse the ones that cannot support a density."""
    n, d = latents.shape
    print(f'\nreference draw: N={n}, D={d}')

    if n < 20 * k:
        raise ValueError(
            f"N={n} is too thin for k={k}: a kNN density wants at least ~20k points, "
            f"and a reference this small estimates its own sampling noise")

    outside = (latents.abs() > 1.0 + 1e-4).any(dim=1).sum().item()
    if outside:
        print(f'  WARNING: {outside} rows ({100 * outside / n:.2f}%) lie outside the '
              f'[-1, 1] box; the policy cannot reach them and they will only ever '
              f'inflate r_k for their neighbours')

    print(f'  {"dim":>4} {"wrap":>5} {"dead":>5} {"min":>8} {"max":>8} {"std":>8}')
    for i in range(d):
        col = latents[:, i]
        print(f'  {i:>4} {str(bool(wrap_mask[i])):>5} {str(i in dead_rows):>5} '
              f'{col.min():>8.3f} {col.max():>8.3f} {col.std():>8.3f}')

    # A dead row is a PINNED constant. If the data disagrees, the gauge used to
    # extract it is not the gauge the policy will emit, and every distance built
    # from this file is wrong. This is the one failure the digest cannot catch.
    for r in dead_rows:
        spread = float(latents[:, r].std())
        if spread > 1e-3:
            raise ValueError(
                f"row {r} is resolved DEAD (pinned, held out of the SDE) but varies by "
                f"std={spread:.4g} in this draw. The latents were not extracted in the "
                f"gauge the policy uses -- check gauge_fix_free_axes, the space group, "
                f"and max_z_prime.")


def informativeness(latents: torch.Tensor, wrap_mask, dead_rows, k: int) -> None:
    """Held-out check: does this density prefer prior samples to noise?

    Split the draw, build on one half, score the other half against a uniform
    draw over the same box. If prior samples do not score LOWER, the reference
    carries no information about where the prior lives and the energy built from
    it is decorative.
    """
    n = latents.shape[0]
    perm = torch.randperm(n)
    fit, held = latents[perm[:n // 2]], latents[perm[n // 2:]]

    knn = PriorKNN(fit, wrap_mask=wrap_mask, dead_rows=dead_rows, k=k)
    uniform = 2.0 * torch.rand(held.shape[0], held.shape[1]) - 1.0
    for r in dead_rows:                       # match the pinned gauge
        uniform[:, r] = fit[0, r]

    e_held = float(knn.energy(held).mean())
    e_uniform = float(knn.energy(uniform).mean())
    gap = e_uniform - e_held
    print(f'\nheld-out separation: prior {e_held:.3f} vs uniform {e_uniform:.3f} '
          f'nats  (gap {gap:+.3f})')

    if gap <= 0:
        raise ValueError(
            f"held-out prior samples score HIGHER than uniform noise (gap {gap:+.3f}). "
            f"This reference does not describe the prior; do not train against it.")
    if gap < 1.0:
        print('  WARNING: separation under 1 nat. Either the prior really is close to '
              'uniform in this box, or the gauge/wrap geometry is wrong. Check the '
              'per-dim spreads above before using this file.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=('dataset', 'latents'), required=True)
    ap.add_argument('--prior-path', help="prior_path file, for --source dataset")
    ap.add_argument('--latents', help="[N, D] gauge-fixed latent tensor, for --source latents")
    ap.add_argument('--out', required=True)
    ap.add_argument('--space-group', type=int, required=True)
    ap.add_argument('--max-z-prime', type=int, default=1)
    ap.add_argument('--periodic-centroids', action='store_true',
                    help="must match model.periodic_centroids in the training config")
    ap.add_argument('--hold-dead-rows', action='store_true', default=True,
                    help="must match model.hold_dead_latent_rows")
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--min-radius', type=float, default=1e-4)
    ap.add_argument('--limit', type=int, default=0, help="subsample the source to N rows")
    ap.add_argument('--note', default='', help="free text stored in provenance")
    args = ap.parse_args()

    if args.source == 'dataset':
        if not args.prior_path:
            ap.error('--source dataset requires --prior-path')
        latents = latents_from_dataset(args.prior_path, limit=args.limit)
        source = args.prior_path
    else:
        if not args.latents:
            ap.error('--source latents requires --latents')
        latents = torch.load(args.latents, map_location='cpu',
                             weights_only=False).detach().to(torch.float32)
        if args.limit and latents.shape[0] > args.limit:
            latents = latents[torch.randperm(latents.shape[0])[:args.limit]]
        source = args.latents

    sg = int(args.space_group)
    axes = sg_periodic_centroid_axes(sg) if args.periodic_centroids else ()
    wrap_mask = build_wrap_mask(args.max_z_prime, axes)
    dead_rows = (resolve_dead_rows(sg, is_crystal=True, max_z_prime=args.max_z_prime)
                 if args.hold_dead_rows else ())

    expected_dim = 6 + 6 * args.max_z_prime
    if latents.shape[1] != expected_dim:
        raise ValueError(
            f"source latents are {latents.shape[1]}-dimensional but max_z_prime="
            f"{args.max_z_prime} implies {expected_dim}")

    print(f'SG{sg}, max_z_prime={args.max_z_prime}, periodic_centroids='
          f'{bool(args.periodic_centroids)}')
    print(f'  wrapped dims: {[i for i, w in enumerate(wrap_mask) if w]}')
    print(f'  dead rows:    {list(dead_rows)}')

    audit(latents, wrap_mask, dead_rows, args.k)
    informativeness(latents, wrap_mask, dead_rows, args.k)

    blob = {
        'reference': latents,
        'wrap_mask': list(wrap_mask),
        'dead_rows': tuple(dead_rows),
        'k': int(args.k),
        'period': LATENT_PERIOD,
        'min_radius': float(args.min_radius),
        'provenance': {
            'source': source,
            'source_kind': args.source,
            'space_group': sg,
            'max_z_prime': int(args.max_z_prime),
            'periodic_centroids': bool(args.periodic_centroids),
            'hold_dead_latent_rows': bool(args.hold_dead_rows),
            'n_reference': int(latents.shape[0]),
            'note': args.note,
        },
        'sha256': reference_digest(latents),
    }
    torch.save(blob, args.out)

    check = PriorKNN.load(args.out)
    print(f'\nwrote {args.out}')
    print(f'  {check.describe()}')


if __name__ == '__main__':
    main()
