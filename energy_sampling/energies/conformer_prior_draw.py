"""One molecule's prior draw, in its OWN chart: states, raw energies and the draw's stats.

WHY A SHARED FUNCTION. Two paths draw a per-molecule prior. The run draws at startup
(``ConformerModeller._draw_prior_states``, called per carrier member by
``_draw_carrier_prior``); the offline builders draw into a ``prior_path`` file. They
diverged once already: the builder's ``draw_prior_states`` went through
``build_prior_states.draw_states``, which emits one column per ROTATABLE AXIS, so at
`full` (3N-6 columns) it died in ``bake_energies`` with a 2-against-39 shape error, and
its uniform fallback put every bond length uniformly over +/- delta_r_max -- a prior with
no content. This is the level-aware draw the run already uses, lifted out so a file and a
run cannot mean different things by "the prior".

WHAT IT MIRRORS, LINE FOR LINE. ``ConformerTorsions.sample_prior_states`` (per-DoF
marginals, sibling torsions and ring systems drawn jointly), then the optional state-space
Rprop relax through ``prior_baselines.descend`` at the member's own temperature, chunked
and under ``enable_grad`` exactly as the modeller does it. The modeller's copy is pinned
equal to this one by tests/conformer/test_build_conformer_set.py, so a change to either
fails there rather than as two priors that quietly disagree.

The energies are RAW POTENTIAL at T = 1 (``bake_energies``), the currency
``prebuilt_sample_to_reward`` expects on a stored row -- it divides by the sampling
temperature itself, and adds the change of measure back on the read side.
"""
from __future__ import annotations

from typing import Optional

import torch


def draw_member_prior(member, n: int, rng, relax_steps: int = 0, prior=None,
                      report: bool = False, chunk: int = 4096):
    """``(states [n, k_member], energies [n], stats)`` for one molecule.

    ``member`` is a ``ConformerTorsions`` (a carrier member, never the dispatcher: a draw
    always happens in one molecule's own chart). ``prior`` is the fitted InternalPrior and
    is REQUIRED -- ``sample_prior_states`` reads its histograms and ring banks, and there
    is deliberately no uniform fallback: at `flex` and `full` a uniform box draw is not a
    weaker prior, it is not a prior (see the module docstring). ``rng`` is a
    ``numpy.random.Generator``; the caller owns its seeding, so a per-molecule seed stays
    a per-molecule seed.
    """
    if prior is None:
        raise ValueError(
            f'{member.smiles}: draw_member_prior needs the fitted InternalPrior '
            f'(energy_config.internal_prior_path). There is no uniform fallback at '
            f'level {member.level!r}: a box draw over bond lengths and angles is not a '
            f'prior')
    n = int(n)
    states, stats = member.sample_prior_states(prior, n, rng, report=report)
    x = torch.as_tensor(states, dtype=member.dtype, device=member.device)
    steps = int(relax_steps or 0)
    stats = dict(stats)
    stats['relax_steps'] = steps
    if steps > 0:
        from energies.prior_baselines import descend

        out = []
        # enable_grad EXPLICITLY, as the modeller does: a caller under no_grad (the churn
        # path is decorated with it) would otherwise leave `descend` nothing to step
        with torch.enable_grad():
            for i in range(0, x.shape[0], chunk):
                best_x, _ = descend(member, x[i:i + chunk], steps)
                out.append(best_x.detach())
        x = torch.cat(out, 0)

    from energies.conformer_data import bake_energies
    with torch.no_grad():
        e = bake_energies(member, x)
    stats['energy_median'] = float(e.median())
    stats['energy_p90'] = float(torch.quantile(e, 0.9)) if n > 1 else float(e.max())
    return x.detach(), e.detach(), stats


def member_prior_seed(identifier: str, seed: Optional[int] = 0) -> int:
    """A per-CONDITION RNG seed from the identifier's hash, offset by ``seed``.

    SEEDED PER CONDITION, NOT PER CALL. The old builder re-seeded every molecule from the
    same ``default_rng(seed)``, so every molecule drew from an IDENTICAL uniform stream --
    correlated priors across the set, invisible in any one molecule's statistics. Hashing
    the identifier also makes a condition's draw independent of which other molecules share
    the file, so growing a set along the run ladder does not redraw the ones already in it.
    """
    import hashlib

    h = hashlib.blake2b(identifier.encode(), digest_size=8, person=b'cond_prior').digest()
    return (int.from_bytes(h, 'big') + int(seed or 0)) % (2 ** 63)
