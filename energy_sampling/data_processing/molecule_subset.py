"""Nested molecule subsets of a conditional data set, by a fixed per-molecule draw.

molecule_u(identifier) is a number in [0, 1) fixed by the identifier alone (blake2b on a stream of its own), so the
subset `molecule_u < f` keeps the same molecules in every file, every build and every run, and the subsets of two
fractions are nested. It is the draw build_rc_prior.py --combine --molecule-fraction uses on the build side; here it is
applied at load time (inn_modeller.py --molecule_fraction), to the conditions file and the prior together.
"""
import hashlib

__all__ = ['molecule_u', 'keep_molecules']


def molecule_u(identifier):
    """The molecule's fixed draw in [0, 1)."""
    return int(hashlib.blake2b(str(identifier).encode(), digest_size=8, person=b'rc_molsub').hexdigest(),
               16) / 2.0 ** 64


def keep_molecules(batch, fraction):
    """(the rows of `batch` whose identifier's molecule_u is below `fraction`, the number of distinct molecules kept).
    A fraction of 1 returns `batch` itself. Raises when the fraction is outside (0, 1] or keeps no row."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"molecule fraction must lie in (0, 1], got {fraction}")
    ids = [str(m) for m in batch.identifier]
    if fraction >= 1.0:
        return batch, len(set(ids))
    keep = [i for i, m in enumerate(ids) if molecule_u(m) < fraction]
    if not keep:
        raise ValueError(f"molecule fraction {fraction} keeps no row of {len(ids)} ({len(set(ids))} molecules)")
    return batch.subsample_new_batch(keep), len({ids[i] for i in keep})
