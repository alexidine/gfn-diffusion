"""
Is the initialiser BIASED away from the known forms, or is 0.9-1.1 just what any
point looks like in 18 dimensions?

These demand opposite responses. Bias -> reshape the starting distribution, big win.
Dimensionality -> the forms are ordinary members of the sampled space, "zero mass
within 0.35" is what random sampling always looks like, and there is nothing to fix.

THE NULL: for a random init proposal, how far is its NEAREST OTHER init? That is the
distance scale the init distribution achieves against a point drawn from itself. If
the known forms sit at that same scale, they are as reachable as anything else and
the miss is pure dimensionality. If they sit in the far tail, they are anomalously
isolated and the distribution really is missing that region.

Init-to-init is the right null, NOT output-to-init: every optimised output descends
from an init, so its nearest init includes its own progenitor and it would look
artificially close.
"""
import os, torch
ROOT = os.path.join('D:', os.sep, 'crystal_datasets', 'acridine')
init = torch.load(os.path.join(ROOT, 'nikos_comparison', 'init_latents_sg14_zp2.pt'),
                  weights_only=False, map_location='cpu').float()
print(f"init proposals: {tuple(init.shape)}")

g = torch.Generator().manual_seed(0)
sub = init[torch.randperm(len(init), generator=g)[:4000]]
D = torch.cdist(sub, sub)
D.fill_diagonal_(float('inf'))
nn = D.min(dim=1).values
q = torch.tensor([0., .01, .05, .25, .5, .75, 1.])
print(f"\nNULL -- init to its NEAREST OTHER init (n={len(sub):,}):")
print(f"   min/1%/5%/25%/median/75%/max: "
      f"{[round(float(v), 4) for v in torch.quantile(nn, q)]}")

# the known forms, measured the same way against the SAME subsample
from mxtaltools.dataset_utils.utils import collate_data_list
poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                  weights_only=False, map_location='cpu').cpu()
pl, pids = poly.batch_to_list(), list(poly.identifier)
print(f"\nKNOWN FORMS -- distance to nearest init, same {len(sub):,} subsample:")
for n in ('ACRDIN07', 'ACRDIN06'):
    b = collate_data_list([pl[pids.index(n)].clone()],
                          exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    v = b.latent_params()[0].float()
    d = (sub - v[None, :]).norm(dim=-1)
    pct = float((nn < float(d.min())).float().mean()) * 100
    print(f"   {n:10s} nearest {float(d.min()):.4f}   median {float(d.median()):.4f}"
          f"   -> further than {pct:.1f}% of inits are from their nearest neighbour")

# WHICH coordinates? if a form sits outside the init range in specific dims, that is
# the actionable part -- a distribution to widen, not a vague "explore more".
print(f"\nper-coordinate: is the form inside the init distribution's range?")
lo, hi = init.min(0).values, init.max(0).values
q01, q99 = torch.quantile(init, 0.01, dim=0), torch.quantile(init, 0.99, dim=0)
for n in ('ACRDIN07', 'ACRDIN06'):
    b = collate_data_list([pl[pids.index(n)].clone()],
                          exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    v = b.latent_params()[0].float()
    out_full = ((v < lo) | (v > hi)).nonzero().flatten().tolist()
    out_99 = ((v < q01) | (v > q99)).nonzero().flatten().tolist()
    print(f"   {n:10s} outside FULL init range in dims {out_full or 'none'}; "
          f"outside 1-99% in dims {out_99 or 'none'}")
    for k in (out_99 or [])[:6]:
        print(f"        dim {k:2d}: form {float(v[k]):8.3f}  "
              f"init 1%/median/99% "
              f"{float(q01[k]):7.3f} {float(init[:, k].median()):7.3f} {float(q99[k]):7.3f}")
