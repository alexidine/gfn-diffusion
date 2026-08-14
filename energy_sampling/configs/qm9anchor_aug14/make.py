"""
qm9anchor_aug14 -- config generator for the QM9 anchor battery.

    python configs/qm9anchor_aug14/make.py               # write arm configs + INDEX
    python configs/qm9anchor_aug14/make.py --preflight   # check every data file exists
    python configs/qm9anchor_aug14/make.py --width       # add the hidden-dim axis

base.yaml is a SNAPSHOT of qm9_anchor_aug13/qm9a98.yaml (2026-08-14), the closest arm that
actually ran on QM9 anchors. Re-snapshot by hand, deliberately, so this battery does not
drift under later edits to that set.

=============================================================================
THE BANDWIDTH AXIS
=============================================================================
qm9a98 already runs the hypergradient sensor on var_conditioning at beta 0.05. `hyper` is
the coherent sensor there: `ray` is only meaningful in a fused stage training replay TB
(protocol.py::_parse_lr_sensor), because its probe draws from replay and scores with
replay_loss_coeffs -- anywhere else it rates a loss nobody is optimising. `hyper` reads no
loss at all, so it is coherent whatever the stage trains.

What is NOT settled is the bandwidth. The parser refuses to default `beta` on exactly these
grounds: swept on the bench across 12 cells the per-cell optimum spanned 20x. The arms
bracket the supported range --

    b005   beta 0.05   the qm9a98 setting. CONTROL, so the battery says whether moving
                       the bandwidth does anything at all before it says which way.
    b010   beta 0.10
    b020   beta 0.20   the bench optimum: b=0.2 won 6 of 7 cells in the single-player
                       suite and took ZERO divergences in the 7-cell hazard battery.

0.4 is deliberately absent: it trips the divergence wire on the hazard suite, so the
safety boundary sits between 0.2 and 0.4 and there is no reason to spend an arm past it.

CAVEAT ON THE BENCH NUMBERS. Those are bench-surface results, not this problem. The bench
was rebuilt 2026-08-13 with no oracle and Adam by default, and the optimizer changes every
LR answer -- best rate moves 10x between sgd and adam. Treat b=0.2 as "the value with the
best evidence anywhere", not as a prediction about var_conditioning on QM9 anchors.

=============================================================================
THE LOSS-BETA AXIS
=============================================================================
Untested hypothesis: a large backward beta against a small forward one is better than the
symmetric 10/10 every QM9 arm has run so far.

    sym     fwd 10 / bwd 10   CONTROL, the qm9a98 setting
    bwd80   fwd 10 / bwd 80

qm9a98d already carries bwd beta 80, but it also moves condition_block_m to 2, restarts
from a checkpoint, halves seed_lr and drops divergence_loss_abs by 1000x -- four other
changes, so whatever it shows cannot be attributed to the beta. That is the reason to
spend arms on a clean axis rather than reading a98d.

replay_loss_coeffs.beta is deliberately LEFT at 10 in every arm. The hypothesis named the
forward and backward legs; moving replay too would make the axis three-legged and no arm
would isolate anything.

=============================================================================
THE WIDTH AXIS (--width, off by default)
=============================================================================
Crossing bandwidth x width doubles the arm count, and the two are not independent: a wider
policy changes the gradient geometry the hypergradient cosine is measuring. Run it as a
second battery on the winning beta rather than crossed, unless the arm budget is free.
`hidden_dim` here moves the whole model-width family together (cond/policy/s/t/flow and
s_emb), since moving one alone tests plumbing rather than capacity.

=============================================================================
DATA -- NOT YET BUILT
=============================================================================
Arms point at the anchor-derived triple from build_anchor_conditions.py:

    <tag>_conditions.pt        molecules_path       one entry per TRAINING molecule
    <tag>_prior.pt             prior_path           anchors of training molecules only
    <tag>_test_conditions.pt   test_molecules_path  held-out SMILES, forward-eval only

Build it after the chunks land, with --holdout-n set (the split is at SMILES level, and
the prior is built from training molecules only, so neither leaks). Then --preflight.
"""
import argparse
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DATA_ROOT = "/scratch/mk8347/data/crystal_datasets/conditional/priors"

BETAS = [("b005", 0.05), ("b010", 0.10), ("b020", 0.20)]
LOSS_BETAS = [("sym", 10.0, 10.0), ("bwd80", 10.0, 80.0)]   # (name, fwd, bwd)
WIDTHS = [("w512", 512), ("w1024", 1024)]
WIDTH_KEYS = ("cond_hidden_dim", "policy_hidden_dim", "s_hidden_dim",
              "t_hidden_dim", "flow_hidden_dim", "s_emb_dim")


def stage(cfg, name):
    for s in cfg["protocol"]["stages"]:
        if s.get("name") == name:
            return s
    raise SystemExit(f"base.yaml has no stage named {name!r}")


def build(base, tag, beta_name, beta, lb_name, fwd_b, bwd_b, width_name=None, width=None):
    cfg = copy.deepcopy(base)

    sensor = stage(cfg, "var_conditioning").get("lr_sensor")
    if sensor is None or sensor.get("kind") != "hyper":
        raise SystemExit("base.yaml's var_conditioning stage is not on the hyper sensor -- "
                         "the battery's whole axis is missing, refusing to write arms")
    sensor["beta"] = beta

    for block, val in (("fwd_loss_coeffs", fwd_b), ("bwd_loss_coeffs", bwd_b)):
        if "beta" not in cfg.get(block, {}):
            raise SystemExit(f"base.yaml has no {block}.beta")
        cfg[block]["beta"] = val

    if width is not None:
        for k in WIDTH_KEYS:
            if k not in cfg["model"]:
                raise SystemExit(f"base.yaml model block has no {k!r}")
            cfg["model"][k] = width

    name = "_".join(x for x in (beta_name, lb_name, width_name) if x)
    cfg["run_name"] = f"qm9anchor_aug14_{name}"
    cfg["molecules_path"] = f"{DATA_ROOT}/{tag}_conditions.pt"
    cfg["prior_path"] = f"{DATA_ROOT}/{tag}_prior.pt"
    cfg["test_molecules_path"] = f"{DATA_ROOT}/{tag}_test_conditions.pt"
    return name, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="qm9c100k",
                   help="build_anchor_conditions --tag the triple was written under")
    p.add_argument("--width", action="store_true", help="cross in the hidden-dim axis")
    p.add_argument("--preflight", action="store_true",
                   help="check referenced data files exist (run on the cluster)")
    args = p.parse_args()

    base = yaml.safe_load((HERE / "base.yaml").read_text(encoding="utf-8"))

    arms = []
    for bn, b in BETAS:
        for lbn, fb, bb in LOSS_BETAS:
            if args.width:
                for wn, w in WIDTHS:
                    arms.append(build(base, args.tag, bn, b, lbn, fb, bb, wn, w))
            else:
                arms.append(build(base, args.tag, bn, b, lbn, fb, bb))

    if args.preflight:
        missing = sorted({v for _, c in arms
                          for v in (c["molecules_path"], c["prior_path"],
                                    c["test_molecules_path"])
                          if not Path(v).exists()})
        for m in missing:
            print(f"MISSING {m}")
        raise SystemExit(1 if missing else "preflight clean")

    written = []
    for name, cfg in arms:
        path = HERE / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        written.append((name, cfg))

    # an arm that silently duplicates another is a wasted GPU-week
    keys = [(stage(c, "var_conditioning")["lr_sensor"]["beta"],
             c["fwd_loss_coeffs"]["beta"], c["bwd_loss_coeffs"]["beta"],
             c["model"]["policy_hidden_dim"]) for _, c in written]
    assert len(set(keys)) == len(written), "arms are not distinct"

    with (HERE / "INDEX.tsv").open("w", encoding="utf-8") as f:
        f.write("name\thyper_beta\tfwd_beta\tbwd_beta\thidden_dim\trun_name\n")
        for (name, cfg), k in zip(written, keys):
            f.write(f"{name}\t{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{cfg['run_name']}\n")

    print(f"wrote {len(written)} arms + INDEX.tsv to {HERE}")
    for (name, _), k in zip(written, keys):
        print(f"  {name:16s} hyper beta {k[0]:<5g} fwd/bwd {k[1]:g}/{k[2]:g}  "
              f"hidden_dim {k[3]}")
    print(f"\ndata tag: {args.tag} -- run --preflight on the cluster before submitting")


if __name__ == "__main__":
    main()
