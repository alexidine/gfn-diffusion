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
THE FORM-B ARM (one extra arm, not an axis)
=============================================================================
`vg_detach_center` treats the VarGrad group centre as a constant inside the
huber'd term. Added to gflownet_losses.py 2026-08-14; default 0.0 reproduces
every previous run bitwise.

WHY IT IS NOT A NO-OP. With a quadratic loss the centre's contribution cancels
exactly -- the centred residuals sum to zero -- so detaching changes nothing.
The huber breaks that cancellation: the influence weights become
psi_beta(d_i) = clip(d_i, +-beta), clip is nonlinear, and sum_i psi_beta(d_i) is
only zero if the group's tails are symmetric. The leftover is

    -(1/K) sum_i psi_beta(d_i)   applied along the batch-mean score,

i.e. an MLE-on-buffer force whose weight is set by TAIL SKEW rather than by any
coefficient -- the same shape as the saturated-backward-TB -> MLE*beta collapse
in module_losses.md L8a. Confirmed in closed form to 1e-16 by
test_vg_detach_center.py, which also proves the difference vanishes without a
biting knee (so it is caused by the clip, not by the detach).

WHEN IT IS LIVE HERE. Identically inert on groups of 2, where d_2 = -d_1 and
clip is odd. condition_grouped_empirical_z pools EVERY same-condition row in the
batch, not just the `repeats` tile, so groups exceed 2 whenever a condition lands
more than twice -- which weighted_condition_sampling makes common. The arm is
therefore live on some batches and inert on others in the same run, and the
honest framing is "removes a skew-weighted absorption term that is present some
of the time", not "changes every step".

ONE ARM, NOT AN AXIS. Crossing it would double the battery. It is pinned to the
CONTROL cell (b005 bandwidth, sym loss-betas) so it is a clean one-change delta
against b005_sym, and set on BOTH fwd and bwd because var_conditioning runs
vg_lb on both legs and the mechanism is identical there -- one mechanism applied
consistently, not two axes.

NOT TESTED HERE, and worth stating: `bwd_loss_coeffs.beta` is read by the
var_conditioning VarGrad branch AND the naive TB branch, so the existing bwd80
arm moves the knee in two stages at once. That confound predates this arm and
this arm does not touch it.

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
import re
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


def build(base, tag, beta_name, beta, lb_name, fwd_b, bwd_b, width_name=None, width=None,
          vg_detach=0.0, extra_name=None):
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

    # Form B. Written from here rather than into base.yaml for the same reason as
    # the batch controls below -- the snapshot stays a faithful copy of the arm
    # that actually ran. 0.0 is the pre-2026-08-14 behaviour, so writing it on
    # every arm makes the control arms explicit rather than merely absent.
    vc = stage(cfg, "var_conditioning").get("loss_coeffs", {})
    if not (vc.get("fwd", {}).get("vg_lb", 0) > 0 and vc.get("bwd", {}).get("vg_lb", 0) > 0):
        raise SystemExit("base.yaml's var_conditioning stage does not run vg_lb on both legs -- "
                         "vg_detach_center would be inert, refusing to write the Form-B arm")
    for block in ("fwd_loss_coeffs", "bwd_loss_coeffs"):
        cfg[block]["vg_detach_center"] = float(vg_detach)

    if width is not None:
        for k in WIDTH_KEYS:
            if k not in cfg["model"]:
                raise SystemExit(f"base.yaml model block has no {k!r}")
            cfg["model"][k] = width

    # CLUSTER-IZE THE BATCH CONTROLS. base.yaml is snapshotted from qm9_anchor_aug13,
    # which ran LOCALLY, so it inherits mk_dev's dev-box settings: grow_batch_size false
    # and max_batch_size == batch_size. Both are hard stops, independently --
    # train.py::train only calls increment_batch_size under the flag, and the growth walk
    # returns immediately on `batch_size >= max_batch_size`. The first submission of this
    # battery therefore ran all six arms pinned at 1000 on an A100 against an analytic ELJ
    # energy, and every arm was cancelled by the scheduler for low utilization.
    #
    # Batch IS the occupancy lever on this route. The comment in increment_batch_size
    # retiring `gpu_util_floor` ("batch size does not move utilization") is a measurement
    # from the MLIP route, where the energy call already saturates the card; it does not
    # transfer to an MLP policy over an analytic energy. Setting these here rather than in
    # base.yaml keeps the snapshot a faithful copy of the arm that actually ran.
    for k, want in (("grow_batch_size", True), ("max_batch_size", 20000)):
        if k not in cfg:
            raise SystemExit(f"base.yaml has no {k!r} -- the batch controls moved, "
                             "refusing to write arms that may pin at the base batch")
        cfg[k] = want

    name = "_".join(x for x in (beta_name, lb_name, width_name, extra_name) if x)
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

    # the Form-B arm: one extra cell, pinned to the control (b005, sym) so it is
    # a single-change delta against b005_sym. Not crossed -- see the docstring.
    if not args.width:
        arms.append(build(base, args.tag, BETAS[0][0], BETAS[0][1],
                          LOSS_BETAS[0][0], LOSS_BETAS[0][1], LOSS_BETAS[0][2],
                          vg_detach=1.0, extra_name="formb"))

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

    # an arm that silently duplicates another is a wasted GPU-week. vg_detach_center
    # is in the key because the Form-B arm matches the b005_sym control on every
    # other axis -- leave it out and this assert correctly rejects the battery.
    keys = [(stage(c, "var_conditioning")["lr_sensor"]["beta"],
             c["fwd_loss_coeffs"]["beta"], c["bwd_loss_coeffs"]["beta"],
             c["model"]["policy_hidden_dim"],
             c["bwd_loss_coeffs"]["vg_detach_center"]) for _, c in written]
    assert len(set(keys)) == len(written), "arms are not distinct"

    with (HERE / "INDEX.tsv").open("w", encoding="utf-8") as f:
        f.write("name\thyper_beta\tfwd_beta\tbwd_beta\thidden_dim\tvg_detach_center\trun_name\n")
        for (name, cfg), k in zip(written, keys):
            f.write(f"{name}\t{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{k[4]}\t{cfg['run_name']}\n")

    # KEEP THE ARRAY RANGE IN SYNC WITH INDEX.tsv. submit.sbatch selects an arm by
    # INDEX row, so a range that is too SHORT drops the tail arms silently -- no
    # error, no missing-config message, they simply never run. (Too long is safe:
    # the script's empty-ARM branch exits 1.) The 2026-08-14 Form-B arm made the
    # battery 7 against a hardcoded 0-5, which is exactly this failure, so the
    # range is written from here rather than maintained by hand.
    sb = HERE / "submit.sbatch"
    text = sb.read_text(encoding="utf-8")
    want = f"#SBATCH --array=0-{len(written) - 1}"
    new, n = re.subn(r"#SBATCH --array=0-\d+", want, text)
    if n != 1:
        raise SystemExit(f"submit.sbatch: expected exactly one '#SBATCH --array=0-N' line, found {n}")
    if new != text:
        sb.write_text(new, encoding="utf-8")
        print(f"submit.sbatch: array range -> 0-{len(written) - 1}")

    print(f"wrote {len(written)} arms + INDEX.tsv to {HERE}")
    for (name, _), k in zip(written, keys):
        print(f"  {name:20s} hyper beta {k[0]:<5g} fwd/bwd {k[1]:g}/{k[2]:g}  "
              f"hidden_dim {k[3]}  vg_detach {k[4]:g}")
    print(f"\ndata tag: {args.tag} -- run --preflight on the cluster before submitting")


if __name__ == "__main__":
    main()
