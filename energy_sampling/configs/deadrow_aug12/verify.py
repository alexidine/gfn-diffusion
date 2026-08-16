"""
deadrow_aug12 -- verify the dead-row change ENGAGED, and that the controls did not move.

    python configs/deadrow_aug12/verify.py [run_name_substring]

Reads the local wandb datastore directly (no network), same as
configs/local_aug09/verify.py.

Every check is a tell chosen because the failure it catches is SILENT in ordinary
metrics. A wrong index set here does not crash; it trains to a plausible but wrong
log Z. So the two structural CONTROLS carry as much weight as the primary arm:

  A-PARITY       a_sg2_on vs a_sg2_off must agree. Triclinic has n_dead = 0, so the
                 knob has nothing to do and any SYSTEMATIC divergence means it acts
                 where it must not. Read as agreement to GPU nondeterminism, not
                 bitwise -- bitwise is the CPU unit test's job
                 (test_dead_latent_rows.py::test_prechange_bitwise_identity).
                 A drifting-apart trend is the signal; a constant small offset is not.

  DEAD ROWS      b_sg14_* startup must print the resolved rows and the probe line
  ANNOUNCED      confirming latent_to_cell_params ignores them. Absence means the
                 resolver never ran -- most likely the run resumed a checkpoint whose
                 gfn_config predates the change, which _assert_dead_rows_match now
                 refuses, but only on the load path.

  B-EFFECT       b_sg14_on vs b_sg14_off must DIFFER. If they agree, the knob is inert
                 on a space group where 2 of 12 rows are dead, i.e. the fix did not
                 engage. This is the mirror of A-PARITY and the pair is only
                 meaningful together: one must move, one must not.

  STEP_VAR       terminal_var / step_var now average over LIVE dims. Between the two
  SCALE          b_sg14 arms they are directly comparable; a ~10/12 ratio between them
                 would mean the live-dim restriction did not apply and the metric is
                 diluted by structurally-frozen dims.

  TOY GATE       d_toy_on must print the non-crystal line and NEVER a dead-row list.
                 sg 1 is a placeholder there; a dead-row list means the is_crystal gate
                 was bypassed, which would freeze real toy dims once free axes land.

  NO SILENT      Every arm: 'Nonthermal Threshold' should equal
  RESCALE        nonthermal_entropy_per_dim * live_dim, not * data_ndim. Only visible
                 on non-triclinic arms.

Numbers are reported, not judged, wherever this battery is too short to have an
expectation -- nothing here converges, and a verdict on convergence would be false
confidence.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ES = os.path.dirname(os.path.dirname(HERE))
for _root in (os.path.dirname(ES),
              os.path.join(os.path.dirname(os.path.dirname(ES)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TAG = 'deadrow0812'
EXPECTED = {
    'a_sg2_on':    dict(sg=2,  ndim=12, dead=(),       control=True),
    'a_sg2_off':   dict(sg=2,  ndim=12, dead=(),       control=True),
    'b_sg14_on':   dict(sg=14, ndim=12, dead=(3, 5),   control=False),
    'b_sg14_off':  dict(sg=14, ndim=12, dead=(),       control=False),
    'c_sg9_zp2_on': dict(sg=9, ndim=18, dead=(3, 5),   control=False),
    'd_toy_on':    dict(sg=1,  ndim=12, dead=(),       control=True),
}


def find_runs(substr=''):
    """Locate local wandb run dirs for this battery."""
    roots = [os.path.join(ES, 'wandb'), os.path.join(os.path.dirname(ES), 'wandb')]
    hits = []
    for r in roots:
        hits += glob.glob(os.path.join(r, 'run-*'))
        hits += glob.glob(os.path.join(r, 'offline-run-*'))
    out = []
    for h in sorted(set(hits)):
        name = os.path.basename(h)
        if substr and substr not in name:
            continue
        out.append(h)
    return out


def scan_logs(arm):
    """
    Grep the arm's stdout capture for the startup announcements. The dead-row
    resolution and probe print ONCE at init and are not metrics, so the log is the
    only place they exist -- which is exactly why they are checked here and not
    inferred from wandb history.
    """
    pats = {
        'resolved': re.compile(r"SG(\d+) \((\w+)\).*dead latent rows \[([^\]]*)\]"),
        'no_dead': re.compile(r"SG(\d+) \((\w+)\).*no dead latent rows"),
        'non_crystal': re.compile(r"non-crystal energy: no dead latent rows"),
        'probe': re.compile(r"dead-row probe: .*rows \(([^)]*)\) are ignored for SG(\d+)"),
        'threshold': re.compile(r"Nonthermal Threshold"),
    }
    found = {k: [] for k in pats}
    for path in glob.glob(os.path.join(HERE, f'{arm}*.log')) + \
                glob.glob(os.path.join(ES, f'*{arm}*.log')):
        try:
            txt = open(path, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        for k, p in pats.items():
            found[k] += p.findall(txt)
    return found


def main():
    substr = sys.argv[1] if len(sys.argv) > 1 else TAG
    print(f"deadrow_aug12 verification  (filter: {substr!r})")
    print()

    cfg_dir = HERE
    arms = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(cfg_dir, '*.yaml')))
    print("ARMS GENERATED")
    for a in arms:
        e = EXPECTED.get(a, {})
        live = e.get('ndim', 0) - len(e.get('dead', ()))
        role = 'CONTROL (must not move)' if e.get('control') else 'must show the effect'
        print(f"  {a:<15} sg {e.get('sg','?'):<3} dead {str(e.get('dead','?')):<8} "
              f"live {live}/{e.get('ndim','?'):<3} {role}")
    print()

    print("STARTUP TELLS  (from captured stdout -- see launch.txt for tee'ing)")
    any_log = False
    for a in arms:
        f = scan_logs(a)
        e = EXPECTED.get(a, {})
        if not any(f.values()):
            print(f"  {a:<15} no log found -- rerun with the tee in launch.txt")
            continue
        any_log = True
        if a == 'd_toy_on':
            ok = bool(f['non_crystal']) and not f['resolved']
            print(f"  {a:<15} non-crystal line: {bool(f['non_crystal'])}  "
                  f"dead-row list present: {bool(f['resolved'])}  -> {'OK' if ok else 'FAIL'}")
        elif e.get('dead'):
            probe = f['probe'][0][0] if f['probe'] else None
            print(f"  {a:<15} resolved: {f['resolved'][:1]}  probe rows: {probe}  "
                  f"-> {'OK' if f['resolved'] and probe else 'CHECK'}")
        else:
            print(f"  {a:<15} no-dead line: {bool(f['no_dead'])}  "
                  f"(knob off or triclinic; a dead-row list here would be wrong: "
                  f"{bool(f['resolved'])})")
    if not any_log:
        print("  -- no logs at all. The startup tells are the cheapest checks in this")
        print("     battery and they only exist in stdout; capture it (launch.txt).")
    print()

    print("WANDB RUNS FOUND")
    runs = find_runs(substr)
    if not runs:
        print(f"  none matching {substr!r}. If the arms ran online, pull with wa.py")
        print("  (reference_local_wandb_reading) and compare these series per arm:")
    else:
        for r in runs:
            print(f"  {os.path.basename(r)}")
    print()

    print("SERIES TO COMPARE  (this script locates runs; read the pairs deliberately)")
    print("  A-PARITY   a_sg2_on vs a_sg2_off: fwd/tb_err, bwd/tb_err, log_Z, step_var.")
    print("             Expect agreement with NO widening trend. A widening gap is the")
    print("             failure; a constant tiny offset is GPU nondeterminism.")
    print("  B-EFFECT   b_sg14_on vs b_sg14_off: the same four. These MUST differ --")
    print("             agreement means the fix did not engage. Compare at matched step.")
    print("  F-009      b_sg14 pair, fwd/bwd gap and tb_err floor. This is the only")
    print("             measurement that settles the tb_err-floor CONJECTURE. Neither arm")
    print("             converges here, so treat it as directional, not final.")
    print("  Z'=2       c_sg9_zp2_on: does it train at all, and is zp_ordering_energy")
    print("             finite and non-degenerate? That term indexes raw_latents by")
    print("             absolute position and is the reason states stayed full width.")
    print("  THRESHOLD  'Nonthermal Threshold' == nonthermal_entropy_per_dim * live_dim")
    print("             (10 for b_sg14_on, 12 for its off partner, 16 for c_sg9_zp2_on).")
    print()
    # Both gaps are now closed by configs/gauss_aug12/, which scores an ANALYTIC
    # latent target and so builds no cell -- meaning it can synthesise a prior for
    # any space group, which a physical energy cannot.
    print("NOT COVERED HERE (see configs/gauss_aug12): orthorhombic (no physical")
    print("prior on disk), free aunit axes (implemented at Z'=1, but a physical")
    print("prior must be a real crystal, so no arm here reaches them),")
    print("hexagonal (asserts by design), and convergence of anything. See make.py.")


if __name__ == '__main__':
    main()
