"""The GPU guard's scope: refuse CONTESTED launches, allow a sole tenant.

The guard exists to stop two training runs sharing one card (three BSODs from two runs,
2026-08-11/12). It is deliberately not a solo-fit checker: a sole tenant that does not fit
OOMs alone, which the batch controller already recovers from.

This pins both halves, because a change that only proves the allow-path would look
identical to having deleted the guard.
"""
import warnings

warnings.filterwarnings("ignore")

from gpu_guard import GPUBusy, require_free_gpu

CONFIG = "configs/qm9_anchor_aug13/qm9a98b.yaml"


def check(label, **kwargs):
    try:
        require_free_gpu(config_path=CONFIG, **kwargs)
        return True, ""
    except GPUBusy as e:
        line = next((l.strip() for l in str(e).splitlines()
                     if "BLOCK" in l or "incoherent" in l), "")
        return False, line


def main():
    ok_solo, why_solo = check("solo")
    print(f"sole tenant                    : {'PASS' if ok_solo else 'BLOCK'}  {why_solo[:90]}")

    # The property that must survive: an incoherent co-tenancy claim is still refused.
    # Each config here declares cuda_memory_fraction 0.9, so N>1 cannot fit by arithmetic.
    ok2, why2 = check("cotenants=2", cotenants=2)
    print(f"declared co-tenancy (2 runs)   : {'PASS' if ok2 else 'BLOCK'}  {why2[:90]}")
    ok3, why3 = check("cotenants=3", cotenants=3)
    print(f"declared co-tenancy (3 runs)   : {'PASS' if ok3 else 'BLOCK'}  {why3[:90]}")

    print()
    if ok_solo and not ok2 and not ok3:
        print("PASS: solo allowed, contested still refused -- guard scope intact")
    elif ok2 or ok3:
        print("FAIL: a co-tenancy claim was ALLOWED -- the guard's whole purpose is gone")
    else:
        print("FAIL: sole tenant refused -- the change did not take effect")


if __name__ == "__main__":
    main()
