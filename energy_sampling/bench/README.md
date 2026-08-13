# bench/

CPU sandbox for the control layer. Runs the **real** `LRController`,
`RayCalibration` and `Modeller.increment_batch_size` against synthetic loss
surfaces and a synthetic GPU, on a laptop, in seconds.

Documentation lives in [`../docs/module_bench.md`](../docs/module_bench.md) —
this file is a pointer, not a second copy.

```bash
python -m pytest bench/ -q            # regression suite
python -m bench.experiments           # the answer-producing runs
python -m bench.experiments probe_blindness
```

Findings produced so far: `F-011`–`F-017` in
[`../docs/findings.md`](../docs/findings.md).

`test_fidelity.py` builds the **real** `train.Modeller` on CPU and checks the
fake against it — surface, config values, and end-to-end LR equivalence. If it
fails, trust it over everything else here.

**The one rule:** the bench fakes the *modeller*, never the controller. If you
find yourself reimplementing a control law here, stop — that is how a bench rots
into reporting green about code that no longer exists.
