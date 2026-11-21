# pool8_smooth_nov21

Conditional MLE on the pool8 P-1 eLJ set (switched K=4 prior), from fresh policies, 12 h per arm. It tests three
smoothing levers: dropping atomwise cross-attention, a fresh chart kick of the buffer on every draw, and noise on the
condition vectors. `make.py` writes every arm; its docstring is the arm definition.

Arm settings. Every arm has the same prior, batch (1,400 rows, no growth) and step cap (60,000). The contrasts to read
are `x_` against `v_` (cross-attention), `_fixed` against `_kick` (fresh kicks), and `v_kick_bn` against the
`v_kick_noise` arms (noise, with the 32-wide bottleneck present in both).

| arm | condition route to the flow | buffer draw | kick radius (Å) | condition noise sigma |
|---|---|---|---|---|
| `x_fixed` | atoms + pooled vectors | stored children | none | none |
| `x_kick` | atoms + pooled vectors | fresh kick | 0.01–0.42 | none |
| `v_fixed` | pooled vectors only | stored children | none | none |
| `v_kick` | pooled vectors only | fresh kick | 0.01–0.42 | none |
| `v_kick_bn` | pooled vectors, bottleneck | fresh kick | 0.01–0.42 | 0 |
| `v_noise05` | pooled vectors, bottleneck | stored children | none | 0.5 |
| `v_kick_noise05` | pooled vectors, bottleneck | fresh kick | 0.01–0.42 | 0.5 |
| `v_kick_noise10` | pooled vectors, bottleneck | fresh kick | 0.01–0.42 | 1.0 |
| `v_wide_noise05` | pooled vectors, bottleneck | fresh kick | 0.10–0.63 | 0.5 |

## Reading it

- Compare arms at matched steps on `eval_bwd/mle_heldout_rows`, `eval_test/mle`, `eval_bwd/mle_anchors`, their gaps,
  `eval_fwd/jensen_z`, `eval_test/jensen_z` and `Excess Energy Nats Per DoF P50`. All arms share one prior, so they
  share floors.
- A fresh-kick arm never trains on the stored children, so `eval_bwd/mle` is on rows it has not seen. A `_fixed` arm
  trains on those rows. The stored-row number is therefore not a like-for-like memorisation measure between the two;
  the held-out rows and held-out molecules are.
- Condition noise acts on the backward (MLE) loss only. Forward draws and evaluations use the clean condition.
- `live_kernel/out_of_box_first_try` and `live_kernel/kept_centre` report the chart kernel's wall behaviour.

## Launch

From `energy_sampling/` on the cluster:

```
sbatch configs/pool8_smooth_nov21/submit_pool8_smooth_nov21.sbatch
```
