> Design note from the hub's active-learning workflow, 2026-11-10 (12 + 3 agents: survey, three designs, refuters, synthesis, critic, revision, re-check). Analysis only; nothing run on crystals. Toy scripts, drafts, critic and re-check are archived in `D:/crystal_datasets/paper_review_nov02/al_design_nov10/` (paths in the text such as `revise/...` are relative to it). Quiver F2 is the entry point.

# F2 phase 2 as active learning: one ledger, no level in the loss, one path (final)

**Labels.** **[D]** derived here or on a cited wiki/design page (analysis, nothing run). **[L]** literature (§12). **[T]** observed in a 1-D CPU toy, never crystals; observations, not results. **[U]** untested.

**Symbols.** c = (molecule, space group g); s ∈ B = [−1,1]^d is the FD chart (d ≤ 12 for norbornane at Z′=1); p_g the per-space-group prior; E_h the expensive fidelity (the label), E_l the cheap one; J = exp(`log_reference`); β = 1/kT; k_H the Huber knee. Path: log R_λ = (1−λ)·log p_g + λ·(−βE_h + log J), π_λ = R_λ/Z_λ(c). Δ = −βE_h + log J − log p_g. w = log R_λ − log q(·|c).

## 0. Topline

**Status.** Not ready: build items B1–B3 and prechecks P0–P5 (§§9–10) precede the short A1 run.

**Protocol.**
- **Warm start.** q ≡ p_g by reward-free MLE on p_g draws; λ_g = 0, where Z = 1.
- **Rounds.** M conditions i.i.d. uniform. Per visit: N candidates in fixed per-component counts from q, p_g, the uniform box U, kicks around c's ledger rows and transplant kicks from other molecules in g, all with closed-form densities; all scored at E_l; exactly b labelled at E_h (a fresh q pair, a tagged audit stratum, a pivotal score stratum); every label appended, in physical energy, to a never-evicted ledger.
- **Learner, no level anywhere.** Term 1: within-condition pairwise log-ratio regression over ledger rows, weighted by tempered target weights. Term 2: fresh-pair reverse KL (RLOO, k = 2).
- **Readout, allocation, anneal.** One deterministic-mixture estimate of Z_λ(c), never in a loss. One λ_g per space group, stepped per round under fresh-pair, ledger-ESS and term-1 gates. Every loop ends on a count, budget or round bound (§8).

**Why this is the fusion.** One label serves three consumers.
1. **The learner needs no level.** Term 1's zero set is q ∝ π_λ on the ledger's support for any acquisition law μ; term 2 is unbiased for ∇KL(q‖π_λ) [D; L 2210.00580, 2010.10436]. Term 1's weights are positive, so they keep that zero set; they stop unfittable rows from setting a hidden level (§2).
2. **The estimator needs exact weights and has them**: every density and inclusion probability is recorded, so readouts are consistent whatever q's state [D; L Owen–Zhou 2000, Elvira 2019].
3. **Allocation, the anneal and term 1 reuse those weights**; stored legs make retargeting λ free [D].

**Axes.** Multi-source: a stratified defensive mixture [L Hesterberg 1995]. Multi-fidelity: two-phase sampling, cheap energy only in selection [L Neyman 1938]. Multi-target: amortisation, one target per space group [D].

**Not claimed.** A basin with no label and no q mass is invisible to every term. The floors find p_g basins of mass ≳ 0.017 at 400 labels per condition and ≳ 0.8 at pool8 scale (§8); beyond that, discovery rests on transplants and transfer [U]. Every wait is bounded; discovery is not guaranteed. Term 1 is shown only in 1-D toys.

## 1. Domain, target, floor

- **Domain.** FD rows are circles of period 2 (hemisphere azimuth, translation-glued centroids) or intervals on [−1,1] (`energies/fd_coordinates.py`; `models/nf_policy.py` resolves `bounded_rows: 'auto'` to `'linear'`). Every proposal lives on B, where the bounding leg Σ relu(|x|−1)² (latent-dimension-structure.md) is 0 [D].
- **Clip settings belong to the target.** `reward_range` (mk_dev.yaml:1255) is documented as invalid under λ mixing, so `reward_range: null`; `physical_energy_clip` is valid under mixing but changes the target, so its value (default null) is recorded with it. Records stay physical; clash Δ is unbounded on the path [D].
- **p_g (build item B1).** `_Flow` accepts `bounded_mask` (uniform interval base, spline mapping [−1,1] to itself), which gives ∫_B p_g = 1 and Z_0 = 1. But `PriorFlow.fit` and `load` build `_Flow` without it, `is_bounded` is a non-persistent buffer and `save` does not write the mask, so a fitted or loaded p_g treats interval rows as lines with Gaussian tails and ∫_B p_g < 1 [D, code read]. B1 threads the mask through `fit`, `save`, `load` and `verify_against_policy`, testing ∫_B p_g = 1 as the mean of 2^d·p_g over U draws. **P0:** PriorFlow's chart is the policy's FD chart (`verify_against_policy` checks only wrap mask, dead rows and T).
- **Uniform floor.** U = 2^{−d}, α_u ≥ α_min, so R_λ/Q̄ ≤ 2^d·sup R_λ/α_u however poorly p_g fits [L Hesterberg 1995]. The bound is astronomically loose: finite variance only, and only if E_h is bounded below on B, which P1 (a pull-through screen) cannot certify. U does not discover (its basin mass is vol/2^d).

## 2. Learner

**Term 1: weighted ledger log-ratio regression.**
- **Draw.** A condition uniformly among those with ≥ 2 finite labels, then two of its finite rows i.i.d. ∝ ω_i. Failure rows (R = 0) never enter.
- **Loss.** ℓ₁ = H_{k_H}(w_i − w_j); the level cancels inside each pair.
- **Weights.** ω_i = (y_i/π_i)^{a_c}, y_i = R_λ/Q̄_c the §7 weight, detached, recomputed at readout cadence. a_c is the largest a ∈ [a_min, 1] with weighted Kish ESS ≥ n_ω, by fixed-iteration bisection; a_min = 0.05 [U].

**Fixed point [D].** The quadratic expected loss is Var_ω(w): stored-row TB at the in-graph level E_ω[w] (the pooled-VarGrad `mean` centre, over different rows). It is zero iff q = k_c·π_λ on supp μ_c, k_c free; positive weights do not move this zero set, so acquisition does not tilt the target. With a finite ledger this is pointwise; whether ratios become basin masses in d ≈ 12 is [U].

**Why weights: the hidden level.** Clash rows (Δ of hundreds of nats, from U, p_g and kick tails, and q's junk) cannot be fit by a flow with density bounded below. Then the zero set is empty, they drag the in-graph level down, and every fittable row gets a uniform upward push: MLE onto μ, the pinned-level tilt again [D; T, Table 1]. At a = 0.1 a row 250 nats below the wells weighs ≈ e^{−25}, so clash rows leave term 1 but stay in the ledger and estimator; the weight is the energy's own, not a row filter. Fallback: a redescending (Cauchy) ψ, which also repairs the toy but zeroes the pull toward a newly labelled deep basin.

*Table 1 [T] (`revise/levelfree_toy.py`, `revise/recheck/clash_toy.py`, `clash_fix_toy.py`).* 1-D, wells of mass .50/.30/.20; a fixed 400-row ledger skewed to well A or C, a share f in a clash region 250 nats below the wells; 3 ledgers per skew; Adam from uniform q; T2 is the exact gradient (an oracle). Tilt = |change in log(q_A/q_C) between skews| (0: acquisition does not move the fit); TV to π off the clash region. Free bins have a 10⁻⁴ density floor; row 1 has no clash rows or floor.

| terms | clash share f | tilt, free bins (nats) | TV, free bins | tilt, Fourier order 6 (nats) | TV, Fourier 6 |
|---|---|---|---|---|---|
| TB at pinned level e = −5, + T2 (draft 1) | 0 | 3.91 | 0.38 | 3.57 | 0.33 |
| T2 only | 0 and 0.25 | 0.00 | 0.00 | 0.00 | 0.02 |
| term 1 only | 0 | 1.42 | 0.23 | 0.33 | 0.10 |
| term 1 + T2, unweighted | 0 | 0.02 | 0.00 | 0.02 | 0.04 |
| term 1 + T2, unweighted | 0.25 | 2.84 | 0.29 | 0.71 | 0.57 |
| term 1 + T2, weighted a = 0.1 | 0.25 | 0.01 | 0.00 | 0.02 | 0.03 |
| term 1, Cauchy ψ, + T2 | 0.25 | 0.14 | 0.01 | 0.00 | 0.03 |

*With an exact T2, T2 alone is tilt-free and term 1 adds nothing; clash rows make unweighted term 1 harmful; tempered weights remove the harm.*

*Table 2 [T] (`revise/final/stoch_toy.py`, `stoch_toy_fourier.py`).* Term 1 under the stochastic T2 (RLOO, k = 2): q starts collapsed on A; 60-row initial ledger skewed to A or C; each 20-step round labels and appends one fresh pair; SGD 3,000 steps; 8 seeds per skew; weights use the exact Q̄.

| terms | f | q masses A/B/C, Fourier 6, C-skewed ledger | tilt, Fourier 6 (nats) | TV, Fourier 6 | TV, free bins |
|---|---|---|---|---|---|
| T2 only | 0 | 1.00 / 0.00 / 0.00 | 0.00 | 0.55 | 0.58 |
| term 1 + T2 | 0 | 0.49 / 0.30 / 0.21 | 0.09 | 0.08 | 0.40 |
| term 1 + T2 | 0.25 | 0.44 / 0.34 / 0.22 | 0.65 | 0.35 | 0.70 |
| term 1 weighted a = 0.1 + T2 | 0.25 | 0.49 / 0.30 / 0.20 | 0.05 | 0.06 | 0.44 |

*The stochastic T2 alone never leaves a collapsed q: it samples only where q has mass. Term 1 over a ledger that keeps appending fresh pairs restores the labelled basins, and weighting keeps that true at 25% clash rows. In free bins only labelled bins recover (TV 0.40 even at 12,000 steps), a coverage artefact; a flow could likewise meet point ratios with spikes (quiver E2). An unlabelled basin stays at 0 (`revise/recover_toy.py`).*

**Why this is not the pooled-VarGrad collapse, and the gate.** The zero sets are the same family (the 11-08 union degeneracy) and term 1's quadratic form is the `mean` centre; the differences are dynamics only [D]:
1. Pooled VarGrad started at λ = 1 with an 85–95-nat own-versus-stored gap; here λ_g starts at 0, where w ≡ 0. Each λ step bounds the change of w at fixed q (dλ·sd Δ ≈ √(2ε)), not the gap between rows of different origin.
2. Its groups were 2 own rows re-drawn from q's latest support plus 2 stored rows (4 of 6 pairs mixed), so the push followed q. Here 33–55% of rows are q-born (2 of 8, plus α_q ∈ [0.1, 0.4] of the rest), 0.45–0.5 of pairs are mixed, and q-born rows never re-enter on-policy.
3. Its forward branch trained only the Z sidecar; term 2 trains the policy.

So term 1 is gated (B3): per space group, log the clash share, the weighted share of pairs past the knee (knee share) and the weighted mean signed gap between q-born and other rows (own gap). The λ hold (§6) requires knee share ≤ 0.25 and |own gap| ≤ k_H. Preregistered kill rule for the short A1 run: knee share > 0.5 for 10 consecutive rounds stops it; fallback, the Cauchy ψ at a_c = 1 [U thresholds].

**Term 2: fresh-pair reverse KL.**
- ĝ = −½[(w₁−w₂)ρ₁∇log q(x₁) + (w₂−w₁)ρ₂∇log q(x₂)], ρ_i = q_θ(x_i)/q_birth(x_i): RLOO with k = 2 [L Kool 2019], on-policy VarGrad [L 2010.10436; 2210.00580].
- Unbiased at every reuse step before the ρ cap (E_{q_birth}[ρ∇log q_θ] = 0) [D]; cap firing logged; reuse ≤ G steps or until ESS(ρ) < ρ_ess. Pairs with a failure row are dropped and counted. No Huber clip (it biases); gradient-norm clipping. Terms weighted γ:1, γ fixed.
- q = π_λ zeroes both terms. π restricted to a basin union is a local optimum of term 2 only for disjoint supports [D].

## 3. Proposals: stratified, exact densities

Fixed counts n_j = α_j·N per component (no count noise, fixed cost).

| component | density | v1 |
|---|---|---|
| q(·\|c) at λ_g | exact (flow) | yes, α_q ≥ α_min |
| p_g | exact on B after B1 | yes, α_g |
| U on B | 2^{−d} | yes, α_u |
| kicks around c's ledger rows | (1/P)Σ_y k(x\|y) | yes, α_k |
| transplant kicks, other molecules' records in g | the same kernel | yes, α_x |
| AR-tail resample (LS-GFN); cheap relaxations | chain-dependent; none | v2 (AR order unchecked); after F3, term-1 rows only |

*Caption: proposal components. Each round's P kick parents are snapshotted and stored, because Q̄ evaluates every round's kernel at every labelled point.*

**Kernel** k(x|y) = Π_circle WN(x_d; y_d, σ_d, period 2) · Π_interval TN(x_d; y_d, σ_d, [−1,1]).
- **Interval rows**, including the hemisphere's polar edge and K-cut (non-translation seams): truncated, normaliser Φ((1−y_d)/σ_d) − Φ((−1−y_d)/σ_d).
- **Circle rows:** sample and wrap; images |m| ≤ 2 give relative error < 6×10⁻⁹ when σ_d ≤ 0.5, which a config check enforces [D].
- **Transplants.** FD lengths are normalised by the length box hi = 2Z(r + 3 Å) (`fd_coordinates.length_box`), so s is comparable across molecules of one space group sharing a chart layout (beyond this rigid Z′=1 bed d varies). Whether transplants hit basins is [U]; A6 ablates them.

## 4. Selection

**Per visit**, b = 2 + b_a + b_s labels exactly: the fresh pair (inclusion 1); an audit stratum of b_a by simple random sampling, tagged and score-independent; a score stratum of b_s from the remaining N − b_a by the pivotal method [L Deville & Tillé 1998], π_i = min{1, τ·b_s/(N−b_a) + (1−τ)·η·a_i}, with η from a fixed-iteration bisection plus a uniform top-up so Σπ_i = b_s. Horvitz–Thompson weights are 1 (audit) and 1/π_i (score) [L Horvitz & Thompson 1952].

**Score.** a_i = ν_i·u_i, u_i the within-condition rank of δ_l = (1−λ_g)·log p_g + λ_g·(−β[Ē_l + κ_g(E_l − Ē_l)] + log J) − log q. Shrinkage acts only on the cheap energy, so at κ = 0 the exact flow and Jacobian legs still rank. A rank, because exp-weights at a 30-nat spread are one-hot [T].

**Trust.** κ_g = max(0, ρ̄_g)·n/(n+n₀); ρ̄_g is the within-condition Spearman correlation of E_l against E_h over finite score-independent rows (audit plus fresh pair) within k_band·kT of the condition's ledger minimum, pooled over g. Near 0 early, so ranking degrades to flow legs and novelty; energies, not δ, since the shared −log q inflates agreement [D].

**Novelty.** ν = (1 + n_c(x; h) − 1[x's parent within h])^{−1/2}, within condition. If the logged fraction of candidates with n_c > 0 is below 5% at A2's end, ν is dropped [D: at d ≈ 12 and ≤ 400 labels, n_c is mostly 0].

## 5. Fidelity

- **E_h is the only label**: it alone enters the target, records and estimators; E_l enters only u and κ [D].
- A surrogate in the reward invites proxy exploitation [L CbAS 2019; COMs 2021]; GFlowNet-AL [L 2203.04115] uses one; we do not.
- **Chained fidelities** (eLJ → exp-6 → MLIP) are two-phase sampling: inclusion probabilities multiply, estimators stay consistent [L Neyman 1938; Peherstorfer 2016]; delayed acceptance is the MCMC analogue [L Christen & Fox 2005; 2202.03876].
- A cheap energy that misses a motif costs discovery rate, never target bias; F3 measures how often.
- A residual model m (E_h ≈ E_l + m) is deferred; it may enter the target only if B̂_g = ½β²·Var_π̂(ΔE − m_oof) ≤ 0.1 nats [D].
- Coulomb pull-through in exp-6 (P1) is answered by changing the fidelity or its clamp, never by filtering rows.

## 6. Annealing from the space-group prior

- **One λ_g per space group**: held-out molecules get a defined, shared target [D].
- **Step, once per round** per visited g: dλ_g = clip(√(2ε)/s_g, dλ_min, dλ_max), s_g² = max(V_q, V_π̂) [lambda-annealing.md; L Del Moral, Doucet & Jasra 2006; Zhou, Johansen & Aston 2016]. V_q = mean over the round's fresh pairs in g of (Δ₁−Δ₂)²/2, unbiased for Var_q(Δ); V_π̂ = the self-normalised deterministic-mixture variance of Δ over g's finite ledger rows. A q that drops a labelled basin underestimates V_q; V_π̂ still sees it [D].
- **Hold.** Advance only while: the median fresh-pair |w₁−w₂|/√2 ≤ s_max; g's ledger Kish ESS at the proposed λ, pooled over its conditions, ≥ n_ess (CESS [L Zhou et al. 2016, attribution from recall]); and the term-1 gate passes (§2). Holds count toward retirement only once g's ledger has ≥ n_min finite rows (at most ⌈n_min/b⌉ visits).
- **Retire.** After H_max consecutive counted holds g stops advancing and is flagged; it is re-tested every R_re rounds and resumes when the holds clear.
- **Reading.** Arms are ranked at λ = 1, with λ_g reached as a primary covariate. Preregistered: an arm ending with λ_g < 1 in a group is not adopted there, and the anneal is fixed next.
- **Retention [D, U].** Term 1 holds labelled basins in the toys (Table 2); in d ≈ 12, [U]. An unlabelled basin whose energy is mis-predicted by ΔE loses target mass once λ ≳ 1/(βΔE); only labels retain it [D].

## 7. The estimator (readout, allocation, term-1 weights)

For condition c, (t, j) indexes rounds × components with counts n_{t,j} and densities Q_{t,j}; the fresh pair is its own stratum (n = 2, Q = q_t, π = 1); N = Σ n_{t,j}.

  Q̄_c(x) = N⁻¹ Σ_{t,j} n_{t,j}·Q_{t,j}(x),  y_i = R_λ(x_i)/Q̄_c(x_i),  Ẑ_c(λ) = N⁻¹ Σ_{i labelled} y_i/π_i.

- **Form.** Balance-heuristic deterministic mixture [L Veach & Guibas 1995; Owen & Zhou 2000; Elvira 2019], i.e. MBAR with known normalisers [L Kong 2003; Tan 2004; Shirts & Chodera 2008]; any λ without a call.
- **Unbiased → consistent.** q_t, kick parents and λ depend on earlier labels, so recycling every round's density makes Ẑ consistent, not unbiased [L Cornuet 2012 AMIS; Marin, Pudlo & Sedki 2019]. The unbiased per-round estimator stays as a diagnostic. Q̄ holds every round's floors, so y_i ≤ 2^d·R_λ/ᾱ_u.
- **Variance (two-phase) [L Neyman 1938; D].** V̂₂ = N⁻²Σ(1−π_i)·y_i²/π_i² (Hájek/Poisson approximation); V̂₁ = N⁻²Σ n_{t,j}·σ̂²_{t,j}, the inclusion-weighted within-stratum variance; a stratum with < 2 labels borrows its component type's pooled σ̂², ignoring q_t's drift. se_log = √(V̂₁+V̂₂)/Ẑ is a plug-in heuristic: no CLT is claimed (a recycled weight depends on later proposals); it is less anticonservative than thinning variance alone, not calibrated under heavy tails.
- **Reliability.** Pareto k̂ of {y_i/π_i} [L Vehtari PSIS]; k̂ > 0.7 is flagged and excluded from allocation.
- **Failures.** A failed or timed-out call is a row with R = 0: it counts in Ẑ (y = 0) and is excluded from term 1, V_π̂ and κ (safe: term 1 depends only on support). The failure fraction is logged.
- **Uses.** KL(q‖π_λ) = log Ẑ − Ĵ_F; basin masses; term-1 weights; the v2 bandit on ΔJ_F; the common reference. Q̄ needs log q_t at labelled rows: keep each round's checkpoint, fill lazily.

## 8. Termination and discovery

| wait | bound |
|---|---|
| outer loop | R_max rounds or label budget Λ |
| labels per visit | exactly b (fixed-size design) |
| η and a_c bisections; pivotal pass | fixed iterations; ≤ N pivotal steps |
| expensive call | timeout T_call, then a failure row (R = 0) |
| condition with < 2 finite labels | no term-1 rows; term 2 runs; no level, no deadlock |
| hold counting | starts within ⌈n_min/b⌉ visits to g |
| λ_g | each round advances ≥ dλ_min or holds; 1 or retired within ⌈1/dλ_min⌉·(H_max+1) counted rounds; un-retirement bounded by R_max |
| fresh-pair reuse | ≤ G steps; earlier if ESS(ρ) < ρ_ess |
| learner | G gradient steps per round |
| kicks, transplants | fixed counts; parents snapshotted |
| κ, novelty | defined at n = 0 (κ = 0); one update per round |
| kill rule; A/A re-read | 10 rounds; one re-read, then "undecidable" |

*Caption: each wait and what ends it if its signal never arrives. Draft 2's fault was the reverse: irreversible retirement on a small ledger ended a wait too early.*

**Discovery bound [D].** Under uniform selection a p_g-born candidate is labelled with probability ≈ b′/N (b′ = b_a + b_s), so expected labelled hits in a basin of p_g mass m_b over V visits are V·α_g·b′·m_b: m_b* = 1/(V·α_g·b′) for one expected hit (P ≈ 0.63), three times that for P ≈ 0.95.

| budget (labels per condition) | visits V at b = 8 | m_b*, P = 0.63 | m_b*, P = 0.95 |
|---|---|---|---|
| A1, 400 | 50 | 0.017 | 0.050 |
| A4 (depth), 800 | 100 | 0.0083 | 0.025 |
| A2 (acquisition), 400; floor π ≥ (b_a + τ·b_s)/N | 50 | 0.033 | 0.10 |
| pool8, ~10 | ~1 | ~0.8 | none |

*Caption: smallest p_g basin mass the floors discover, assuming α_g = 0.2 and b′ = 6 (A2: b_a = 2, b_s = 4, τ = 0.25); excludes q's own mass, kicks and transplants.*

## 9. Cost and build

**Per round:** M·b expensive labels; M·N cheap energies; ≈3·M·N flow passes plus kernel sums; G gradient steps; throughput unmeasured [U]. One visit to each of 72,464 molecules at b = 8 is 5.8×10⁵ labels per space group, so per-condition spend must stay O(10) and transfer must do the rest [D].

**Reuse:** `get_tb_loss`; the `_Flow` bounded base; `lambda_mix`, `prior_flow_path`; `ConditionLogZTracker`; `sg<g>/` prefixes; FD identity tooling; `MolCrystalData.analyze`. Not `condition_is`/`probe_phase2` KL: it uses each arm's own log Ẑ and silently drops non-finite log w.

**Build:**
- **B1:** the bounded `PriorFlow` and its normalisation test (§1).
- **B2:** the append-only ledger (x, cid, g, component, round, stratum, π_i, E_h, E_l, log J, log p_g, birth log Q_{t,j}, round parents, lazy log q_t, failure rows), term-1 weights and the weighted pairwise draw. If `RowStore` ever hosts it (v2): `is_exponent: 0` (`draw_tb` defaults to 1.0), no admission bar or eviction, `log_r` composed live from the legs (lifting the `prior_flow_path`/`anneal_coeffs` refusals in `row_store_is_well_formed`).
- **B3:** the term-1 gate readouts and the kill rule.
- The sampler, selector, κ, novelty, estimator and term 2; per-space-group PriorFlow and λ_g state; the round driver in `train.py`, run short and production-shaped before tests (AGENTS rule 5).

## 10. First experiment: norbornane, eLJ → exp-6

**Bed.** 117 molecules × {P-1, P2_1/c} = 234 conditions on the existing held-out split (about 44 held-out conditions; recount). p_g per space group, molecule labels dropped, fit on training molecules' search rows only (Q1), so held-out readouts are not circular through p_g.

**Prechecks.** P0 chart identity (§1). P1 minimum exp-6 energy over 10⁴ p_g and U draws. P2 exp-6 wired into the trainer. P3 native-reference basins the eLJ minima miss. P4 per-condition sd and tail of log w on C's checkpoint and the short A1 run (sets n_d). P5 clash share, knee share and own gap in the short A1 run.

**Reference.** An exp-6-native search on 20 held-out conditions: 100 random starts from U each, relaxed under exp-6 without eLJ seeding, deduplicated by the FD identity cut; about 2×10⁵ calls. In d ≈ 12 that is a lower bound: the lowest reference basin may not be global. Recall counts the union of these and the re-relaxed eLJ minima, tagged by origin. **One common log Z per condition**: the §7 estimator pooled over every arm's ledgers and readout draws, with k̂.

**Metrics.**
- **J_F(c)** = mean finite log w over n_d fresh q draws at λ = 1, with the failure fraction; paired ΔJ_F(c) = KL_C(c) − KL_A1(c) needs no Z. It is mode-seeking: collapse onto a basin of π-mass m costs only −log m (0.51 nats at m = 0.6, about δ*), so it cannot decide alone.
- **Low-band recall** (coverage): the fraction of reference basins within k_band·kT of the condition's lowest that relaxing 32 q draws under exp-6 finds (calls counted; a draw joins a basin only within the FD identity cut, else "new").
- **Secondary:** KL(π‖q) (k̂-flagged); calls to first label the lowest reference basin; λ_g reached; κ_g; each source's share of new basins.

**Arms.** Matched at 400 exp-6 training labels per training condition. The training-molecule search behind p_g and C's MLE rows is counted once, charged to every arm; eLJ and exp-6 calls are reported separately (cost ratio ≈1).

| arm | content; λ schedule | differs from A1 in | seeds |
|---|---|---|---|
| C | current recipe: conditional MLE on search rows, then TB with learned head (tbhead), fresh q draws labelled; λ = 1 throughout phase 2 | warm start, learner, level, proposals, λ | 3 |
| A1 | core (§§1–8); κ = 0, ν = 1; λ_g by §6 | — | 3 |
| A2 | A1 + score a_i | selection score | 2 |
| A3 | A1, q-only proposals | proposal mixture | 2 |
| A4 | A1 on half the training conditions at 800 labels each | allocation | 2 |
| A5 | A1, term 2 off (Q4) | learner term 2 | 2 |
| A6 | A1, α_x = 0 | transplants | 2 |

*Caption: arms and seeds; A2–A6 anneal by §6; A1's seeds give the A/A pair. A2 asks whether the cheap rank carries information, not whether multi-fidelity pays at MLIP cost ratios.*

**Why the rules changed [T].** Draft 2's rule (paired mean ≥ δ*, ≥ 80% of conditions positive; 256 draws, 44 conditions, 3 seeds) adopts a true 0.5-nat gain with P = 0.49 at a 3-nat within-condition spread and 0.006 at 30 nats (today's own samples: 28–34), and its A/A gate passes with P = 0.46 at 30 nats (`revise/recheck/power_toy.py`, 5,000 replicates, best case: no training variance).

**Decision rules (preregistered).**
- **Draws.** n_d from P4 so the paired arm-mean se of ΔJ_F is ≤ δ*/4: n_d ≥ 2s²/((δ*/4)²·C_h·S), with s the per-draw sd including the tail, C_h held-out conditions, S seeds; s = 30 nats, C_h = 44, S = 3 gives n_d ≈ 900. If between-seed variance alone exceeds (δ*/4)², add seeds instead.
- **A/A width check.** The A1 seed pair's held-out ΔJ_F interval must have half-width ≤ δ*/2; if not, re-read the checkpoints once at 4·n_d (no retraining), then report the bed undecidable.
- **Adopt A1 over C** iff neither metric is inferior (one-sided 95% lower bound of A1 − C above −δ*/2 for J_F, above −r*/2 for recall) and at least one is superior (lower bound > 0 and point estimate ≥ δ*/2, or ≥ r*/2). δ* = 0.5 nats, r* = 0.1 (Q2). Intervals use paired per-condition differences, combining between-seed variance (t, 4 df) with evaluation variance.
- **Screens** (A2–A6): read only if the interval excludes 0. Common random numbers: identical condition schedules, eval seeds, held-out set.

**Order.** B1; P0–P2; the short A1 run (P4, P5, kill rule); A1 × 3 with the width check; the rest. The owner gives the go before the battery.

**Cost [U]** (exp-6 calls; wall time unmeasured): training 16 × 7.6×10⁴ ≈ 1.2×10⁶; readouts ≈ 2.2×10⁵ per run at n_d ≈ 900, ≈ 3.5×10⁶ total (≈3× training; P4 moves it); reference 2×10⁵; screening ≈ 6×10⁵ eLJ per run; the search once.

## 11. Owner questions

1. **Practice-bed p_g:** training molecules' search rows only (recommended; not circular for held-out readouts; fast), or a search-free cheap prior first (fully non-circular; slower; construction unspecified)?
2. **Preregistered effects:** δ* = 0.5 nats ΔJ_F and r* = 0.1 low-band recall (recommended), or δ* = 1.0 and r* = 0.2 (fewer false adoptions, smaller n_d; may miss a real gain)?
3. **λ granularity:** one λ per space group (recommended; held-out targets defined), or per condition fed to the conditioner (finer; history-dependent targets)?
4. **A5's ablation:** term 2 off (recommended; exact T2 does most of the work in Table 1, stochastic T2 alone cannot recover in Table 2, so its crystal-scale value is open), or term 1 at a pinned lower-bound level (re-tests level-freedom at scale)?

## 12. Literature

**Identifier and authors confirmed by web search (draft-2 pass):** Malkin et al., arXiv:2210.00580; Richter et al., arXiv:2010.10436; Kool, van Hoof & Welling 2019; Kirsch et al., arXiv:2106.12059 (no closed-form inclusion probabilities); Peherstorfer et al. 2016, MFIS; Elvira et al. 2019 (arXiv:1511.03095); Cornuet et al., AMIS, arXiv:0907.1254; Marin, Pudlo & Sedki 2019 (arXiv:1211.2548); Shirts & Chodera 2008, MBAR; Zhou, Johansen & Aston 2016 (arXiv:1303.3123); Gunter et al., WSABI, arXiv:1411.0439; Podryabinkin & Shapeev, arXiv:1611.09346; Smith et al., arXiv:1801.09319; Vandermause et al., FLARE, arXiv:1904.02042; Klein & Noé, arXiv:2406.14426; Jain et al., arXiv:2203.04115.

**From recall:** Horvitz & Thompson 1952; Deville & Tillé 1998; Neyman 1938; Christen & Fox 2005; Veach & Guibas 1995; Kong et al. 2003; Tan 2004; PSIS (arXiv:1507.02646); Del Moral, Doucet & Jasra 2006; Osborne et al. 2012; CbAS (arXiv:1901.10060); COMs (arXiv:2107.06882). **From draft 1's sweeps:** Owen & Zhou 2000; Hesterberg 1995; 2202.03876; 2403.03208; 2410.01432; 2208.01893; 2306.11715. **Not adopted:** Bayesian quadrature (a GP per condition in d ≈ 12, no transfer). **Unverified, not relied on:** 2006.01541; 2504.11713; Douc et al. 2007; FastCSP's "200 GPU-h"; Adaptive-teachers' "C = 19".

## Unresolved / disputed

- **Re-check new defect 6, "point estimate ≥ δ*":** rejected; a threshold equal to the effect caps power at 50%, the fault the same item names. Used: point ≥ δ*/2 with se ≤ δ*/4 (power ≈ 0.98 at δ*, evaluation noise only).
- **Re-check new defect 1, a_c set by an ESS floor alone:** amended; the floor can drive a to 0 and restore the clash tilt, so a_c ≥ a_min.
- **Re-check new defect 3, fallback = the defect-1 weighting:** superseded, since the weighting is now default; the fallback is the Cauchy ψ at a_c = 1.
- **Hub fact "PriorFlow accepts bounded_mask":** true only of `_Flow`; `PriorFlow.fit`, `save` and `load` drop it (working-tree read). Carried as B1, unfixed.
- **Free-bin residual in Table 2:** read as a coverage artefact; unconfirmed until a d = 2–4 flow toy.
- **Still unchecked:** the ≈44 held-out count; the CESS attribution; real clash share (P5); log w sd and tail (P4).

## Changes from draft 1

**Draft 1 → draft 2 (critic).** Majors 1–9: level out of the loss; uniform floor on the correct domain; exact stratified kick densities; two-phase variance; transplants with m_b*; native reference, training-only p_g; common log Z, paired J_F; A/A pair, CRN, 3×3 seeds, δ*; arm C. Minors 10–20: recall membership, call counts; fixed-size pivotal selection, tagged audit; timeouts, reuse ESS; low-band κ; parent-relative novelty; λ step on max(Var_q, Var_π̂); overclaims, P0; citations; `is_exponent: 0`.

**Draft 2 → final (re-check).**
- New 1: tempered deterministic-mixture weights on term-1 rows; clash and knee shares logged; `reward_range: null` and `physical_energy_clip` stated with the target.
- New 2: Table 1 gains T2-only and clash rows; new Table 2 with the stochastic T2 from collapse; term 1's claim restated.
- New 3: term-1 gate in the λ hold, kill rule and fallback; difference 1 corrected.
- New 4: bounded `PriorFlow` as B1 with a Monte Carlo test.
- New 5–6: two-sided decision on J_F and low-band recall; n_d from P4; interval rules; A/A width check; the 80% rule dropped.
- New 7: holds counted past n_min rows, ESS pooled over g, reversible retirement, one λ step per round, λ_g as covariate with a preregistered reading.
- New 8–16: retention [D]/[U]; variance claims weakened; failures out of term 1, V_π̂, κ; U bound = finite variance; shared search counted once, readout cost restated; Var(w), k_H notation; length box 2Z(r + 3 Å); lower-bound reference; A6. Residues: σ_d ≤ 0.5 enforced; parents stored. Draft 2's "removes" table is folded in here.