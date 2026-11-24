# Neural SDEs that learn enhanced-sampling properties: design note (v4)

**Status:** design proposal, not policy. **Date:** 2026-11-24. **Version:** v4. v2 revised v1 (`energy_sampling/SCRATCH/nsde_design_nov24/design_note_raw.md`) after an independent review (`critique.json`, same folder) and added the owner's own proposal (§3.19). v3 fixes the errors a second review found in v2 and the corrections from a re-derivation of §3.19 with closed-form OU checks. v4 adds what pins the learned helper in §3.19 (Table 5c) and updates D5. Appendix A lists the changes.

**Sources.**
- Four derivation and verification pairs (committor, bias and free energy, collective variables, simplest targets), two literature surveys, and a toy catalogue with a validated ground-truth solver.
- An independent review of v1, which re-ran the rung-1 pass criteria at that rung's own settings.
- The owner's proposal and its analysis in the session of 2026-11-24, and a novelty search for that proposal.

Every numerical check ran on CPU on a 1D or 2D toy, most with one seed. Scripts and outputs are in `energy_sampling/SCRATCH/nsde_design_nov24/` (a gitignored local folder): `derive_committor/`, `verify_committor/`, `derive_bias/`, `verify_bias/`, `derive_cv/`, `verify_cv/`, `derive_simple/`, `verify_simple/`, `lit_tps_committor/`, `lit_bias_cv_transport/`, `toys/` and `critic/`. The raw records are `raw_derivations.json`, `raw_surveys.json`, `raw_novelty.json` and `critique.json` in the same folder. The scripts still hard-code their original scratchpad output folder (`OUT = ...`), so edit it before rerunning any of them. The v3 closed-form OU checks of §3.19 are `check_owner_idea/ou_checks.py` and `check_owner_idea/ou_checks.json` in the same folder. The v2 and v3 drafts and the reviser's notes are in `revision/`.

**Claim labels.**
- **PROVED**: the derivation and a numerical check agree.
- **DERIVED**: derivation only.
- **CONJECTURE**: neither.
- **REFUTED**: shown false.

Where a verifier corrected a deriver, this note uses the corrected statement and says so. [cont.] marks a continuous-time statement. [EM] marks a statement about the Euler–Maruyama (EM) chain at finite step dt. The analysis of the owner's proposal (§3.19) is DERIVED in the session; in v3, items 3, 7, 9 (the KL-form parts), 11, 12 and 13 (the cooling cost) are PROVED on a closed-form Gaussian OU Euler–Maruyama chain, and the rest stays DERIVED (§3.19 Status). Statements marked "reviser" were added in v2 and are unchecked unless they say otherwise. "Re-deriver" marks a v3 correction from that re-derivation. "Reviewer" marks a point taken from the independent review of v1.

---

## 1. Topline

1. **Every path formulation with a fixed Langevin reference is one Doob h-transform** (§3.1–3.9, 3.12, 3.13). The comparators in §3.10, 3.11, 3.14 and 3.15 are not. With a reward $R=\tilde\pi\,g$:
   - The trajectory-balance (TB) target is the reference reweighted by $g$.
   - The optimal learned drift is $u^*=\sigma^2\nabla\log h=(2/\beta)\nabla\log h$, where $h(x,t)=\mathbb E[g(X_T)\mid X_t=x]$.
   - A start-dependent $\log Z_\phi(x_0)$ converges to $\log h(x_0,0)$. In the owner's form, add $\log\tilde\pi(x_0)-\log\mu_0(x_0)$. [PROVED in 1D with a trained per-start head (§3.5). Elsewhere it was checked only with a fixed $u^*$, or with a point start and a scalar $\log Z$.]

   The committor is the stopped case. There $h=q$, the drift $\sigma^2\nabla\log q$ does not depend on time or on the start, and $\log Z_\phi(x_0)=\log q(x_0)$ gives a second, absolute readout of the committor.

   A fixed horizon gives a time-dependent $h$, which tends to the unconditioned reference as $T\to\infty$. It encodes the slowest mode $\varphi_1$ and the rate. It gives only a *soft* version of the committor control.

   In continuous time, with $R=\tilde\pi g$, the same $\sigma$, and the last factor $p_B(x_0\mid x_1)$ kept, the owner's fixed backward kernel gives the same optimum as the forward-reference form. At finite dt the two targets differ by $e^{C_{dt}}$ (Lemma C). For h-type readouts, implement it as the forward-reference density ratio.

2. **The committor result is already published; what remains open is narrower.**
   - Drifts trained by a path objective toward the stationary committor drift $\sigma^2\nabla\log q$, against a fixed Langevin reference: Earle & Van Koten (2025) and REACT (Du et al., 2026). REACT covers stopping times, off-policy value matching, a uniqueness theorem, and chignolin.
   - Related work that does not path-train toward $\sigma^2\nabla\log q$: Singh & Limmer 2023 (forward KL on transition-path-sampling data, a time-dependent $q(x,t)$ at fixed $t_f$); Singh & Limmer 2024 (the spectral-window theory); Yuan et al. 2023 (a $q$ learned by a PDE loss, then plugged into the control).
   - TB or log-variance with a learned $\log Z$ for transition paths: TPS-DPS (ICLR 2025). It uses a fixed horizon, no backward kernel, and a state-only bias.
   - Not found in the surveys (which does not prove absence):
     - TB with a fixed Langevin backward kernel and a per-start $\log Z$ read as the committor.
     - TB or log-variance under a fixed Langevin backward, compared with non-path baselines at matched force calls as the barrier grows. This depends on REACT's Appendix F, which was not read; REACT v4 is dated 23 Nov 2026.
     - A collective variable (CV) learned as a bottleneck inside $u_\theta$ under a path loss.
     - Quantified finite-dt biases.
     - Transfer across temperature and across systems. Generalisation across starts is standard (REACT, and every PDE or Dirichlet committor learner).

3. **Finite dt and the sampling law decide practice.**
   - Use a per-start $\log Z$. A scalar one is unbiased only on-policy.
   - Off-policy reuse has a measurable bias law, not a ban. With fixed-variance Gaussian kernels, the free-drift least-squares fixed point is biased per step by about $\sigma^2dt\,H(u^*-u_b)$, with $H=-\nabla^2\log h$ and $u_b$ the behaviour drift (Lemma D).
     - One Ornstein–Uhlenbeck (OU) check, with the behaviour drift at half the optimum and $dt/\gamma^2=0.22$: 14% drift error off-policy, against 0.6% on-policy.
     - The law predicts a small bias for a lagging buffer on smooth targets.
     - For the tied form the zero set does not depend on the behaviour law in continuous time [DERIVED, reviewer; Lemma D].
     - Rung 5 measures the law.
   - Report the importance-sampling $\log Z$. The TB-fitted value is low by the KL.
   - Put a floor $\epsilon$ under hard indicator rewards. Without one, reverse KL and TB are infinite at every dt.
   - Stopped problems carry an $O(\sqrt{dt})$ boundary-monitoring bias: +4% in $q$ at $x_0=-0.8$, dt = 1e-3. Brownian-bridge killing reduces it to $O(dt)$ at no extra force calls. That was shown in 1D with point boundaries; the method over-corrects below $\epsilon^*$ and is untested on curved sets and in $d>1$ (rung 10).
   - On-policy TB has **spurious zero-variance solutions**. They have the right CV but the wrong profile and the wrong normaliser. A convergence check must therefore include the ELBO and the fraction of paths that exit through B (Lemma G). The importance-sampling $\log Z$ cannot be that check, because it is unbiased for any policy with full support.

4. **Bias potentials from path losses: one negative result, narrowly stated.**
   - An on-policy reverse path-KL with a fixed, $F$-free, pointwise running cost does not have the flat free-energy bias $-F(s)$ as its optimum for a proper CV.
     - The necessary and sufficient condition on the cost is PROVED; it contains the conditional force variance.
     - Impossibility for *every* $F$-free cost is DERIVED (argued, not proved). It covers only pointwise costs, the $V(s)$ family, $s=x_1$ and the ergodic limit.
     - The only $F$-free cost tested, $c_W$, left 0.65 kT RMS error.
     - Self-consistent outer iterations, which feed the current estimate of $F$ into the cost, are not covered.
   - Other path optima *are* bias potentials. Ground-state tilts give $-(2/\beta)\log\varphi_\lambda$ (§3.9). The stopped committor drift is the bias $-(2/\beta)\log q$. With $U_1$ known, §3.12 gives $-\nabla(U_1-U_0)$.
   - The path objective that reaches $-F$ inside $u_{\rm phys}+u_\theta$ is a forward cross-entropy from projected Langevin trajectories to a Markov CV model. It reduces to mean-force regression, which is ABF.
     - On one easy toy it had the lowest error per force call from 6.4e4 calls on (Table 4b). The crossover with umbrella sampling plus WHAM lies between 3.2e4 and 6.4e4 calls, and the Fourier-regression variant leads only marginally at 6.4e4. The VES, metadynamics and OPES-like baselines it beat were not reproduced by the verifier.
     - At 3.2e4 calls, umbrella sampling with WHAM (0.36 kT) beat both mean-force variants (0.51 and 1.35 kT).
   - For free-energy differences, a fixed annealed backward with a scalar $\log Z$ has a floor of $\mathrm{KL}(\pi_0\Vert\rho^R_0)$. A per-start $\log Z$ removes it (verifier). Plain Jarzynski still beat a trained escort on the Gaussian toy.

5. **CVs through the drift.**
   - A bottleneck drift $\sigma^2\nabla\log f(s_\theta(x))$ is exact if and only if $q$ is a function of $s$. On a rotated 2D double well, a linear $s_\theta$ got the angle error below 2° within about 4–5e6 force calls, and to 0.4° by 3.1–3.3e7 (two runs). The profile was still about 10% off.
   - For fixed-horizon targets, a fixed CV is exact at all times if and only if it is *lumpable*: $Ls$ and $\lvert\nabla s\rvert^2$ are both functions of $s$. This is the classical criterion for $s(X_t)$ to be an autonomous diffusion. The committor generally is not lumpable.

6. **The owner's proposal (§3.19): a start-conditioned sampler with a Boltzmann endpoint and a bias anchored to physics.** [DERIVED in the session; the value identities of items 3, 9, 11 and 12 PROVED on a closed-form OU Euler–Maruyama chain (§3.19)]
   - Pin the backward at the start $A$, and the endpoint is exactly Boltzmann for *any* backward. The choice of backward then selects which transport from $\delta_A$ to $\pi$ is learned.
   - With the physical bridge as the backward, the target is the Schrödinger half-bridge from $\delta_A$ to $\pi$ with the physical Langevin as reference: $u^*=\sigma^2\nabla\log h$, with $h=\mathbb E_{\rm phys}[\pi(X_T)/p_T(X_T\mid A)\mid X_t=x]$. Among all drifts with a Boltzmann endpoint it has the smallest bias, and its paths that land in $B$ are physical paths from $A$ into $B$.
   - The only intractable quantity is the scalar $\log p_T(x_T\mid A)$, and no fixed tractable backward supplies it. The combined reverse KL, minimised jointly over the drift and a learned helper backward pinned at $A$, has the minimum-bias object as its $c\to\infty$ optimum [PROVED on OU]. That holds only when the helper is trained by maximum likelihood on the forward's own paths (ASBS corrector matching). Under a squared TB residual with a free $\log Z$, every helper has a zero-residual partner, so zero residual does not identify the optimum [REFUTED as a training statement, OU]. A fixed backward plus a KL term does not reach the optimum either. What pins the helper is the KL to physics, which weakens as $c$ grows. Train by alternation with the helper frozen while the forward converges, since joint gradient training slows by a factor of order $c$. Give the helper a Brownian-bridge variance schedule near the pin (Table 5c, PROVED on OU).
   - There is a certificate that needs no helper: at the optimum, the Girsanov log-weight is a function of the endpoint only, and it flags the wrong zero-residual points above. Transition probabilities $P_{\rm phys}(X_T\in B\mid A)$ are unbiased by reweighting for any trained drift with full support.
   - The novelty search (92 papers, 8 read in full) found no one building the whole thing. Chen, Georgiou & Pavon 2015 pose the same variational problem (underdamped, from a given law to a colder Boltzmann target) and solve it only for linear systems. ASBS supplies the training machinery. TITO is the MD-trained, transferable competitor.
   - Pending: D5 (fixed backward or learned helper). The proposed first step is a grid comparison that needs no training (ladder branch O0). Nothing has been run, and the owner has not given a go.

7. **First experiment** (§6, rung 1).
   - Toy: 1D asymmetric double well, $U=(x^2-1)^2+0.25x$, $\beta=5$, stopped at the two minima.
   - Start: point start $x_0=0$.
   - Reward: $\mathbf 1_B+0.01\,\mathbf 1_A$.
   - Model: free MLP drift $u_\theta(x)$ with a scalar $\log Z$, trained by on-policy TB.
   - Success, in one run at dt = 2e-3, repeated at dt = 1e-3, within 2e7 training force calls at dt = 2e-3 and 4e7 at dt = 1e-3, with bridge killing on:
     - the ELBO gap $\log h_\epsilon(0)-\mathbb E[w]$ is at most 0.007 nats, where $\log h_\epsilon(0)=-0.953$. The exact control gives 0.004 and the spurious $\sigma^2\nabla\log q$ gives 0.020;
     - the B-exit fraction is within 2 standard errors of 0.984;
     - the residual variance is within 2× the floor that the exact control reaches at the same dt.
   - The importance-sampling $\log Z$ is recorded as a readout, not a test. It would pass the spurious solution (−0.9547 against −0.9529).

---

## Design summary

**Forward process.** $dX_t=\big[-\nabla U(X_t)+u_\theta(X_t,t;c)\big]dt+\sigma\,dW_t$, with $\sigma^2=2/\beta$ and $X_0\sim\mu_0$. It runs to a fixed horizon $T$ or a stopping time $\tau$, and $c$ is an optional condition (start $A$, $\beta$, system). The physical force is fixed; only $u_\theta$ is learned.

**Table S1.** The design choices this note analyses. Each row is one component, the middle column lists the options covered, and the last column says what the choice decides.

| component | options analysed | what the choice decides |
|---|---|---|
| start | point $x_0$; law $\mu_0$ over $D$; start $A$ as a network input (§3.19) | where $h$ is identified; whether a scalar $\log Z$ suffices (Lemma D) |
| horizon | fixed $T$; stopping time at $A\cup B$; $T\sim\mathrm{Exp}(\gamma)$ (§3.18); $T$ as a network input | time-dependent or time-independent optimum; force calls spent waiting |
| endpoint target (reward $R=\tilde\pi g$) | $g=\mathbf 1_B+\epsilon\mathbf 1_A$ (committor); smooth $g$ (terminal conditioning); killing $e^{-s\tau}$ (rates); $R=\tilde\pi_1$ (transport); $R=\tilde\pi$ from a point start (§3.19) | which h-transform $u_\theta$ converges to (Lemma B) |
| reference process | Table S2 | what the learned drift is a bias relative to |
| backward kernel or helper | none evaluated (forward-reference ratio); reversed Langevin; Brownian bridge pinned at the start; physical bridge (needs $p_T$); learned helper | for h-type targets, only finite-dt terms (Lemma C); for a Boltzmann endpoint from a point start, *which* transport is learned (§3.19, D5) |
| form of $u_\theta$ | free $u(x,t)$; tied $\sigma^2\nabla\log h_\theta$; bottleneck $\sigma^2\nabla\log f(s_\theta(x))$; conditioned on $A$, $\beta$ or system | gauge, CV readout, off-policy behaviour (Lemma D), amortisation |

**Table S2.** Reference processes. "Tractable" means the density can be evaluated: the per-step path density, or the time-$T$ transition density $p_T(y\mid x)$. The last column says what $u_\theta$ is a bias relative to.

| reference | path density tractable? | $p_T$ tractable? | the learned drift is the bias relative to |
|---|---|---|---|
| physical overdamped Langevin ($-\nabla U$, $\sigma$) | yes (Gaussian per EM step) | no | the physical dynamics: $u_\theta$ is a bias force (§3.1–3.11, §3.19) |
| OU, or free Brownian motion | yes | yes (closed form) | free diffusion: $u_\theta$ must also carry the physical force (§3.12) |
| Brownian motion from a pinned start (the implicit reference of this repository's sampler; its bridge is the sampler's $P_B$) | yes | yes | free diffusion tied to the start: the learned part largely cancels the force (§3.19 item 7) |
| annealed Langevin along $U_t$ | yes | no | the protocol (Jarzynski, escorts; §3.13–3.14) |
| hot Langevin, $\beta_{\rm ref}<\beta$, run at the model's $\sigma$ as the drift $-(\beta_{\rm ref}/\beta)\nabla U$ (Lemma C(d)) | yes | no | hot physics: the bias also has to cool (§3.19 item 13) |
| underdamped Langevin | yes, on velocity increments only (hypoelliptic) | no | phase-space physics (rung 14) |

---

## 2. Formulation map

**Notation.**
- $\sigma^2=2/\beta$, so $\beta^{-1}=\sigma^2/2$ and $1/(2\sigma^2)=\beta/4$.
- $\tilde\pi=e^{-\beta U}$ and $\pi=\tilde\pi/Z_U$. Drift $b=-\nabla U$. Generator $Lf=b\cdot\nabla f+\beta^{-1}\Delta f$.
- $P^u$ is the law of $dX=(b+u)dt+\sigma dW$, and $P=P^0$ is the reference.
- TB identity $Z\mu_0P_F=R\,P_B$, with log-weight $w=\log R+\log P_B-\log\mu_0-\log P_F$ and residual $\delta=\log Z-w$.
- Extensions of the common notation, collected from the derivers:
  - $g=R/\tilde\pi$ and $h(x,t)=\mathbb E[g(X_T)\mid X_t=x]$; time-to-go $\vartheta=T-t$.
  - $D=(A\cup B)^c$, $\tau$ the first exit from $D$, $q$ the committor, and $h_\epsilon=\epsilon+(1-\epsilon)q$.
  - $(\lambda_k,\varphi_k)$ are eigenpairs of $-L$ in $L^2(\pi)$ with $0=\lambda_0<\lambda_1\le\lambda_2$, and $\mathcal E(f)=\beta^{-1}\int\lvert\nabla f\rvert^2d\pi$.
- Rates: reactive flux $\nu_{AB}=\beta^{-1}\int\pi\lvert\nabla q\rvert^2$, $\rho_A=\int\pi(1-q)$, rate constant $k_{A\to B}=\nu_{AB}/\rho_A$. The inputs write "k_AB" for $\nu_{AB}$ (toy catalogue) and for $k_{A\to B}$ (committor family). This note keeps them apart.
- Two forms of the TB identity:
  - *Forward-reference form*: $Z(x_0)P_F(\tau\mid x_0)=g(x_T)P(\tau\mid x_0)$, with no backward kernel evaluated.
  - *Owner form*: the same identity with the reversed Langevin as $P_B$.

**Table 1.** The formulations that survived verification, plus the owner's proposal in the last row, which is checked only on a closed-form OU chain. Each row states the optimum that $u_\theta$ and $\log Z$ converge to, in continuous time unless marked [EM]. For the h-type rows (null case to ground-state tilt), $\log Z$ is in nats and in the forward-reference form; for the owner form add $\log\tilde\pi(x_0)-\log\mu_0(x_0)$ (Lemma C, §3.0). The transport rows (§3.12–3.14) state their start-term convention in the cell. Status is as left by the verifier. The proof is in the section named in column 1, and the toy in the last column is where the claim was checked numerically. Formulations that were named but not developed are in §3.18.

| formulation (proof) | target property | forward setup: start law; horizon or stopping; reward | backward reference $P_B$ | loss | optimum of $u_\theta$ (drift) | optimum of $\log Z$ (nats) | status | toy test |
|---|---|---|---|---|---|---|---|---|
| Null case (§3.1) | none; unit test | $x_0\sim\pi$; fixed $T$; $R=\tilde\pi$ | reference reversal | TB, log-variance | $0$ [cont.]; small contraction near $T$ [EM; closed form for OU] | $\log Z_U$ [cont.]; TB value $\log Z_U-\mathrm{KL}$ [EM] | PROVED (1D) | OU; quartic double well |
| Fixed-horizon terminal conditioning (§3.2) | reference conditioned on $X_T\in B$ or tilted by $g$; transition probability | point or $\mu_0$; fixed $T$; $R=\tilde\pi g$, $g$ smooth (hard $\mathbf 1_B$ gives infinite loss) | reference reversal, equal to the forward ratio | TB, log-variance, reverse KL, forward KL | $\sigma^2\nabla\log h(x,t)$, time-dependent | $\log h(x_0,0)$ | PROVED [cont. and EM, 1D] | 1D double well with sigmoid reward; OU soft constraint |
| Spectral window of the fixed horizon (§3.3) | $\varphi_1$ (an affine function of $q$ in metastable systems), $\lambda_1$, rate | as above with $1/\lambda_2\ll T\ll1/\lambda_1$; two or more $T$ | as above | as above | $\approx\sigma^2c\nabla\varphi_1/(\langle g\rangle+c\varphi_1)$ once $\vartheta\gtrsim\lambda_2^{-1}\log(C\lambda_2/\lambda_1)$; two-state form $\sigma^2\nabla\log(\epsilon_{\rm eff}+(1-\epsilon_{\rm eff})q)$ with $\epsilon_{\rm eff}\approx\langle q\rangle\lambda_1\vartheta$ | $\log Z(T)$ with $Z'(T)=-\lambda_1e^{-\lambda_1T}\varphi_1(x_0)\langle\varphi_1,g\rangle+O(e^{-\lambda_2T})$ | exact parts PROVED; the $\varphi_1\approx$ affine($q$) and two-state forms DERIVED (heuristic, numerically supported); "$u^*\approx\sigma^2\nabla\log q$" REFUTED | 1D double well, rotated double well, Müller–Brown grids |
| Stopped committor (§3.4) | $q$; reactive ensemble | $x_0\in D$, point or $\mu_0$; stop at $A\cup B$; $R=\mathbf 1_B$ | forward ratio only | reverse KL, per-start TB, log-variance | $\sigma^2\nabla\log q$, independent of time and start | $\log q(x_0)$; [EM] $\log q_{dt}(x_0)$, off by $O(\sqrt{dt})$ | PROVED [cont.]; $O(\sqrt{dt})$ bias PROVED [EM, 1D lattice]; on-policy uniqueness REFUTED | 1D double well; rotated double well |
| Floor-regularised, tied stopped TB (§3.5) | $q$ without the singularity at $\partial A$ | $\mu_0$ over $D$; stop; $R=\mathbf 1_B+\epsilon\mathbf 1_A$ | forward ratio | TB with $\log Z_\theta(x_0):=\log h_\theta(x_0)$ | $\sigma^2\nabla\log(\epsilon+(1-\epsilon)q)$ | $\log(\epsilon+(1-\epsilon)q(x_0))$; read $q=(h-\epsilon)/(1-\epsilon)$ | PROVED [cont.]; readout returns $q_{dt}$ [EM]; training single-seed | 1D double well, spread starts |
| Bottleneck CV (§3.6) | reactive coordinate $s_\theta$ | as the stopped rows | forward ratio | log-variance, TB | $\sigma^2\nabla\log f(s(x))=u^*$ iff $q=F\circ s$; otherwise a policy-dependent compromise | $\log h_\epsilon(x_0)$ | PROVED (rotated double well: grid, Monte Carlo, two training runs); claim that the restricted reverse-KL optimum is a projection of $u^*$ REFUTED | rotated double well (exact); Müller–Brown (not exact) |
| Forward-KL drift regression (§3.7) | $\sigma^2\nabla\log q$ from reactive samples | no simulation during the fit; samples from rejection, MD or replay | none | least squares of $\Delta/dt-b$ onto $u_\theta$ | $\sigma^2\nabla\log q_{dt}+O(dt)$ [EM, stopped] | none | PROVED; EM target corrected to $q_{dt}$ | 1D double well rejection samples |
| First passage and killing (§3.8) | $F(T)=P(\tau_B\le T)$, $\mathbb Ee^{-s\tau_B}$; $\lambda_1$, MFPT | point or $\mu_0$; stop at $B$ or $T$, or at $B$ with weight $e^{-s\tau_B}$ | forward ratio | per-start TB, log-variance | $\sigma^2\nabla\log H(x,T-t)$; $\sigma^2\nabla\log h_s(x)$ | $\log F_{x_0}(T)$; $\log h_s(x_0)$ | PROVED (1D) | 1D double well, $\beta=3$; OU |
| Ground-state tilt and killing (§3.9) | $\varphi_\lambda$, bias $-(2/\beta)\log\varphi_\lambda$, $\Lambda_0$; weak tilt gives $\varphi_1$; killing in $B$ gives $1-q$ | long $T$; path reward $e^{\lambda\int f}$ or killing | reversal (the weights are invariant under reversal) | TB, log-variance, reverse KL | $\sigma^2\nabla\log\varphi_\lambda$ for $T-t\gg1/(\Lambda_0-\Lambda_1)$ | $\Lambda_0T+\log\varphi_\lambda(x_0)+$ const | DERIVED (spectral identities checked; no training); weak-tilt horizon corrected | rotated double well; 1D double well |
| Flat bias by reverse path KL (§3.9) | is $V=-F(s)$ reachable? | stationary drift; ergodic window; running cost $c$ | Langevin tilted by $e^{-\int c}$ | ergodic reverse KL | ground-state drift; $V^*=-F$ iff $c$ contains $F'$, $F''$ and the conditional force variance | principal eigenvalue | condition PROVED; impossibility for every $F$-free pointwise cost DERIVED (argued); the one $F$-free cost tested leaves 0.65 kT | periodic 2D toy |
| Mean-force regression (§3.10) | $F'(s)$; bias $-F$ or $-(1-1/\gamma)F$ | Langevin with the current bias $V(s)$; time window | none (the data are the fixed Langevin) | $\mathbb E\int(F_\theta'(s)-\partial_1U)^2dt$ | $F'(s)\nabla s$, cancelling the mean CV force | none | PROVED, given $x_\perp$ in conditional equilibrium given $s$ | periodic 2D toy |
| Variationally enhanced sampling (VES), non-path (§3.11) | bias with a prescribed CV marginal | biased stationary sampler | none | $\Omega[V]$ | $-\nabla V^*$ with $V^*=-F-\beta^{-1}\log p_{\rm tgt}$ | none | PROVED | periodic 2D toy |
| Transport, fixed reference that does not change in time (§3.12) | sampler of $R/Z_R$; $\Delta F$ | $x_0\sim\rho_T$ (exact) or $\pi_0$; fixed $T$; $R=\tilde\pi_1$ | reversal of the start (noise) potential's Langevin | TB, log-variance; importance-sampling readout | $\sigma^2\nabla\log(\rho_{T-t}/\pi_0)$ | start drawn from normalised $\rho_T$: $\log Z_R$. Scalar, start drawn from normalised $\pi_0$: $\log Z_R-\mathrm{KL}(\pi_0\Vert\rho_T)$. Writing the start term with $\tilde\pi_0$ subtracts $\log Z_0$, which gives $-\beta\Delta F$ and $-\beta\Delta F-\mathrm{KL}(\pi_0\Vert\rho_T)$. Importance sampling exact in all cases | PROVED (1D) | OU to Gaussian; OU to asymmetric double well |
| Annealed fixed backward (§3.13) | $\Delta F$ along a protocol $U_t$ | $x_0\sim\pi_0$; drift $-\nabla U_t+u$ | reverse-protocol Langevin, fixed | TB, scalar or per-start $\log Z$ | $\sigma^2\nabla\log h$ with $h=\mathbb E[e^{-\beta\int_t^T\partial_rU_r}\mid X_t=x]$ | start term written with $\tilde\pi_0$. Scalar: $-\beta\Delta F-\mathrm{KL}(\pi_0\Vert\rho^R_0)$; per-start: $\log h(x_0,0)$, zero loss | PROVED (1D Gaussian), corrected by the verifier | 1D Gaussian pair |
| Shared escort (§3.14) | exact $\Delta F$; exact samples of $\pi_t$ | $x_0\sim\pi_0$; drift $-\nabla U_t+u$ | reverse-time Langevin for $\pi_t$ plus the reversed learned escort (learned, not fixed) | TB, log-variance; Dirichlet functional | any $u$ with $\nabla\cdot(\pi_tu)=-\partial_t\pi_t$; the gradient solution is unique | start term written with $\tilde\pi_0$: $-\beta\Delta F$ exactly | PROVED (2D Gaussian) | 2D Gaussian pair |
| Dirichlet principle, non-path (§3.15) | $q$; flux $\nu_{AB}$ | none; samples $x\sim\rho$ with weights $\pi/\rho$ | none | $\mathbb E_\rho[(\pi/\rho)\lvert\nabla f\rvert^2]$ with $f=0$ on $A$, $1$ on $B$ | $q$ (drift $\sigma^2\nabla\log q$) | none | PROVED | 1D grid; Müller–Brown |
| Start-conditioned minimum-bias Boltzmann transport: the owner's proposal (§3.19) | Boltzmann endpoint from any start $A$ at finite $T$; physical $A\to B$ paths; $P_{\rm phys}(X_T\in B\mid A)$ by reweighting | point start $A$, also a network input; fixed $T$; $R=\tilde\pi$ | pinned at $A$: the physical bridge (needs $p_T$) or a learned helper. The reference $P_{\rm phys}$ is fixed and enters through a KL term | combined reverse KL $c\,\mathrm{KL}(P_F\Vert\pi P_B)+\mathrm{KL}(P_F\Vert P_{\rm phys})$, i.e. the expected residual $(1+c)\log P_F-c(\log\tilde\pi+\log P_B)-\log P_{\rm phys}$; the helper trained by maximum likelihood on forward paths | with a helper trained that way: $\sigma^2\nabla\log h$, $h=\mathbb E_{\rm phys}[(\pi/p_T(\cdot\mid A))^{c/(1+c)}(X_T)\mid X_t=x]$; as $c\to\infty$, the minimum-bias drift | $-(1+c)\log N_c-c\log Z_U$ with $N_c=\int\pi^{c/(1+c)}p_T(\cdot\mid A)^{1/(1+c)}$. Not a readout of the bias cost (the $-c\log Z_U$ term diverges); read $\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$ as $\frac\beta4\mathbb E\int\lvert u\rvert^2dt=-\mathbb E_{P_F}\log w$ | combined-KL optimum PROVED (OU EM chain); identifiability under a squared residual with free $\log Z$ REFUTED (OU); the rest DERIVED | OU, $\beta=5$, $A=2$, $T=2$ (closed form, `check_owner_idea/ou_checks.py`); proposed: 1D double-well grid (branch O0) |

---

## 3. The formulations

### 3.0 Common machinery

**Lemma A (Girsanov).** Take drifts $b+u$ and $b+v$ with the same $\sigma$, up to a stopping time $\tau$. Assume a Novikov-type condition, $\mathbb E\exp\big(\tfrac1{2\sigma^2}\int_0^\tau\lvert u-v\rvert^2dt\big)<\infty$. Bounded $u-v$ suffices on a fixed horizon. For an unbounded stopping time it needs an exponential moment of $\tau$ at rate $\sup\lvert u-v\rvert^2/(2\sigma^2)$, which for hitting times holds only below the tail rate $\lambda_D$; otherwise localise, as Lemma B does. Then
$$\log\frac{dP^u}{dP^v}=\frac1{\sigma^2}\int_0^\tau(u-v)\cdot\big(dX_t-(b+v)dt\big)-\frac1{2\sigma^2}\int_0^\tau\lvert u-v\rvert^2dt .$$
Under $P^u$ we have $dX-(b+v)dt=(u-v)dt+\sigma dW^u$. Hence $\mathrm{KL}(P^u\Vert P^v)=\frac{\beta}{4}\mathbb E_{P^u}\int_0^\tau\lvert u-v\rvert^2dt$.

[EM] For two Gaussians with the same covariance, exactly,
$$\log p_u(x'\mid x)-\log p_0(x'\mid x)=\sigma^{-2}u\cdot(\Delta-b\,dt)-\lvert u\rvert^2dt/(2\sigma^2),\qquad\Delta=x'-x .$$

**Lemma B (Doob transform; the master identity).** Let $h>0$ solve $(\partial_t+L+\lambda f)h=0$ on $D\times[0,T)$, with $h=r$ at the exit or terminal time $\tau$. Here $f$ is a running reward or minus a killing rate, and $\lambda f=0$ for a plain terminal reward. The target is
$$\frac{dQ}{dP}=\frac{e^{\lambda\int_0^\tau f}\,r(X_\tau)}{h(x_0,0)} .$$
It is a probability measure because $e^{\lambda\int_0^tf}h(X_t,t)$ is a martingale.

1. *Itô under $P^u$.* $d\log h=[\partial_t\log h+(b+u)\cdot\nabla\log h+\tfrac{\sigma^2}{2}\Delta\log h]dt+\sigma\nabla\log h\cdot dW^u$. Substitute $\Delta\log h=\Delta h/h-\lvert\nabla\log h\rvert^2$ and the PDE, which reads $\partial_th/h+b\cdot\nabla h/h+\tfrac{\sigma^2}{2}\Delta h/h=-\lambda f$. This gives
$$d\log h=\big[u\cdot\nabla\log h-\tfrac{\sigma^2}{2}\lvert\nabla\log h\rvert^2-\lambda f\big]dt+\sigma\nabla\log h\cdot dW^u .$$
2. *Integrate from 0 to $\tau$.* $\log r(X_\tau)-\log h(x_0,0)$ equals the integral of the bracket plus $\int\sigma\nabla\log h\cdot dW^u$.
3. *Lemma A with $v=0$.* $\log dP^u/dP=\sigma^{-1}\int u\cdot dW^u+(2\sigma^2)^{-1}\int\lvert u\rvert^2dt$.
4. *Combine.* Put $u^*:=\sigma^2\nabla\log h$ and use $\lvert u\rvert^2-2u\cdot u^*+\lvert u^*\rvert^2=\lvert u-u^*\rvert^2$:
$$\log\frac{dP^u}{dQ}=\log\frac{dP^u}{dP}-\lambda\!\int_0^\tau\! f-\log r(X_\tau)+\log h(x_0,0)=\frac1\sigma\int_0^\tau(u-u^*)\cdot dW^u+\frac{1}{2\sigma^2}\int_0^\tau\lvert u-u^*\rvert^2dt .$$

Consequences [PROVED. The Lemma B identities were simulated on the 1D double wells, OU, the 2D rotated double well and the 1D Gaussian pair (§3.13). The periodic 2D toy and the 2D Gaussian pair had path simulations of other identities (§3.10, §3.14). Müller–Brown, three-hole, ring, neck and annulus have grid PDE solutions only, and alanine has nothing]:
- **KL in both directions.** $\mathrm{KL}(P^u\Vert Q)=\frac\beta4\mathbb E_{P^u}\int\lvert u-u^*\rvert^2$. Also $\mathrm{KL}(Q\Vert P^u)=\frac\beta4\mathbb E_Q\int\lvert u-u^*\rvert^2$, because under $Q$, $dW^u=dW^Q+\sigma^{-1}(u^*-u)dt$.
- **Uniqueness.** The minimiser $u=u^*$ is unique $dt\otimes dP^u$-a.e., so it is identified only where the paths go.
- **Stochastic optimal control.** Reverse KL is the control problem $J(u)=\mathbb E_{P^u}[\tfrac1{2\sigma^2}\int\lvert u\rvert^2-\lambda\int f-\log r]=\mathrm{KL}(P^u\Vert Q)-\log h(x_0,0)$.
- **TB residual.** At $u^*$ the forward-reference residual is $\delta=\log Z-\log h(x_0,0)$ on every path. So the variance is zero and $\log Z^*=\log h(x_0,0)$. At a fixed $u$, the squared loss is minimised at the ELBO $\log h(x_0,0)-\mathrm{KL}(P^u\Vert Q)$.
- **Factor check.** The factor is $\sigma^2=2/\beta$.
  - Survey check on the 1D double well at $\beta=3$, $x_0=-0.5$, dt = 2e-4: with $(2/\beta)\nabla\log q$, 99.9% of paths are reactive and the mean hitting time is 0.898 (unbiased reactive paths: 0.895). With $(1/\beta)\nabla\log q$, 81% are reactive and the mean is 1.451.
  - The toy solver's importance-sampling check has its smallest relative weight variance at factor 1 (0.0126, against 0.909 at 0.5 and 47.8 at 2).
- **Assumptions.**
  - $h>0$ on the support of $P^u$, and Girsanov integrability holds.
  - Wherever $\log h\to-\infty$ (the committor at $\partial A$), the argument is localised.
  - The reference and the model use the same $\sigma$.

**Lemma C (the owner's fixed backward kernel).** [cont.] Reversibility gives path detailed balance, $\pi(dx_0)P(d\tau\mid x_0)=\pi(dx_T)P_B(d\tau\mid x_T)$, where $P_B$ is the reference read backwards. So
$$\frac{R(x_T)P_B(\tau\mid x_T)}{Z}=\frac{g(x_T)\,\tilde\pi(x_0)\,P(\tau\mid x_0)}{Z},\qquad g=R/\tilde\pi .$$
Integrate with $x_0$ fixed. The $x_0$-marginal is $Q_0=\tilde\pi h(\cdot,0)/Z_R$ with $Z_R=\int R$, and $Q(\cdot\mid x_0)$ is the Lemma B transform with $r=g$.

(a) **$R=\tilde\pi r$.** $Q(\cdot\mid x_0)$ is the reference conditioned on, or tilted by, $r$, which is exactly the intended object [PROVED].

(b) **$R=r$, i.e. $\pi$ left out.** The target is tilted by $e^{\beta U(x_T)}$. The residual variance stays near 0.6 nats² and does not shrink with dt [PROVED].

(c) **Backward kernel with $\sigma'\ne\sigma$.** The two path measures are mutually singular, and the EM residual variance grows like $T/dt$ [DERIVED].

(d) **Backward Langevin for $U'\ne U$ at the same $\sigma$.** It reads out the $U'$ system, $u^*=\nabla U-\nabla U'+\sigma^2\nabla\log h'$ [DERIVED]. A "different temperature" backward kernel is admissible only as the drift $-(\beta'/\beta)\nabla U$ at the forward $\sigma$.

[EM] **Exact identity.** With $\Delta_k=x_{k+1}-x_k$ and midpoints $m_k$:
$$\log P(\tau\mid x_0)-\log P_B(\tau\mid x_N)=\log\pi(x_N)-\log\pi(x_0)+C_{dt},$$
$$C_{dt}=\tfrac{\beta dt}{4}\big(\lvert\nabla U(x_N)\rvert^2-\lvert\nabla U(x_0)\rvert^2\big)-\tfrac{\beta}{12}\textstyle\sum_kD^3U(m_k)[\Delta_k^{\otimes3}] .$$
- The second term is the trapezoid-rule remainder. It is exact for quartic $U$ and zero for quadratic $U$.
- $C_{dt}$ is the Metropolis-adjusted Langevin (MALA) log-acceptance ratio summed along the path. Over a fixed horizon its mean and its standard deviation are both $O(dt)$.
- The identity was checked path by path to 6e-13 (deriver) and 1.4e-10 (verifier).
- Prior art: annealed importance sampling with unadjusted-Langevin backward kernels works with this same EM time-reversal defect (Thin et al. 2021; Geffner & Domke 2021; Zhang et al. 2021, differentiable AIS; named by the reviewer, not fetched). This bears on novelty claim 1 of §4.

**Point start.** Keep the last backward factor $p_B(a\mid x_1)$. Dropping it leaves a residual-variance floor of $d/2$ and shifts the optimal $\log Z$ by $(d/2)(1+\log 2\pi\sigma^2dt)$. That constant is the verifier's correction: the deriver had dropped the leading $d/2$.

**Stopping times.** The reversed, unstopped Langevin is not a normalised backward policy over paths of variable length, because nothing stops it at $x_0$. With a stopping time, the identity holds only as a pathwise density ratio. That ratio reduces to the forward-reference form up to the constant $\beta U(x_0)$ and, at finite dt, the $O(dt)$ term $C_{dt}$ (rung 0, check (iii), tests this path by path). So for stopped targets the owner's $P_B$ enters only as this density ratio; its normalisation is never used. A killing indicator inside $P_B$ is redundant, since it equals 1 on every forward-stopped path.

**Backward sampling** [DERIVED]. Drawing $x_T\propto R$ and running $P_B$ gives exact samples of $Q$. At finite dt these are exact samples of the owner-form EM target, which differs from the forward-reference EM chain by $e^{C_{dt}}$. They come only from the equilibrium-started ensemble: the reversal of an A→B reactive path is a B→A reactive path, and is just as rare.

**Recommendation** (two verifiers and the simple-targets deriver agree):
- For h-type readouts (committor, hitting), use the forward-reference ratio $w_0=\log g(x_N)+\sum_k\log[p_{\rm ref}(x_{k+1}\mid x_k)/p_F(x_{k+1}\mid x_k)]$. It is exact for the discrete EM chain: $\mathbb E[e^{w_0}\mid x_0]=h_N(x_0)$.
- For Z-type readouts ($\Delta F$), use the normalised-backward form, which gives Lemma E's exact importance-sampling identity.
- Both cost the same force calls. The backward step reuses $\nabla U(x_{k+1})$, plus one extra call at the endpoint.

**Lemma D (start law and normaliser).**
- **Chain rule.** $\mathrm{KL}(\mu_0P_F\Vert Q)=\mathrm{KL}(\mu_0\Vert Q_0)+\mathbb E_{\mu_0}\mathrm{KL}(P_F(\cdot\mid x_0)\Vert Q(\cdot\mid x_0))$. The first term does not depend on $u$, so $u^*$ is the per-start transform whatever $\mu_0$ is.
- **Residual at $u^*$.** In the owner form, $\delta=\log Z+\log\mu_0(x_0)-\log\tilde\pi(x_0)-\log h(x_0,0)$, plus $C_{dt}$ at finite dt. This depends on $x_0$ only, and it is constant iff $\mu_0\propto\tilde\pi h_0$.
- **On-policy with detached samples** [DERIVED; consistent with every run]. For any $c(x_0)$,
$$\nabla_\theta\mathbb E[(c(x_0)-w_\theta)^2]=2\nabla_\theta\mathbb E_{\mu_0}\mathrm{KL}(P_\theta(\cdot\mid x_0)\Vert Q(\cdot\mid x_0)),$$
because $\nabla_\theta w=-\nabla_\theta\log P_\theta$ and $\mathbb E[\nabla_\theta\log P_\theta\mid x_0]=0$. So a scalar $\log Z$ costs gradient variance, not bias. This is the known on-policy equivalence of the TB or log-variance gradient with the reverse-KL gradient, which fails off-policy (Richter et al. 2020, VarGrad; Malkin et al. 2023, "GFlowNets and variational inference"; both named by the reviewer and not fetched by the surveys).
- **Off-policy** (behaviour law $P_b\ne P_\theta$) [DERIVED, untested]. At $u^*$ the stationarity condition picks up $2\,\mathrm{Cov}_b(c(x_0),\nabla_\theta\log P_\theta)\neq0$, so a per-start $\log Z_\phi(x_0)$ is needed. Its optimum is $\log h(x_0,0)$ in the forward-reference form, plus $\log\tilde\pi-\log\mu_0$ in the owner form.
- **Verifier correction: per-start $\log Z$ is necessary but not sufficient off-policy at finite dt** [PROVED in the OU affine class; general $U$ DERIVED].
  - The conditionals of $Q$ have variance below $\sigma^2dt$, and fixed-variance Gaussian kernels cannot represent them.
  - So the off-policy least-squares fixed point is, per step, $u_{\rm learned}-u^*\approx\sigma^2dt\,H(u^*-u_b)$ with $H=-\nabla^2\log h_{k+1}$.
  - The bias depends on the behaviour *mean* and not on its noise.
  - OU check, dt = 0.02, constraint width 0.3: a behaviour drift at half the optimum gave 14.2% relative L2 error in the learned drift. On-policy gave 0.6%. The correct mean with noise ×1.5 gave 1.0%.
  - This is one check at $dt/\gamma^2=0.22$, with a behaviour drift far from the optimum. The law predicts a small bias for a replay buffer that lags a smooth target slightly. Treat it as a law to measure (rung 5), not as a rule against replay.
- **Off-policy validity of the tied form** [DERIVED by the reviewer; consistent with Lemma F, whose identity was PROVED up to a $\sqrt{dt}$ discretisation gap]. Take $u=\sigma^2\nabla\log h_\theta$ and $\log Z=\log h_\theta(x_0)$. Itô's formula on paths generated by *any* drift $b+v$ with the same $\sigma$ gives
$$\delta=[\log h_\theta-\log r](X_\tau)-\int_0^\tau\frac{(\partial_t+L)h_\theta}{h_\theta}\,dt .$$
  - The behaviour drift cancels, so the zero set of tied TB does not depend on the behaviour law in continuous time.
  - For the tied arm, Lemma D's off-policy bias is therefore a finite-dt, fixed-variance effect only. Rung 5's prediction differs between the tied and free arms.
  - It may explain why REACT's off-policy value matching works [CONJECTURE].

**Lemma E (what $\log Z$ estimates)** [PROVED].
- For a normalised $P_B$ and a $P_F$ with full support, $\mathbb E_{P_F}e^w=\int RP_B=Z$ exactly, for every $u$ and every dt.
- $\mathbb Ew=\log Z-\mathrm{KL}(P_F\Vert Q)\approx\log Z-\tfrac12\mathrm{Var}\,w$, and a scalar TB $\log Z$ converges to $\mathbb Ew$.
- At finite dt, $Q$ is unreachable for every non-null target. So report $\log\frac1M\sum_ie^{w_i}$ together with $\mathrm{Var}\,w$.
- Example (OU training run): TB $\log Z=-4.5163$ against $\log h_N=-4.4213$, a gap of about the KL, 0.095.

**Lemma F (a tied normaliser turns TB into a backward-Kolmogorov residual).** Take $u_\theta=\sigma^2\nabla\log h_\theta$ and $\log Z_\theta(x_0)=\log h_\theta(x_0,0)$. Itô's formula gives, path by path,
$$\delta=\big[\log h_\theta(X_\tau)-\log r(X_\tau)\big]-\int_0^\tau\frac{(\partial_t+L+\lambda f)h_\theta}{h_\theta}(X_t,t)\,dt .$$
- The martingale term cancels. With hard boundary conditions the bracket is zero. Tied TB is then the Feynman–Kac residual of the PDE, sampled along the policy's own paths.
- It needs only $\nabla h_\theta$, because the Itô increments supply the Laplacian [PROVED: the RMS gap between the two sides is 0.855, 0.456 and 0.252 nats at dt = 4e-3, 1e-3 and 2.5e-4, scaling like $\sqrt{dt}$, against a residual RMS near 9 nats].
- Residuals integrated along a path can cancel, so a small loss localises errors poorly.
- Prior art: this identity, with the Itô increments supplying the Laplacian and a window-length trade-off, is the "diffusion loss" of Nüsken & Richter 2023 ("Interpolating between BSDEs and PINNs", arXiv:2112.03749, abstract fetched in survey 1), which includes committor examples.
- **Local losses.**
  - Per-step detailed balance with this tied flow is biased by an amount that does not vanish as dt → 0, and its loss landscape is flat to 0.1%.
  - In the committor verifier's 1D family: argmin $\lambda=-0.05$ instead of 0, and a $q$ readout error of −0.025 at $x=0$. Full-path TB has the correct minimum [PROVED in that family].
  - Sub-trajectory windows of $l$ steps suppress the bias by roughly $1/l$ [DERIVED].
  - Subtracting the known Itô term as a control variate may remove it [CONJECTURE].

**Lemma G (spurious zero-variance points under on-policy training)** [verifier; DERIVED in continuous time, supported numerically]. Take the stopped reward $R_\epsilon=\mathbf 1_B+\epsilon\mathbf 1_A$ and any exit set $E\subset\partial B$ (or $E\subset\partial A$).
- The transform conditioned on $E$, $u_E=\sigma^2\nabla\log P_x(X_\tau\in E)$, makes $\delta$ constant on its own paths: $-\log P(E)$, or $-\log P(E)-\log\epsilon$.
- $Q\ll P^{u_E}$ fails, so these points are not the target, but on-policy TB and log-variance cannot detect that.
- In a bottleneck class two of them survive:
  - $\sigma^2\nabla\log q$: KL 1.43e-3 nats; readout $\log q(x_0)$ instead of $\log h_\epsilon$.
  - $\sigma^2\nabla\log(1-q)$: KL 6.55 nats; readout $q(x_0)\approx5.9\times10^{-4}$ instead of 0.41.
  - Both are local minima of reverse KL under bounded perturbations.
- Residual-variance floors at dt = 2.5e-4 (rotated double well, $x_0=0$, $\epsilon=10^{-3}$): 7.25e-4 for the target, 1.23e-3 and 7.08e-4 for the two spurious points.
- **Convergence certificate**: $\mathrm{Var}\,\delta\to0$ with dt, AND ELBO $=\log h_\epsilon$, AND $P(\text{exit through }B)=q/(q+\epsilon(1-q))$.
  - The importance-sampling $\log Z$ is not part of the certificate, because it is unbiased for any full-support policy. At the rung-1 settings it passes $u_0=\sigma^2\nabla\log q$ (Table 9, §6).
- **A certificate without ground truth** (reviewer; needed at rung 15, where $q$ is unknown). Use the model's own readout $\hat q=(e^{\log Z_{\rm TB}}-\epsilon)/(1-\epsilon)$ and check two things:
  - the B-exit fraction against $\hat q/(\hat q+\epsilon(1-\hat q))$;
  - the ratio $(\log\hat Z_{\rm IS}-\mathbb E w)/(\tfrac12\mathrm{Var}\,w)\approx1$, which holds when the log-weights are close to Gaussian.

  It flags both spurious points. $u_0$ exits through B almost always, against a predicted 0.984. Read through a free TB $\log Z$, $\sigma^2\nabla\log(1-q)$ has $\log Z_{\rm TB}=\log(\epsilon(1-q))<\log\epsilon$, so $\hat q<0$. With bridge killing, the ratio computed from the reviewer's run at the rung-1 settings (Table 9; one seed, $10^5$ paths) was 1.00 at $u^*_\epsilon$ and 4.2 at $u_0$. Under plain monitoring the same ratio at $u_0$ is about 4e-4, because the heavy tails inflate Var $w$ (310.7 nats²); the ratio therefore flags $u_0$ in both cases, but in opposite directions.
- At $u=0$, a batch with no B-exit has *exactly* zero loss and zero gradient under the log-variance loss (the deriver said "tiny"). Under TB with a free scalar $\log Z$ (rung 1's loss), $\delta=\log Z-\log\epsilon$ is nonzero until $\log Z=\log\epsilon$. The per-batch $\theta$-gradient $2\delta\sum\nabla\log P_\theta$ is then zero-mean noise, not zero (reviewer).
- The same argument applies to any reward that has a hard-indicator component [DERIVED here, not checked].
- In the two runs tried, training from $u=0$ at $q_0\approx10^{-3}$ escaped to the target.

**Table 2.** Finite-dt error laws. Each was checked on the 1D toy named in the "measured" column; forms in $d$ dimensions are DERIVED. "Var" is the variance of the TB log-weight in nats².

| error source | where it applies | size law | measured check (toy, samples) | remedy |
|---|---|---|---|---|
| EM null defect $C_{dt}$ | only with an EM backward kernel | std $=O(dt)\sqrt{c_1+c_2T}$; Var $\approx\mathrm{Var}(dt[f_1(x_N)-f_1(x_0)])+\tfrac{\sigma^2}{6}dt^2\int\mathbb E\lVert\nabla^3U\rVert_F^2$, where $f_1=\tfrac\beta4\lvert\nabla U\rvert^2-\tfrac12\Delta U$ | quartic double well, $\beta=2$, $10^5$ paths: simulated/law = 1.03–1.16 for dt ≤ 0.01 | exact OU kernel; otherwise accept it. No normalised backward removes it. The floor over all drifts drops only 3–9% on the double well and 46% on OU (verifier) |
| fixed-variance kernel mismatch | every target with non-constant $h$ | Var $\approx\tfrac{\sigma^4}{2}dt\int\mathbb E_Q\lVert\nabla^2\log h\rVert_F^2dt$; for terminal width $\gamma$, $\approx\sigma^2dt/(2\gamma^2)$ | OU soft constraint: Var/law = 1.03 at dt = 0.0025; at $dt/\gamma^2=11$, 251 measured against 11 predicted (law invalid) | keep $\sigma^2dt\ll$ target width²; learn the kernel covariance (untested) |
| discrete boundary monitoring | stopped problems | $O(\sqrt{dt})$ bias in $q$, $\log Z$ and exit fractions, equal to an outward boundary shift of $0.5826\,\sigma\sqrt{dt}$ | 1D double well, $\beta=5$, dt = 1e-3, exact lattice solve: +4.1% at $x_0=-0.8$, +22.5% at $-0.95$ | Brownian-bridge killing: bias becomes $O(dt)$ (−0.39%, −0.30%) at no extra force calls |
| singular drift at $\partial A$ | hard committor | heavy tails governed by $\kappa=\lvert\nabla q(\partial A)\rvert\sigma\sqrt{dt}/q(x_0)$ | $\kappa=0.009$: Var 346–407, falling to 0.0063 once the 0.5% of paths near $A$ are removed | floor with $\lvert\nabla q(\partial A)\rvert\sigma\sqrt{dt}\lesssim\epsilon\ll q(x_0)$ |
| unstopped hard terminal indicator | fixed horizon with $\mathbf 1_B(X_T)$ | Var $\propto dt^{1/2}$; miss fraction $\propto dt^{1/2}$, with $w=-\infty$ on misses | Var 0.43 to 0.11 and misses 11% to 3.1% over dt = 0.02 to 0.00125 | stop on hitting, or use a smooth reward. Importance sampling stays unbiased |
| dimension | mismatch: targets with $\lVert\nabla^2\log h\rVert_F$ growing in $d$ (e.g. product targets); null defect: an EM backward with $U$ of many coordinates | the laws add over coordinates | not measured | at fixed residual, dt scales as $1/d$ (mismatch) or $1/\sqrt d$ (null defect) [DERIVED]. The null defect depends on $U$, not on $h$: with an EM backward in the rotated well the $d-1$ OU coordinates each add to Var$(dt[f_1(x_N)-f_1(x_0)])$ whatever $h$ is. With the forward-reference ratio there is no $C_{dt}$, and in the rotated well $h$ depends on $\xi$ only, so $\lVert\nabla^2\log h\rVert_F$ does not depend on $d$; the floor is then exactly $d$-independent (reviewer) |

**Table 3.** Corrections the verifiers made to the derivers. Each row names the original claim, the correction, the evidence behind the correction, and how this note uses it.

| original claim (source) | correction (source) | basis | used here |
|---|---|---|---|
| on-policy log-variance is zero only at $u_\epsilon$ (CV deriver) | spurious exit-conditioned zeros exist (CV verifier) | Monte Carlo table plus derivation | Lemma G; certificate |
| reverse-KL best fixed-CV profile is the occupation-weighted projection of $u^*$ (CV deriver) | REFUTED; the projection is exact only for forward KL (CV verifier) | linear-Gaussian counterexample: profile ratio 2.9 at $t=0$ | §3.2, §3.6 |
| in the spectral window $u^*\approx\sigma^2\nabla\log q$ (CV deriver) | it is the soft form with $\epsilon_{\rm eff}$; window entry grows like $\log(\lambda_2/\lambda_1)/\lambda_2$ (CV verifier). This matches the committor deriver's $\nabla q/(q+\alpha)$ | grid at two resolutions | §3.3 |
| metastable parts of the window are PROVED (committor deriver) | DERIVED heuristic; drift-level error up to 44% inside the wells; rate formula needs $q(a)\ll\pi_B$ (committor verifier) | grid | §3.3 |
| stopped owner-form $\log Z$ correct to $O(dt)$; $O(\sqrt{dt})$ monitoring bias left open (committor deriver) | $O(\sqrt{dt})$ from monitoring; bridge killing fixes it (committor verifier) | exact lattice solve of the EM chain | §3.4, §3.5 |
| regression target is $\sigma^2\nabla\log q+O(dt)$ (committor deriver) | target is $\sigma^2\nabla\log q_{dt}$; $O(1)$ error next to $\partial A$ (committor verifier) | lattice | §3.7 |
| fixed annealed $P_B$: "$Q$ not conditioned on anything", "exact transport impossible" (bias deriver) | $Q$ is a Feynman–Kac tilt by the work; a per-start $\log Z$ reaches zero loss (bias verifier) | Monte Carlo at 8 points; fit of the residual against $x_0$ | §3.13 |
| off-policy works with a per-start $\log Z$ (simple deriver) | necessary, not sufficient, at finite dt (simple verifier) | OU training | Lemma D |
| rates from killing need a quasi-stationary start (simple deriver) | an affine fit in $s$ debiases a point start (simple verifier) | grid plus controlled Monte Carlo | §3.8 |
| learning halves the EM floor (simple deriver, OU) | true for OU only; on the double well learning removes 3–9% (simple verifier) | exact grid chain rule | §3.1 |
| weak tilt needs $T$ only beyond fast relaxation (CV deriver) | needs $T\gg1/\lambda_1$; $T\gg1/\sqrt{\lambda_1\lambda_2}$ for alignment (CV verifier) | spectral computation | §3.9 |
| mean-force regression about 10× cheaper than VES, OPES and metadynamics; Jarzynski RMS values (bias deriver) | adaptive mean-force run reproduced worse (0.093 against 0.065 kT at 1.28e5 calls); the metric floor is 0.011 kT; VES, OPES and metadynamics rows not reproduced; Jarzynski RMS optimistic by 1.5–2× (bias verifier) | reruns and the delta method | Table 4 caption |
| dropping the last backward step shifts $\log Z$ by $(d/2)\log 2\pi\sigma^2dt$ (committor deriver) | by $(d/2)(1+\log2\pi\sigma^2dt)$ (committor verifier) | algebra; the deriver's own table matches | Lemma C |

Disagreements that remain open:
1. **PIPS loss direction.** The survey's reading, forward KL by self-normalised importance sampling, comes from garbled equations and is unverified.
2. **REACT value matching versus tied TB.** Survey 2's CONJECTURE that REACT's value-matching loss equals a TB residual with a log-flow that depends on $x_0$, i.e. the tied form of Lemma F, is unverified. The formula was not fully read.
3. **Local losses.** The simple-targets family proposes local detailed-balance losses with a learned $\log h$ as the generalisation lever (CONJECTURE). The committor verifier showed that per-step detailed balance with a tied flow is biased independently of dt. These have not been reconciled.
4. **Tied form and spurious zeros.** Whether the tied form with hard boundary conditions excludes the spurious zeros of Lemma G. With the boundary value fixed at $\epsilon$ on $\partial A$, the exact family $\kappa+(1-\kappa)q$ with $\kappa\ne\epsilon$ is excluded [DERIVED here]. Near-zeros in thin boundary layers have not been analysed.
5. **A consistency check.** The toy catalogue says $\sigma^2\nabla_{x_0}\log Z_\phi=u_\theta(x_0,0)$ "for free". That holds in the forward-reference form only. In the owner form, first subtract $\nabla(\log\tilde\pi-\log\mu_0)$.

### 3.1 Null case (unit test)

- **Claim** [PROVED, 1D]. With $\mu_0=\pi$, $R=\tilde\pi$ and $P_B$ the reversal, $Q$ is the stationary reference path measure, $u^*=0$ and $\log Z^*=\log Z_U$. This is Lemma C with $g\equiv1$, so $h\equiv1$.
- **Finite dt** [PROVED]. $w-\log Z_U=-C_{dt}$ exactly.
  - For OU ($U=kx^2/2$, $a=1-k\,dt$, $N=T/dt$), $\mathbb Ew-\log Z_U=-(k\,dt)^2(1-a^{2N})/(8(1-k\,dt/2))$, and $\mathrm{Var}\,w\to(k\,dt)^2/4$.
  - The EM optimum for OU is not zero. It is $u^*_k=(aq_{k+1}/q_k-a)x/dt$ with $q_N=1/(\beta k)$ and $q_k=a^2q_{k+1}+\sigma^2dt$. It is a contraction confined to the last ~$1/(2k)$ of the horizon. It halves the KL floor to $\approx(k\,dt)^2/16$.
  - For the quartic double well the floor over all drifts is $\approx\mathrm{KL}(\pi\Vert\pi P_{EM}^N)+\tfrac{\sigma^2}{12}dt^2\int\mathbb E\lVert\nabla^3U\rVert_F^2$. Learning removes only 3–9% of the $u=0$ KL (verifier: exact grid chain rule), so a correct trainer barely moves.
- **Pass criteria.**
  - The exact OU kernel gives $\lvert w-\log Z_U\rvert\le10^{-13}$.
  - EM on OU matches the closed forms.
  - EM on the double well matches the variance law (simulated/law → 1).
- **Traps.**
  - A MALA-type backward with an EM forward makes things worse: the bias becomes $-\Theta(T\sqrt{dt})$, e.g. −0.113 nats against −1.3e-5 for EM (OU, $T=5$, dt = 0.01). Importance sampling is then biased too.
  - A Metropolis-adjusted forward kernel paired with the same backward passes the test for every $u$, so it says nothing about the drift.

### 3.2 Fixed-horizon terminal conditioning

- **Claim** [PROVED, cont. and EM, 1D]. $u^*=\sigma^2\nabla\log h(x,t)$, and $\log Z^*(x_0)=\log h(x_0,0)$ in the forward-reference form. This is Lemma B with $D=\mathbb R^d$, $f=0$, $r=g$.
- **No time-independent optimum** [PROVED; verifier sharpening DERIVED].
  - Suppose $\nabla\log h=f(x)$ on $[t_1,T)$. Then $h=e^{\gamma(t)}\phi(x)$, with $L\phi=-c\phi$ and $R\propto\phi$.
  - For bounded $R\ge0$ and confining $U$: $c>0$ forces $\phi$ to be orthogonal to the constants, so it cannot be positive. $c<0$ contradicts boundedness. $c=0$ forces $\phi$ to be constant.
  - So an exactly optimal time-independent drift exists only when there is no conditioning.
  - Measured KL of frozen controls: 10.3 nats ($u=0$), 3.14 ($u^*(x,0)$), 0.75 ($u^*(x,T/2)$), 4.69 ($u^*(x,T-0.1)$). Setup: 1D double well, $\beta=5$, $a=-1.2$, $T=5$, reward sigmoid$((x-0.5)/0.1)$.
- **Long horizon** [PROVED]. $\nabla\log h=O(e^{-\lambda_1\vartheta})$. As $T\to\infty$ the target is the unconditioned reference on any fixed window, not the committor.
- **Hard reward** [PROVED]. With $\mathbf 1_B$, the KL is infinite for every bounded $u$. At finite dt the EM chain has full support, so the TB loss is infinite for every $\theta$. Smooth surrogates converge in total variation.
- **Finite dt** [PROVED].
  - At the continuum $u^*$, the residual mean and variance scale with dt. Same toy: Var 0.0444, 0.0221 and 0.0112 (deriver) and 0.0453, 0.0225 and 0.0113 (verifier) at dt = 4e-3, 2e-3 and 1e-3, with $\log h(a,0)=-3.8662$.
  - The variance comes from the kernel-mismatch law (Table 2).
- **OU oracle** [PROVED]. Soft constraint $g=e^{-(y-y^*)^2/2\gamma^2}$ with $y^*=2.5$, $\gamma=0.3$, $T=1$. $h$ and $u^*$ are in closed form, $\log h(0,0)=-4.45418$, and the variance law gives about $8.6\,dt$. On-policy TB with an affine drift per step reached the discrete-exact optimum to 0.22%.
- **Fixed CV** [DERIVED]. A fixed $s$ is exact at all $t$ for every reward $F\circ s$ if and only if $Ls=a(s)$ and $\lvert\nabla s\rvert^2=D(s)$.
  - *Proof outline.* If (ii) holds, then $L(F\circ s)=(aF'+\beta^{-1}DF'')\circ s$, and uniqueness of the backward problem gives $h=H(s,t)$. Conversely, differentiate at $t=T$ with $F(z)=z$ and $F(z)=z^2/2$.
  - *Examples.* The reaction coordinate of the rotated double well is lumpable. The committor is not.
  - *Prior art.* This is the classical criterion for $s(X_t)$ to be an autonomous diffusion (Dynkin; Rogers–Pitman Markov functions), named by the reviewer; cite it as such.
- **Best profile for a fixed, non-lumpable CV.**
  - For **forward KL** (weight $\rho_t$ = the marginals of $Q$), $g^*(z,t)=\mathbb E_\rho[\nabla s\cdot u^*\mid s=z]/\mathbb E_\rho[\lvert\nabla s\rvert^2\mid s=z]$. The best $s$ maximises the captured energy $\int\!\!\int\rho\,\mathbb E[\nabla s\cdot u^*\mid s]^2/\mathbb E[\lvert\nabla s\rvert^2\mid s]$ [DERIVED, exact].
  - For **reverse KL** the stationary profile is $-\sigma^2\mathbb E_{\rho^u}[\nabla s\cdot\nabla V^u\mid s]/\mathbb E_{\rho^u}[\lvert\nabla s\rvert^2\mid s]$, where $V^u$ is the policy's own cost-to-go. This is not a projection of $u^*$: the deriver's claim is REFUTED by the verifier's linear-Gaussian counterexample. KL is 1.13122 at the optimum against 1.13752 at the claimed point, and the profile at $t=0$ is 0.0215 against 0.0075.
- **Gauge.** The drift fixes $h(\cdot,t)$ only up to a factor $c(t)$; the per-start $\log Z$ pins it.
- **Failure modes.**
  - Rare events at the start: only 2.1% of $u=0$ paths reach $B$ by $T=5$.
  - The drift is singular near $T$.
  - When $T\gg1/\lambda_1$ the signal concentrates near $T$.
- **Cost.**
  - Each path costs $T/dt$ force calls whether or not it reacts. Under $u^*$ the crossing is spread over $[0,T]$, so many calls are spent waiting: 5,000 per path at $T=5$, dt = 1e-3, against about 1,100 in the stopped form.
  - Those two numbers come from different problems: start $-1.2$ with a sigmoid reward at $T=5$, against start $-0.5$ stopped at $x=1$. $T=5$ was the deriver's test horizon, not an optimised one. §3.3 suggests about $4/\lambda_2\approx1.2$ time units for the function readout on the 1D well, about 1,200 calls at dt = 1e-3, and a longer horizon for the control. A comparison at matched start, target and accuracy has not been made.
  - Detached-sample TB needs no Hessian of $U$. The reparameterised reverse-KL gradient needs Hessian-vector products at every step.

### 3.3 Spectral window of the fixed horizon: slow mode and rate

- **Expansion** [PROVED]. $h(\cdot,\vartheta)=\langle g\rangle+e^{-\lambda_1\vartheta}\langle g,\varphi_1\rangle\varphi_1+\rho_\vartheta$, with $\lVert\rho_\vartheta\rVert_\pi\le e^{-\lambda_2\vartheta}\lVert g\rVert_\pi$. In Dirichlet energy the remainder ratio is at most $\frac{\lambda_2}{\lambda_1}e^{-2(\lambda_2-\lambda_1)\vartheta}\lVert g\rVert^2/\langle g,\varphi_1\rangle^2$ for $\vartheta\ge1/(2\lambda_2)$.
- **Function readout** [PROVED].
  - The affine ratio $(h(x)-h(x_A))/(h(x_B)-h(x_A))$ cancels the gauge $c(t)$ and returns affine($\varphi_1$) with error $O(e^{-(\lambda_2-\lambda_1)\vartheta})$.
  - 1D double well, $\beta=5$: max error against $q$ is 0.155 at $\vartheta=0.5$, 0.030 at 1, 0.0026 at 2, and a floor of 0.00206 for $\vartheta\ge5$. The floor is the gap between $\varphi_1$ and affine($q$).
- **Metastable identification** [DERIVED; two-state heuristic, numerically supported].
  - $\varphi_1\approx\gamma(q-\langle q\rangle_\pi)$. The node sits at $q=\langle q\rangle_\pi\approx\pi_b$, not at $q=1/2$.
  - 1D rotated well, $\beta=4$: node at $q=0.183$ with $\pi_b=0.183$. Müller–Brown: 0.0253 against 0.0254.
  - Gap between $\varphi_1$ and affine($q$): 2.06e-3 (1D, $\beta=5$) and 1.8e-3 (Müller–Brown).
- **Control readout** (verifier-corrected).
  - The window form also needs $e^{-(\lambda_2-\lambda_1)\vartheta}\lVert g\rVert\ll\min_x(\langle g\rangle+c\varphi_1)$. Hence $\vartheta\gtrsim\lambda_2^{-1}\log(C\lambda_2/\lambda_1)$, which grows logarithmically with metastability. The Q-weighted relative error of the window profile is 16 (116 on a 4× finer grid) at $\vartheta\lambda_2=3.3$, and 1.2e-3 at 5.3.
  - With $g=\mathbf 1_B$ in the two-state regime: $u^*\approx\sigma^2\nabla\log(\epsilon_{\rm eff}+(1-\epsilon_{\rm eff})\bar q)$, with $\epsilon_{\rm eff}=\langle q\rangle(1-e^{-\lambda_1\vartheta})/(\langle q\rangle(1-e^{-\lambda_1\vartheta})+e^{-\lambda_1\vartheta})\approx\langle q\rangle\lambda_1\vartheta$. This is A1's soft target with an $\epsilon$ that grows with time-to-go.
  - It equals the committor deriver's $\sigma^2\nabla q/(q+\alpha)$ with $\alpha=\pi_B(e^{\lambda_1\vartheta}-1)$. Reason: $\epsilon_{\rm eff}/(1-\epsilon_{\rm eff})=\langle q\rangle(e^{\lambda_1\vartheta}-1)$ (algebra in this note), and $\langle q\rangle\approx\pi_B$.
  - Q-weighted errors 0.088, 0.055, 0.022 and 0.0067 at $\vartheta\lambda_2$ = 3.3, 5.3, 10 and 33 (rotated-well $\xi$ chain, $\beta=4$, CV verifier). Relative error inside the wells, where the drift is small, is up to 44% (1D double well, $\beta=5$, committor verifier).
  - The form $\sigma^2\nabla\log q$ is wrong here: $A$ is not absorbing, and its error diverges under grid refinement.
- **Rates.**
  - [PROVED] $Z(T)=h(x_0,\vartheta=T)$. $Z'(T)=-\lambda_1e^{-\lambda_1T}\varphi_1(x_0)\langle\varphi_1,g\rangle+O(e^{-\lambda_2T})$, and $\lambda_1=-Z''/Z'$.
  - [DERIVED] In the two-state limit, $Z'\approx\lambda_1(\pi_B-q(x_0))e^{-\lambda_1T}$. This equals $k_{\rm eig}e^{-\lambda_1T}$, with $k_{\rm eig}=\pi_B\lambda_1$, only when $q(x_0)\approx0$ (verifier correction).
  - Measured: $Z'/(k_{\rm eig}e^{-\lambda_1T})=1.0045$ for $T\ge2$. Here $k_{\rm eig}=0.0055526$ is 1.3% above the TPT rate constant $k_{A\to B}=0.0054819$. The difference is a matter of definition (point sets).
- **Cost.** For the function readout the horizon is a few multiples of $1/\lambda_2$; for Müller–Brown, $4/\lambda_2\approx0.1$ time units. For the control readout it grows like $\lambda_2^{-1}\log(C\lambda_2/\lambda_1)$, which is linear in $\beta\Delta U$ in units of $1/\lambda_2$ but still far below $1/\lambda_1$ (verifier). This is the owner's literal setup: no stopping, no absorbing sets.
- **No trained evidence.** Everything in this section and in §3.9 comes from grid and spectral computations. No drift was trained for the fixed horizon with a Langevin $P_B$, reading out $\varphi_1$ or a rate.

### 3.4 Stopped committor

- **Claim** [PROVED, cont.]. Apply Lemma B on $D$ with $r=\mathbf 1_B$, localised at $\tau_n=\inf\{q\le1/n\}$. On B-ending paths $\log q(X_\tau)=0$. So:
  - $u^*=\sigma^2\nabla\log q$, independent of time and of $x_0$;
  - $\log Z^*(x_0)=\log q(x_0)$, or $\log q(x_0)-\beta U(x_0)$ in the owner form;
  - $\mathrm{KL}(Q\Vert P)=-\log q(x_0)=\frac\beta4\mathbb E_Q\int\lvert u^*\rvert^2$.
  - In 1D, $q(x)=\int_{x_A}^xe^{\beta U}/\int_{x_A}^{x_B}e^{\beta U}$.
- **Capped horizon** [PROVED]. Convergence is $\lvert\log h_T-\log q\rvert\le Ce^{-\lambda_D\vartheta}/q$, with Dirichlet eigenvalue $\lambda_D=2.988$ and prefactor about 34 (1D double well, $\beta=5$).
- **Evidence.** 1D double well, $\beta=5$, $x_0=-0.5$, dt = 1e-3, one seed.
  - $10^5$ reference paths, of which 2,506 hit $B$ first, against $10^5$ paths under $u^*$.
  - Transit-time quartiles: 0.592/1.000/1.672 (reference) against 0.586/0.988/1.671 (controlled). Kolmogorov–Smirnov distance 0.018, against a 95% noise scale of 0.0275.
  - Importance-sampling $q$: 0.025469, against 0.025518 in closed form.
- **Near $\partial A$** [DERIVED; verifier adds a factor]. $u^*\approx\sigma^2\nabla d/d$, a repulsion like that of a 3D Bessel process. Under $Q$, the probability that a path comes within $\epsilon$ of $\partial A$ is $\approx(1-q(x_0))\lvert\nabla q(\partial A)\rvert\epsilon/q(x_0)$. EM cannot resolve this within $\sigma\sqrt{dt}$, which gives the heavy tails of Table 2.
- **Finite dt** [verifier; PROVED in 1D on the lattice].
  - The optimum is $q_{dt}\approx(Iq+\delta_b)/(I+2\delta_b)$ with $\delta_b=0.5826\sigma\sqrt{dt}$ and $I=\int e^{\beta U}$.
  - Brownian-bridge killing ends each step with probability $e^{-2d_kd_{k+1}/(\sigma^2dt)}$ for each boundary and assigns the reward of the boundary crossed. It is drift-free, so it cancels in the Girsanov ratio, and it leaves only $O(dt)$ bias. This was shown in 1D with point boundaries, using the half-space formula. It over-corrects below $\epsilon^*$ (§3.5) and is untested on curved sets and in $d>1$ (rung 10).
- **Spurious zeros.** See Lemma G.
- **Failure modes.**
  - Rare-event start: under $u=0$ only a fraction $q(x_0)$ of paths reaches $B$ (2.5% from −0.5, 0.29% from −0.8). With hard $\mathbf 1_B$ every other path has an infinite residual.
  - A start on $\partial A$ is ill-posed.
  - $q$ is identified only on the region the start law covers.
- **Cost.** Each reactive path costs $\mathbb E_Q[\tau]/dt\approx1{,}074$ force calls from $x_0=-0.5$, with no waiting. Because $u^*$ does not depend on $x_0$, starts can be put where $q(x_0)$ is not small.
- **Interface splitting** [DERIVED; standard forward-flux and transition-interface identities; untested].
  - Write $q$ as a product of stage probabilities between nested interfaces.
  - Train one network with a curriculum of starts taken from each stage's hit points.
  - $k_{A\to B}$ = the flux of first crossings of the first interface times $\mathbb E[q]$ on those crossings.

### 3.5 Floor-regularised, tied stopped TB (trainable form of §3.4)

- **Claim** [PROVED, cont.].
  - $h_\epsilon=\epsilon+(1-\epsilon)q$ is $L$-harmonic, with values $\epsilon$ on $\partial A$ and 1 on $\partial B$. Since $h_\epsilon\ge\epsilon$, no localisation is needed.
  - $u^*=\sigma^2(1-\epsilon)\nabla q/(\epsilon+(1-\epsilon)q)$. Its value at $\partial A$ is $\sigma^2(1-\epsilon)\lvert\nabla q(\partial A)\rvert/\epsilon$, which is not a global bound (verifier).
  - $q=(h_\epsilon-\epsilon)/(1-\epsilon)$ is exact.
  - Under $Q_\epsilon$, a fraction $\epsilon(1-q)/h_\epsilon$ of paths end in $A$.
- **Tie.** $\log Z_\theta(x_0):=\log h_\theta(x_0)$ with hard boundary conditions. In 1D, $q_\theta=\frac{x+1}{2}e^{g_\theta(x)-g_\theta(1)}$ and $h_\theta=q_\theta+\epsilon(1-q_\theta)$.
  - By Lemma F, $\delta=-\int(Lh_\theta/h_\theta)dt$ along each path. So $\delta\equiv0$ if and only if $h_\theta=h_\epsilon$ on the visited set.
  - The tie removes the multiplicative gauge, which matters because $\epsilon$ has an absolute scale.
  - With the boundary value fixed at $\epsilon$ on $\partial A$, the exact spurious family of Lemma G is excluded [DERIVED here; near-zeros not analysed].
- **Choice of $\epsilon$** [DERIVED; numerics agree]. Use $\lvert\nabla q(\partial A)\rvert\sigma\sqrt{dt}\lesssim\epsilon\ll q(x_0)$. On the symmetric double well, $\epsilon^*\approx2.3\times10^{-4}$ at dt = 1e-3. At $\epsilon=10^{-5}$ the residual variance returns to 5.7–23. At $\epsilon=10^{-3}$ from $x_0=-0.8$, 23% of paths end in $A$.
- **Finite dt** (verifier).
  - The readout returns $q_{dt}$ (+4.1% at $x_0=-0.8$, dt = 1e-3).
  - The fraction of paths ending in $A$ is biased low: 22.4% against 25.4% continuum at $x_0=-0.8$, $\epsilon=10^{-3}$. The deriver reported this gap but did not explain it; it is not noise.
  - Bridge killing gives 25.7%. Below $\epsilon^*$ the bridge over-corrects (4.8% against 3.3%).
- **Training** (single-seed observations, symmetric double well, $\beta=5$, dt = 2e-3, 256 paths per iteration, 300 iterations).
  - Uniform starts on (−0.95, 0.95), $\epsilon=10^{-2}$: max $q$ error 0.0087 at 2.17e7 force calls. Readout at $x=-0.8$: 0.00269 against 0.00293.
  - Point start at −0.5, $\epsilon=10^{-3}$: 0.055 at 4.29e7, still improving.
  - At initialisation, 72–80% of paths already reached $B$ because of the linear boundary-condition construction.
- **Prior art.** REACT's $q_\xi=\xi+(1-2\xi)q$ is the same device with the floor on both sides. It is the h-transform of the reward $\xi\mathbf 1_A+(1-\xi)\mathbf 1_B$ [DERIVED in the survey].
- **Failure modes.**
  - $\epsilon\ge q(x_0)$ spends most samples on A-returning paths.
  - In $d>1$ the hard boundary conditions need distance or indicator functions for $A$ and $B$, which is close to needing a CV already.
  - $\lvert\nabla q(\partial A)\rvert$ is unknown a priori. Choosing $\epsilon$ adaptively from the observed A-ending fraction is open.

### 3.6 Bottleneck CV in the stopped committor

- **Representability** [PROVED].
  - For $k=1$, every field $g(s)\nabla s$ is the gradient $\nabla G(s)$. Then $\sigma^2\nabla\log f(s)=u^*$ iff $f\circ s=Ch_\epsilon$ iff $q$ factors through $s$.
  - For $k\ge2$, $\sum_ag_a(s)\nabla s_a$ is a gradient only if $g$ is curl-free in $s$-space. So parameterise $u_\theta=\sigma^2\nabla_x\log f_\phi(s_\theta(x))$ with a scalar $f_\phi$.
- **Rotated double well** [PROVED]. Take $U=V(e\cdot x)+\tfrac\kappa2\lvert P_\perp x\rvert^2$ with half-space sets. Since $\nabla U\cdot e=V'(\xi)$, $L[q_1(e\cdot x)]=\beta^{-1}q_1''-V'q_1'=0$. Uniqueness then gives $q=q_1(e\cdot x)$.
  - Grid check: sup error 8.1e-4 and 4.1e-4 at $h$ = 0.025 and 0.0125, i.e. $O(h)$ from the oblique staircase.
- **Evidence.** Rotated well $V=(\xi^2-1)^2+0.2\xi$, $\kappa=4$, at 30°, $\beta=4$, $x_0=0$.
  - Fixed, untrained controls at $\epsilon=10^{-3}$: only the exact direction gives a residual variance proportional to dt (2.94e-3, then 7.46e-4). At 10° misalignment the variance is 0.084 and does not depend on dt. A profile computed at the wrong $\beta$ gives 2.5.
  - Training: linear CV with an MLP profile, log-variance loss, dt = 2e-3, $\epsilon=10^{-2}$, two runs. Final angles 29.70° and 29.64° against the true 30°. The angle error fell below 2° after 4.96e6 and 3.92e6 force calls. At the end (3.1–3.3e7 calls) the profile was still about 10% off.
- **Gauge and identifiability.**
  - $s$ is identified up to a monotone reparameterisation; make $f$ monotone.
  - The A-exits are what identify the profile (Lemma G).
  - $s$ is unconstrained off the visited set.
- **When $q$ does not factor through the CV class** (e.g. a linear $s$ on Müller–Brown). Reverse KL selects a CV that depends on the policy (§3.2). Forward KL selects the captured-energy maximiser. Which CV joint training actually reaches is open. For reference, of the three linear CVs scored ($x$, $y$, $x-y$), the best is $x-y$, with spectral score 0.80. No optimisation over linear directions was done.

### 3.7 Forward KL: drift regression on reactive paths

- **Claim** [PROVED].
  - $\mathrm{KL}(Q\Vert P^u)=\frac\beta4\mathbb E_Q\int\lvert u-u^*\rvert^2$ by Lemma B. It is finite even for a hard $\mathbf 1_B$.
  - [EM] $-\log p_\theta=\lvert\Delta-(b+u)dt\rvert^2/(2\sigma^2dt)+{\rm const}$, so the minimiser is $\mathbb E_Q[\Delta/dt-b\mid x]$.
- **Correction** (verifier). For stopped problems this is $\sigma^2\nabla\log q_{dt}+O(dt)$, not $\sigma^2\nabla\log q+O(dt)$.
  - The bias relative to the continuum is $O(\sqrt{dt}/(Iq(x)))$: 2.4% at $x=-0.725$ for dt = 1e-3, under 0.3% where $q\gtrsim0.1$.
  - Next to $\partial A$ it is $O(1)$: in the bin at −0.975, the exact EM-conditioned drift is 9.43 against 18.76 in the continuum.
  - This explains the deriver's 3-standard-error bin at −0.925 (exact 4.90; regression 4.64 ± 0.42).
- **Reactive MD segments** [DERIVED]. The forward drift is $b+\sigma^2\nabla\log q$, and the backward drift is $b+\sigma^2\nabla\log(1-q)$.
- **Transport analogue** [DERIVED]. Forward KL from target samples is denoising score matching and needs no energy calls. It agrees with the reverse-KL optimum at finite dt only when the conditionals of $Q$ are Gaussian.
- **Cost.** The fit itself costs no force calls. Each sample costs about $1/q(x_0)$ reference paths, or about $1/k_{A\to B}$ of MD. Replay with importance weights inherits weight degeneracy (effective sample size 2.2% at $u=0$ in §3.2). This is where "data is instrumental" fits.

### 3.8 Rates: first passage by $T$, and killing at rate $s$

- **Hit-by-$T$** [PROVED, 1D].
  - $H(x,\vartheta)=P_x(\tau_B\le\vartheta)$ solves $\partial_\vartheta H=LH$ with $H=1$ on $\partial B$. Then $u^*=\sigma^2\nabla\log H(x,T-t)$, a function of time-to-go only, and $\log Z^*=\log F_{x_0}(T)$.
  - Spectral expansion: $-\log(1-F)=\lambda_1T-\log(c_1\varphi_1(x_0))+O(e^{-(\lambda_2-\lambda_1)T})$. Read $\lambda_1$ as the slope between two horizons $T_1\gg1/\lambda_2$, both far below $1/\lambda_1$. MFPT $\approx e^{-\text{intercept}}/\text{slope}$.
  - 1D double well, $\beta=3$, $x_0=-1$: $\lambda_1=0.0388447$ and MFPT = 26.640. Two-horizon errors are −10.9% for $T$ = 1 to 2, −0.20% for 2 to 5, and 0.000% for 5 to 10. The slope of $F$ itself between 2 and 5 is 10% low.
  - From the quasi-stationary distribution the relation is exact at every $T$.
  - OU closed form: $\mathrm{erfc}(x/\sqrt{2v})$ with $v=(e^{2k\vartheta}-1)/(\beta k)$.
  - Bridge correction for OU hitting 0:
    - $k=\beta=1$, $x_0=3$, $T=0.3$ ($H=9.374\times10^{-4}$), under $u^*$ at dt = 1e-3: the −11.5% bias drops to +0.5% (toy solver).
    - $k=\beta=1$, $x_0=1.5$, $T=0.5$ (exact 0.2525, SE 0.0014; simple deriver): grid monitoring gives 0.2093, 0.2285 and 0.2400 at dt = 0.02, 0.005 and 0.00125, an $O(\sqrt{dt})$ bias. The bridge-corrected values are 0.2584, 0.2544 and 0.2531, an $O(dt)$ bias.
- **Killing at rate $s$** [PROVED, 1D; the verifier upgraded this from DERIVED with a controlled simulation].
  - $(L-s)h_s=0$ with $h_s=1$ on $\partial B$. Then $u^*=\sigma^2\nabla\log h_s$ is time-homogeneous, and $\log Z^*=\log\mathbb E_{x_0}e^{-s\tau_B}$.
  - Controlled check at $s=0.5$: importance-sampling $\log Z$ is −3.0561 and −3.0584 at dt = 0.01 and 0.005, against −3.0601 on the grid. Residual variance is 0.064 and 0.033, i.e. proportional to dt.
  - From the quasi-stationary distribution, $Z=\lambda_1/(\lambda_1+s)$ exactly.
  - From a point start (verifier correction): $s/(1-Z(s))$ is affine in $s$ for $s\ll\lambda_2$, so $\lambda_1$ = intercept/slope. Errors are −0.1% to −1.8%, against −7.5% to −79% for the naive $sZ/(1-Z)$.
  - Cost per conditioned path scales like $1/(\lambda_1+s)$, so choose $s\gtrsim\lambda_1$.
- **Survival conditioning.** It is rare only when $T\gtrsim1/\lambda_1$, so it is unsuitable for metastable rates.
- **General Feynman–Kac.** For $R=g(X_T)e^{-\int V}$, $u^*=\sigma^2\nabla\log h$ with $(\partial_t+L-V)h=0$ [DERIVED].

### 3.9 Ground-state tilts, killing, and the flat-bias question

- **Ground state** [DERIVED; the spectral identities were checked numerically, but no drift was trained].
  - $H_\lambda=L+\lambda f$ is self-adjoint in $L^2(\pi)$ and has a positive, simple ground state $\varphi_\lambda$. So $u^*\to\sigma^2\nabla\log\varphi_\lambda$ for $T-t\gg1/(\Lambda_0-\Lambda_1)$.
  - Bias reading: $V_\lambda=-(2/\beta)\log\varphi_\lambda$. The biased reversible law is $\propto\pi\varphi_\lambda^2$, which is also the mid-horizon marginal of $Q$.
  - $\log Z(T)=\Lambda_0T+\log\langle\varphi_\lambda\rangle+\log\langle r,\varphi_\lambda\rangle+o(1)$.
  - Prior art: the generalised Doob transform and driven process of Chetrite & Touchette 2015 (named by the reviewer; not fetched).
- **Weak tilt** [DERIVED; verifier correction on the horizon].
  - $\psi=(-L)^{-1}(f-\langle f\rangle)=\sum_k\langle f,\varphi_k\rangle\varphi_k/\lambda_k$ is dominated by $\varphi_1$. The share of $\mathcal E(\psi)$ carried by $\varphi_1$ was 0.99925 for $f=x$ on the rotated well.
  - The tilted gap is $\approx\lambda_1$. So the ground state needs $T\gg1/\lambda_1$, and finite-horizon alignment needs $\vartheta\gg1/\sqrt{\lambda_1\lambda_2}$. At $\vartheta=1$ the orthogonal share was 0.171 against 1.5e-3 for the terminal reward of §3.3, which therefore dominates this route per force call.
- **Killing outside $B$** [DERIVED; two-state heuristic, numerically supported]. $\varphi_0^D\approx C(1-q)$, with residual 5.4e-4 at $\beta=4$, and $\lambda_0^D/\lambda_1\approx\langle q\rangle$ (0.177 against 0.183).
- **Nodal min–max** [PROVED, 1D]. $\lambda_1\le\max(\lambda_0^{D_1},\lambda_0^{D_2})$ for any partition, with equality at the nodal partition of $\varphi_1$. The minimising cut was −0.184 against a node at −0.18405. This gives a variational principle for the dividing surface from two survival problems; whether it is practical is open.
- **Flat bias by ergodic reverse path KL** [the condition PROVED; impossibility for every $F$-free cost DERIVED (argued: a conditional variance is nonlinear in the conditional law, a conditional mean is linear), not proved].
  - With the reference tilted by $e^{-\int c}$, the optimum is the ground-state drift, with stationary law $\varphi_0^2\pi$.
  - A flat marginal ($V^*=-F$) requires $c=c_{-F}+$ const. Restricted to $V(s)$ with $s=x_1$, this becomes $\bar c(s)=-\tfrac\beta4F'^2+\tfrac12F''=\mathbb E[c_W\mid s]-\tfrac\beta4\mathrm{Var}(\partial_1U\mid s)$, where $c_W=-\tfrac\beta4(\partial_1U)^2+\tfrac12\partial_{11}U$. This uses $F''=\mathbb E[\partial_{11}U\mid s]-\beta\mathrm{Var}(\partial_1U\mid s)$.
  - The variance term needs $F'$. With the $F$-free cost $c_W$, the optimum has RMS $\beta(V^*+F)=0.655$ kT and marginal max/min = 6.9, on the periodic 2D toy, by quadrature and an independent BFGS check.
  - It works only for $s=x$. A cost built from two replicas of $x_\perp$ at the same $s$ is a CONJECTURE.
  - Scope: pointwise costs, the $V(s)$ family, $s=x_1$ and the ergodic limit. $c_W$ is the only $F$-free cost tested ($c=0$ gives $V=0$). Self-consistent outer iterations, where the cost or a terminal weight uses the current estimate of $F$, are not covered.
  - Prior art: Guo, Hartmann, Berger & Nüske (2026) obtain $-(2/\beta)\log\varphi$ from the same ergodic problem, solved as an eigenproblem on equilibrium samples rather than trained on trajectories.
- **Factor-of-two trap** (simple verifier). The stationary drift that makes $\pi_1$ invariant is $\tfrac{\sigma^2}{2}\nabla\log(\pi_1/\pi)=-\nabla(U_1-U)$. Every h-transform drift, including the transport drift at $t=T$, is $\sigma^2\nabla\log(\cdot)$. So reading a transport drift as a bias potential is off by 2. The Doob committor bias is $-(2/\beta)\log q$; Kang–Trizio–Parrinello's $-(1/\beta)\log\lvert\nabla q\rvert^2$ targets a different density.

### 3.10 Mean-force regression: projected path cross-entropy

- **Claim** [PROVED, given conditional equilibrium].
  - With $s=x_1$: $ds=\eta\,dt+\sigma dW^1$ with $\eta=-\partial_1U-V'(s)$. The model is $M_\theta$: $ds=b_\theta(s)dt+\sigma dW$ with $b_\theta=-F_\theta'-V'$.
  - The cross-entropy is $H(P^s,M_b)=\tfrac\beta4\mathbb E\int(\lvert b-\eta\rvert^2-\eta^2)dt$, because the martingale term has zero mean. It is minimised at $b^*=\mathbb E[\eta\mid s]$.
  - If $x_\perp$ given $s$ is distributed as $\pi(x_\perp\mid s)$, which holds at stationarity under any bias $V(s)$, then $\mathbb E[-\partial_1U\mid s]=-F'(s)$. So $F_\theta'^*=F'$ for every data-generating $V$.
- **Rao–Blackwellisation.** Replace the increment target $\Delta s/dt+V'$ by the force $-\partial_1U$, which the integrator already computes. The minimiser is unchanged; the noise per sample drops by $\sigma^2/dt$. On the same data the force target was 4.7–8.8× more accurate (deriver) and 4.7–7.6× (verifier).
- **General CV** [PROVED to 5e-13]. $b^*=\mathbb E[Ls\mid s]$, $D^*=\mathbb E[\lvert\nabla s\rvert^2\mid s]$, and $F'=(\beta^{-1}D^{*\prime}-b^*)/D^*$. If $\lvert\nabla s\rvert$ is not constant on level sets, the path KL is $+\infty$ (singular), and only the regression form applies.
- **In the owner's frame.** $u_\theta=-\nabla V_\theta(s)=F_\theta'(s)\nabla s$ cancels the mean CV force. The fixed Langevin is the data generator; there is no $P_B$ and no TB. The tied path-likelihood form needs $V$ detached, otherwise $b_\theta\equiv0$ (verifier).
- **Verifier caveats.**
  - Whenever there is net $s$-flux (filling, spreading, bias updates), the data's conditional law departs from $\pi(x_\perp\mid s)$. Transients therefore bias the fit, by an amount set by the ratio of the $x_\perp$ relaxation time to the $s$ transport time.
  - The reverse-direction side claim has no derivation and is a CONJECTURE.
- **Prior art.** This is ABF / thermodynamic-integration mean-force regression, with "reinforced dynamics" (Zhang, Wang & E 2018, named by the verifier) as the neural version. Novelty would have to come from amortisation.
- **Extended-Lagrangian variant** [DERIVED, standard, unchecked]. Couple $\lambda$ through $\tfrac\kappa2(s(x)-\lambda)^2$. This makes the path reading exact for nonlinear or learned CVs, and recovers $F$ by the CZAR correction.
- **Cost.** See Tables 4 and 4b.

### 3.11 VES: forward KL between stationary laws (non-path)

- **Claim** [PROVED].
  - $\Omega[V]=\beta^{-1}\log\mathbb E_\pi e^{-\beta V(s)}+\mathbb E_{p_{\rm tgt}}V=\beta^{-1}[\mathrm{KL}(\pi_{\rm tgt}\Vert\pi_V)-\mathrm{KL}(\pi_{\rm tgt}\Vert\pi)]$, where $\pi_{\rm tgt}=\pi\,p_{\rm tgt}(s)/p(s)$.
  - $\delta\Omega/\delta V=p_{\rm tgt}-p_V$, so $V^*=-F-\beta^{-1}\log p_{\rm tgt}$: $-F$ for a flat target, $-(1-1/\gamma)F$ for well-tempered. It is convex for linear $V_\theta$.
  - Verifier quadrature: RMS 3.2e-13 kT.
- **Caveat** (verifier). The self-consistent well-tempered iteration contracts by $-1/(\gamma-1)$. It converges undamped only for $\gamma>2$, and diverges at $\gamma=1.5$ unless damped.
- **Not a path KL.** A path lift would need samples of $Q$, which contains $F$.
- **Cost.** One force call per walker step, using positions only (Table 4).

### 3.12 Transport with a fixed reference that does not change in time (time-reversal score; $\Delta F$)

- **Claim** [PROVED, 1D].
  - Take reference potential $U$ (e.g. OU) and target $R$. Detailed balance gives $\rho_s/\pi=(Z_U/Z_R)h(\cdot,T-s)$ with $g=R/\tilde\pi$. So $u^*=\sigma^2\nabla\log(\rho_{T-t}/\pi)=2\nabla U+\sigma^2\nabla\log\rho_{T-t}$, which is a case of Lemma B.
  - From $\rho_T$, $\log Z^*=\log Z_R$.
  - From $\pi$, the TB value is $\log Z_R-\mathrm{KL}(\pi\Vert\rho_T)$, with $\mathrm{KL}\lesssim\tfrac12e^{-2\lambda T}\chi^2(R/Z_R\Vert\pi)$.
  - Importance sampling is exact for every $u$ and dt.
  - Free energy: reference $U_0$, $R=\tilde\pi_1$, so $g=e^{-\beta\Delta U}$ and $-\beta\Delta F=\log\frac1M\sum e^{w_i}-\log Z_0$. At $u=0$ this reduces to free-energy perturbation (Zwanzig).
  - Evidence: OU to $\mathcal N(2,0.25)$ at $T=1$ from $\pi$: $\mathbb Ew-\log Z_R=-0.3045$ against the predicted −0.3042. Asymmetric double well target: importance-sampling $\Delta F$ within 2e-3, against a free-energy-perturbation variance of 208.6.
- **$P_B$ reversible with respect to the target.** $Q$ is the stationary $U_1$ path measure. Then $u^*=-\nabla(U_1-U_0)$ returns its input, and the TB $\log Z$ is the Gibbs–Bogoliubov bound $-\beta\mathbb E_{\pi_0}\Delta U$ [PROVED]. This is useless for $\Delta F$, but it is the bias-potential optimum when $U_1$ is known.
- **Cost.** With a cheap OU reference as $u_{\rm phys}$, one trajectory costs a single target-energy call and no target forces under detached TB. All structure of the target then sits in $u_\theta$, which is the generalisation question.

### 3.13 Annealed fixed backward (verifier-corrected)

- **Claim** [PROVED on the 1D Gaussian pair].
  - With $P_B$ the reverse-protocol Langevin, $Q=P^{\rm ann}_{\rm ref}\,e^{-\beta(W-\Delta F)}$, where $W=\int\partial_tU_t(X_t)dt$. This is the forward annealed reference tilted by its normalised Jarzynski factor, i.e. Feynman–Kac conditioned on the work. The deriver's "not conditioned on anything" is wrong.
  - $u^*=\sigma^2\nabla\log h$ with $h(x,t)=\mathbb E_{\rm ref}[e^{-\beta\int_t^T\partial_rU_rdr}\mid X_t=x]$. Equivalently $u^*=\sigma^2\nabla\log(\rho^R_t/\pi_t)$ with $\rho^R_t=\pi_th_tZ_t/Z_1$.
- **Scalar $\log Z$.** It converges to $-\beta\Delta F-\mathrm{KL}(\pi_0\Vert\rho^R_0)$. That KL is 3.196, 0.989 and 0.197 nats at $T$ = 0.5, 2 and 8. It is fixed by the reference, and learning the drift cannot reduce it.
- **Per-start $\log Z$** (verifier).
  - At $u^*$ the residual depends on $x_0$ alone: $\mathrm{Var}(\text{res}-\log Z^*(x_0))=3\times10^{-4}$ out of 13.2. So $\log Z_\theta(x_0)$ reaches zero loss with $\log Z^*(x_0)=\log h(x_0,0)$, and $\mathbb E_{\pi_0}e^{\log Z^*}=Z_1/Z_0$.
  - Then $-\beta\Delta F=\log(Z_1/Z_0)$ comes from $\pi_0$ draws alone: −0.689 nats against the exact −0.693, where a scalar $\log Z$ gives −3.889 nats. At $\beta=2$ that is $\Delta F=+0.347$.
  - Resampling $x_0$ with weights $e^{\log Z^*}$ and running $u^*$ gives exact $\pi_1$ samples: mean 1.503 and variance 0.1255, against 1.5 and 0.125.
  - The weight variance does not vanish, but it now sits at $t=0$ in a learned function.
- **Jarzynski limit.** At $u=0$, the residual is $-\beta W$.

### 3.14 Shared escort (learned backward kernel)

- **Claim** [PROVED on the 2D Gaussian pair].
  - With forward marginals $\pi_t$ the Langevin flux vanishes, so $\partial_t\pi_t=-\nabla\cdot(\pi_tu)$. The reversal of $P_F$ has drift $-\nabla U_t-u$, which is $P_B$'s drift, so $P_F=Q$ and $\log Z=-\beta\Delta F$ exactly.
  - Zero-loss set: every $u$ with $\nabla\cdot(\pi_tu)=\beta\pi_t(\partial_tU_t-\langle\partial_tU_t\rangle)$. Divergence-free fields with respect to $\pi_t$ are a gauge.
  - A full-matrix escort learned a deviation that was 97% pure gauge. The gradient parameterisation is unique: median relative error 0.40 for the full matrix against 0.053 for the gradient form.
- **Dirichlet escort functional** (verifier). $J_t[\phi]=\mathbb E_{\pi_t}[\tfrac12\lvert\nabla\phi\rvert^2+\beta(\partial_tU_t-\langle\partial_tU_t\rangle)\phi]$ recovers the gradient escort to 0.2–0.65% from 4e6 samples of $\pi_t$. It is non-path, local, and free of the gauge.
- **CV-only escort.** It is exact iff the conditional law is advected by the escort; this is the verifier's generalisation of the deriver's condition. Measured floor 0.45 nats.
- **Cost.** Training brought the mean residual within 0.005 nats of truth at 1.0e7 force calls. Plain Jarzynski has a delta-method RMS error of 0.023, 0.011 and 0.0056 nats at 6.5e5, 2.6e6 and 1.0e7 calls (verifier; the deriver's values were optimistic). Training an escort pays off only under amortisation, strongly dissipative protocols, or when many $\pi_1$ samples are needed.
- **Prior art.** CMCD, NETS, FEAT, and Vaikuntanathan & Jarzynski 2008. This leaves the owner's non-learned-backward frame.
- **The lever that stays inside that frame** is sequential importance resampling along the protocol under the fixed $P_B$ (CRAFT- and SNF-type; §3.18). It was not run here.

### 3.15 Non-path committor comparators

- **Dirichlet principle** [PROVED]. $q=\arg\min\int_D\lvert\nabla f\rvert^2e^{-\beta U}$ with $f=0$ on $A$ and $1$ on $B$. The first variation gives $\beta\pi Lq=0$, and strict convexity gives uniqueness. $\nu_{AB}=I[q]/(\beta Z_\pi)$, and $k_{A\to B}=\nu_{AB}/\rho_A$ equals 1/MFPT exactly in 1D.
- **Weakness.** The integrand sits at the barrier, where $\pi$ is about $e^{-\beta\Delta U}$, so the rare-event problem moves into the sampling.
- **Backward-Kolmogorov residual.** It needs a Laplacian. The tied TB of §3.5 is its trajectory-integrated, $1/h$-weighted form, without one.
- **Restricted Rayleigh quotient** $\min\mathcal E(F\circ s)/\mathrm{Var}(F\circ s)$ as a CV loss. It costs no force calls beyond the samples (CV verifier, missed formulation).
- **Semigroup committor losses** (Strahan et al. 2023; Mitchell & Rotskoff 2024; listed in survey 1, not fetched). They are REACT's comparator, so they enter rung 7.
- **Spectral CV baselines** (TICA, VAMP) on unbiased trajectories enter rung 8.
- These are the baselines the path formulations must beat at equal force calls. Each needs a stated sample generator, and its force calls count: producing samples that cover the barrier is the whole cost of these methods.

### 3.16 What "the drift converged to a CV" should mean

- **Spectral score** $S(s)=\lambda_1/\lambda_1(s)\in(0,1]$ [PROVED]. Here $\lambda_1(s)$ is the first non-zero eigenvalue of the effective 1D generator $(\beta p)^{-1}(pDF')'$. $S=1$ iff $\varphi_1=F\circ s$, and $S$ is invariant under injective reparameterisation.
  - Rotated well: 10° off gives 0.947, 20° gives 0.796, the orthogonal direction gives 0.0093.
  - Müller–Brown: $x-y$ 0.797 (verifier, binned: 0.806), $y$ 0.116, $x$ 0.033.
- **Committor energy** $E_q(s)=\beta^{-1}/\int dz/(Dp)\ge\mathcal E(q)$, with equality iff $q=F\circ s$.
- **Projection defect** of an unstructured drift onto $g(s,t)\nabla s$, scored against the floor the exact optimum reaches at the same dt and grid (about 3e-5 here), not against zero.
- **Proposed acceptance rule.** On measures not used in training (target marginals, held-out starts):
  - (a) the defect is at the floor;
  - (b) $S\ge1-\delta$, and the Dirichlet ratio $\mathcal E(q_\theta-q)/\mathcal E(q)\le\delta$;
  - (c) the drift readout and the normaliser readout agree, and the B-exit fraction matches (verifier's addition, from Lemma G).
  - Scores weighted by the policy's own occupation are circular.

### 3.17 Measured cost per force call

**Table 4.** Measured cost to a stated accuracy. Rows are on different toys and are not comparable across toys; compare within a toy only. Force calls are walkers × steps, or paths × EM steps.

| formulation | toy | accuracy reached | force calls | seeds | note |
|---|---|---|---|---|---|
| tied stopped TB, spread starts (§3.5) | 1D symmetric double well, $\beta=5$ | max $\lvert\hat q-q\rvert$ = 0.0087 | 2.17e7 | 1 | still improving |
| tied stopped TB, point start −0.5 (§3.5) | same | 0.055 | 4.29e7 | 1 | region between $\partial A$ and $x_0$ poorly identified |
| linear-CV bottleneck (§3.6) | rotated double well, 2D, $\beta=4$ | angle error < 2° | 3.9e6 / 5.0e6 | 2 runs | profile still about 10% off at 3.1–3.3e7 |
| Fourier mean-force regression (§3.10) | periodic 2D toy, $\beta=2$ | $\beta$ RMS $F$ error 0.065 ± 0.010 kT (verifier: 0.093 ± 0.011) | 1.28e5 | 5 | 0.032 kT at 1.28e6 (verifier: 0.020); all budgets and the other bias methods in Table 4b |
| shared-escort training (§3.14) | 2D Gaussian pair | mean residual within 0.005 nats of $-\beta\Delta F$ | 1.0e7 | 1 | Jarzynski at $u=0$: 0.011 nats RMS at 2.6e6 (delta method) |
| literature: TPS-DPS against PIPS | alanine dipeptide | path sampling | 1.6e7 against 2.4e8 training energy evaluations | – | from TPS-DPS's own table |
| literature: Doob's Lagrangian against shooting | Müller–Brown | 1,000 paths | 1.28e6 against 1.03e9 (fixed-length) and 3.53e6 (variable-length) | – | the only force-call comparison against shooting found (survey 2's wording) |

**Table 4b.** Free-energy error of the bias methods against training force calls on the periodic 2D toy ($\beta=2$, $s=x_1$, 32 walkers, EM dt = 0.005). Each cell is the $\beta$-RMS error of $\hat F$ over 64 bins after removing the mean (kT), as a 5-seed mean ± SE, from the deriver's run (`energy_sampling/SCRATCH/nsde_design_nov24/derive_bias/cost_compare_5seeds.txt`). Compare within a column; the metric floor is 0.011 kT.

| method | 3.2e4 calls (kT) | 6.4e4 calls (kT) | 1.28e5 calls (kT) | 1.28e6 calls (kT) | reproduced by the bias verifier? |
|---|---|---|---|---|---|
| binned mean force (ABF, 64 bins) | 0.51 ± 0.07 | 0.109 ± 0.009 | 0.049 ± 0.009 | 0.025 ± 0.003 | not rerun |
| Fourier mean-force regression (12 harmonics) | 1.35 ± 0.10 | 0.183 ± 0.018 | 0.065 ± 0.010 | 0.032 ± 0.001 | rerun worse: 0.093 ± 0.011 at 1.28e5, 0.020 at 1.28e6 |
| umbrella sampling + WHAM, 32 fixed windows | 0.36 ± 0.05 | 0.193 ± 0.039 | 0.152 ± 0.013 | 0.059 ± 0.008 | not stated; windows placed in advance |
| VES, flat target | 2.83 ± 0.03 | 2.42 ± 0.05 | 0.61 ± 0.04 | 0.078 ± 0.012 | no |
| well-tempered metadynamics | 2.68 ± 0.03 | 1.63 ± 0.06 | 0.29 ± 0.04 | 0.129 ± 0.012 | no |
| OPES-like | 3.81 ± 0.21 | 2.50 ± 0.24 | 1.10 ± 0.16 | 0.096 ± 0.010 | no |

The free-energy metric has a floor: in the deriver's binned metric, the exact $F$ scores 0.011 kT. The "about 10×" advantage of mean-force regression is an observation on one easy toy with fast $x_\perp$ and lightly tuned baselines. Mean force is lowest from 6.4e4 calls on: the crossover with umbrella sampling plus WHAM lies between 3.2e4 and 6.4e4, and the Fourier variant leads umbrella only marginally at 6.4e4 (0.183 against 0.193 kT). The VES, metadynamics and OPES-like rows it is measured against were not reproduced.

### 3.18 Formulations named but not developed

Verifiers or the reviewer named these. None was trained here. Each line gives its status and the rung that would test it.

- **Two-sided tied committor** (CV verifier). The forward-to-$B$ control $\sigma^2\nabla\log q_\theta$ and the forward-to-$A$ control $\sigma^2\nabla\log(1-q_\theta)$ share one $q_\theta=f_\phi\circ s_\theta$, with hard rewards and per-start heads that satisfy $e^{\log Z_B}+e^{\log Z_A}=1$.
  - It turns Lemma G's two spurious zeros into the estimator. It removes $\epsilon$, and with it the unknown lower bound $\lvert\nabla q(\partial A)\rvert\sigma\sqrt{dt}$ (risks 3 and 6).
  - Each control is singular at the opposite set, so the heavy tails of Table 2 apply to both.
  - Structure DERIVED; that it trains is a CONJECTURE. Optional second arm of rung 3.
- **Exponential random horizon** $T\sim\mathrm{Exp}(\gamma)$ with a learned stop intensity (committor verifier). It solves $(\gamma-L)h_\gamma=\gamma R$, with drift $b+\sigma^2\nabla\log h_\gamma$ and stop intensity $\gamma R/h_\gamma$; the TB residual adds the Poisson log-likelihood ratio of the stop events.
  - It is the only horizon-type formulation with an exactly time-homogeneous optimum, so it needs no time input. Between $\lambda_1$ and $\lambda_2$ it encodes the same slow mode as §3.3.
  - DERIVED. Relevant to D1(b).
- **Owner-form rate readout** (committor verifier). Start law $\mu_0=\pi\mathbf 1_A/\pi(A)$, reversed-Langevin $P_B$, reward $\tilde R=\mathbf 1_B\pi$.
  - Then $Z/\pi(A)=C(T)=P_{\pi_A}(X_T\in B)$, and $k_{A\to B}=dC/dT$ on the plateau $1/\lambda_2\ll T\ll1/\lambda_1$.
  - It removes the dependence on a chosen point start. A per-start $\log Z_\phi(x_0)$ is needed for zero variance and for off-policy use.
  - DERIVED.
- **Learned forward-kernel covariance** $\Sigma=\sigma^2dt\,(I+\sigma^2dt\,H)^{-1}$ with $H=-\nabla^2\log h_{k+1}$. This is the covariance of $Q$'s one-step conditional (reviewer).
  - It is a common remedy candidate for the kernel-mismatch law (Table 2), for Lemma D's off-policy bias, and plausibly for the dt-independent bias of per-step detailed balance (Lemma F).
  - CONJECTURE. Rung 6.
- **Sequential importance resampling along the protocol under a fixed $P_B$** (bias verifier; CRAFT- and SNF-type). Resample on the exact sub-path weights when the effective sample size drops, and train $u_\theta$ stage by stage.
  - The $\Delta F$ estimator stays unbiased for $Z_1/Z_0$. It is the established per-force-call lever for transport under a non-learned backward.
  - Any force-call gain over Jarzynski with resampling is a CONJECTURE. No rung yet; it belongs to D1(d).
- **Off-policy validity of the tied form** (reviewer): see Lemma D.

### 3.19 The owner's proposal: a start-conditioned sampler with a Boltzmann endpoint and a bias anchored to physics

**Status.** This is the owner's proposal from the session of 2026-11-24. Items 1–14 were DERIVED in the session. In v3 a re-derivation checked them in closed form on a Gaussian OU Euler–Maruyama chain ($U=x^2/2$, $\beta=5$, $A=2$, $T=2$, dt = 0.02, 100 steps, unless stated; script `check_owner_idea/ou_checks.py`, output `check_owner_idea/ou_checks.json`, in `energy_sampling/SCRATCH/nsde_design_nov24/`).
- PROVED on that chain: item 3 (the value identity), item 7 (the bridge mismatch), item 9 (the geometric-mixture identity, the fixed-$P_B$ optimum $M$ and the reviser's h-transform form of $M$ for the Brownian bridge, and the joint combined-KL optimum $\rho_c$ with its $\log Z^*$), items 11 and 12, and item 13's cooling cost.
- REFUTED: identifiability of the optimum under a squared single residual with a free $\log Z$ (item 9).
- DERIVED only: items 1, 2, 4–6, 8, 10, 13(b) and 14. CONJECTURE: item 13(a)'s claim that the barrier-relevant bias appears only at the physical $\beta$.
- Lines marked "reviser" were added in v2; those not listed above are unchecked. The OU checks are not branch O0 and do not decide D5. Nothing has been trained, and the owner has not given a go.

**The proposal, in substance.**
- Train a GFlowNet-style diffusion sampler conditioned on the initial state $A$, so that from any $A$ the endpoint at a finite $T$ is Boltzmann.
- Train it on a combined loss: a TB-like term plus a KL to the reference (physical) process. In the owner's words, the KL term "pins u_learned to a meaningful quantity rather than some freely learned object".
- If it converges, the learned drift should be "something like the thermally weighted finite-time minimum bias potential".
- Read $A\to B$ from the model's own trajectories that land in $B$. Under Boltzmann weights some always do. The transitions of interest are between low-energy states. Assume the owner's usual tricks give full terminal support.
- Value claims, which the owner calls aggressive: (1) the whole landscape in one object, for example the free-energy map of a multi-polymorph system; (2) conditional training over many systems.
- The owner expects it to be very expensive, possibly the reason the field does not do it. The owner calls the $A\to B$ drift "a" CV-bias object, not "the" CV-bias object.
- The owner wanted the backward and the reference not to be learned (for example a Brownian bridge), and suggested running the reference very hot and annealing.

**Notation.** $P_{\rm phys}$ is the law of $dX=-\nabla U\,dt+\sigma dW$ from $X_0=A$ on $[0,T]$, $p_T(y\mid A)$ is its transition density, $\pi$ is the normalised Boltzmann density, and $\sigma^2=2/\beta$. In Table 1's terms: a point start, a fixed horizon, and $R=\tilde\pi$.

**Analysis.**

1. **A free reversible backward cancels the reward.** Take $P_B$ the reversed Langevin, unpinned: a free law over backward paths whose density is merely evaluated at the start, with the last factor $p_B(A\mid x_1)$ kept (Lemma C, point start). Take $R=\tilde\pi$. Lemma C applies with $g=R/\tilde\pi=1$. The target is plain physics from $A$, so $u^*=0$, and the endpoint is $p_T(\cdot\mid A)$, not $\pi$.

2. **Any backward pinned at $A$ gives an exactly Boltzmann endpoint.** Let $P_B(\cdot\mid x_T)$ be a normalised law over paths that end at $A$. Integrating the TB identity $Z\,P_F(\tau\mid A)=R(x_T)\,P_B(\tau\mid x_T)$ over the path interiors gives $Z\,p_F(x_T\mid A)=R(x_T)$. TB alone then has the unique optimum $P_F=\pi\otimes P_B$: endpoint $\pi$, bridges $P_B$. So the choice of $P_B$ selects which of the many transports from $\delta_A$ to $\pi$ in time $T$ is learned.
   - "Exactly" holds at the TB zero, which needs $\pi\otimes P_B$ to be reachable by the forward class. In continuous time a same-$\sigma$ pinned $P_B$ makes it reachable; at finite dt fixed-variance kernels cannot reach it (Table 2, kernel mismatch) (re-deriver).

3. **The physical bridge gives the minimum-bias object.** Take $P_B$ the physical bridge, which is physics conditioned on both ends. The target is
$$Q_A(d\tau)=P_{\rm phys}(d\tau\mid A)\,\frac{\pi(x_T)}{p_T(x_T\mid A)} .$$
   - This is §3.2's terminal conditioning with $g=\pi/p_T(\cdot\mid A)$. So $u^*=\sigma^2\nabla\log h$ with
$$h(x,t;A)=\mathbb E_{\rm phys}\Big[\frac{\pi(X_T)}{p_T(X_T\mid A)}\,\Big|\,X_t=x\Big],\qquad h(A,0;A)=1 .$$
   - It is a gradient, so $V=-(2/\beta)\log h$ is a time-dependent bias potential.
   - By the chain rule, $\mathrm{KL}(P^u\Vert P_{\rm phys})=\mathrm{KL}(p^u_T\Vert p_T(\cdot\mid A))+\mathbb E_{p^u_T}\mathrm{KL}(\text{bridge of }P^u\Vert\text{physical bridge})$. With the endpoint fixed at $\pi$, the first term is fixed, and the second is zero only for physical bridges.
   - So $u^*$ minimises $\mathrm{KL}(P^u\Vert P_{\rm phys})=\frac\beta4\mathbb E\int_0^T\lvert u\rvert^2dt$ among all drifts with a Boltzmann endpoint. The minimum is $\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$, which tends to 0 as $T\to\infty$.
   - This is the Schrödinger half-bridge from $\delta_A$ to $\pi$ with the physical dynamics as the reference.
   - The minimum is itself a readout per start: how far $A$ is from equilibrium at time $T$ (reviser).
   - [PROVED on the OU chain for the value identity.] $\mathrm{KL}(\pi\otimes\text{physical bridge}\Vert P_{\rm phys})$ equals $\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$ to about 1e-12 at every horizon tested (Table 5a). Minimality over *all* drifts rests on the exact chain rule; numerically it was compared only against the Brownian-bridge alternative, which is always higher.

**Table 5a.** Bias cost of two transports from $A=2$ to $\pi$ on the OU EM chain ($k=1$, $\beta=5$, dt = 0.02), against the horizon $T$. Columns are path KLs to the physical process in nats, computed in closed form. Compare the two transport columns: the physical-bridge column must equal $\mathrm{KL}(\pi\Vert p_T)$, and the Brownian-bridge column shows the excess of a non-physical backward.

| horizon $T$ | $\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$ (nats) | $\pi\otimes$ physical bridge (nats) | $\pi\otimes$ Brownian bridge (nats) |
|---|---|---|---|
| 0.5 | 5.727 | 5.727 | 5.744 |
| 1 | 1.518 | 1.518 | 1.633 |
| 2 | 0.1773 | 0.1773 | 0.907 |
| 5 | 4.31e-4 | 4.31e-4 | 5.32 |
| 10 | 2.52e-5 (EM stationary-variance floor) | 2.52e-5 | 17.65 |

4. **"Thermally weighted" is literal.**
$$u^*(x,t)=\int w(y\mid x,t)\,\sigma^2\nabla_x\log p_{T-t}(y\mid x)\,dy,\qquad w(y\mid x,t)\propto p_{T-t}(y\mid x)\,\frac{\pi(y)}{p_T(y\mid A)} .$$
   This is the average of the physical bridge drifts toward endpoints $y$, weighted as shown.

5. **The paths that land in $B$ are physical paths.** Conditioned on its endpoint $b$, a $Q_A$ path is the physical bridge $A\to b$; $\pi$ drops out, and only the endpoint carries a weight.
   - The drift that generates the $B$-landing sub-ensemble is $u_B=\sigma^2\nabla\log h_B$, with $h_B=\mathbb E[\mathbf 1_B(X_T)\,g(X_T)\mid X_t=x]$, and
$$u_B=u^*+\sigma^2\nabla\log w_B,\qquad w_B(x,t;A)=Q_A(X_T\in B\mid X_t=x) .$$
     Here $w_B$ is the model's probability of landing in $B$. It can be learned as a classifier on the model's own samples.
   - More generally, $u_B=u+\sigma^2\nabla\log w^u_B$ holds for any trained Markov drift $u$, with $w^u_B$ its own landing probability; the $B$-landing paths are physical only at the optimum $u=u^*$ (re-deriver; DERIVED).
   - If $p_T\propto\pi$ inside $B$ ($B$ locally equilibrated by $T$), then $u_B=\sigma^2\nabla\log P_{\rm phys}(X_T\in B\mid X_t=x)$. This is a finite-time h-transform whose potential has finite-time iso-committor level sets: "a" CV-bias object. (The session called it a hitting transform. Strictly, it conditions on the position at time $T$, not on first hitting; reviser.)
   - Reviser (two-state heuristic, as in §3.3; not checked). This $u_B$ is §3.2's drift with $g=\mathbf 1_B$. In the spectral window $1/\lambda_2\ll T\ll1/\lambda_1$, §3.3 then gives $u_B\approx\sigma^2\nabla\log(\epsilon_{\rm eff}+(1-\epsilon_{\rm eff})q)$, a soft committor drift. In the same window $g=\pi/p_T(\cdot\mid A)$ is nearly constant on each basin. So $u^*$ itself takes the same soft-committor form, with a floor set by the ratio of $g$ between the two basins.

6. **The only intractable piece is one scalar per endpoint pair: $\log p_T(x_T\mid A)$.**
   - Every fixed tractable backward runs into it.
   - Dropping it, so that the target is $\pi(x_T)\,P_{\rm phys}(\tau\mid A)$, tilts the endpoint to $\propto\pi\,p_T(\cdot\mid A)$ and brings the rare-event problem back.
   - Detailed balance, $p_T(b\mid A)\,\pi(A)=p_T(A\mid b)\,\pi(b)$, means that reversing trajectories moves the number to the other end. Both ends cannot be pinned cheaply.
   - Learning a scalar $F(x_T;A)$ in place of $\log p_T(x_T\mid A)$ inside TB does not help: every $F$ gives zero residual, with endpoint $\propto\tilde\pi\,p_T(\cdot\mid A)\,e^{-F}$, so $F$ is not identified (re-deriver; DERIVED).

7. **A fixed Brownian bridge pinned at $A$** (the PIS and GFN-diffusion choice) gives the target $\pi(x_T)\times$ Brownian bridges.
   - The total drift is the PIS-type minimum-energy transport relative to free diffusion. The learned part largely cancels the physical force, and the paths that land in $B$ are Brownian bridges, not physical paths.
   - This repository's sampler uses this kind of $P_B$ (checked against the code in v2). `models/gfn.py::GFN.var_drift_coeff` and `::GFN.var_bridge_step` contract the backward step toward the deterministic start $x_0=0$. `configs/mk_dev.yaml` adds a learned, bounded correction (`model.learn_pb: true`, `model.pb_drift_range: 0.4`).
   - Its `model.force_drift_bwd` option adds a gated force term to $P_B$'s mean (`docs/wiki/trajectory-balance.md`; null in `mk_dev.yaml`). That is physics-flavoured, but it is not the physical bridge.
   - [PROVED on the OU chain for the bridge mismatch.] At $T=2$, $\mathbb E_\pi\mathrm{KL}(\text{Brownian bridge}\Vert\text{physical bridge})=0.729$ nats (0.907 − 0.177, Table 5a), and $\mathrm{KL}(\pi\otimes\text{BB}\Vert P_{\rm phys})$ grows with $T$, to 17.65 nats at $T=10$ against the minimum 2.5e-5.

8. **The reference and the backward separate.** In the standard GFN sampler the Brownian bridge is both the TB backward and, implicitly, the reference, because it is Brownian motion's own bridge. With physics as the reference the two come apart:
   - the reference stays fixed and non-learned, and enters through the KL term;
   - the backward becomes either the physical bridge, which needs $p_T$, or a helper.

9. **The combined loss.** For $c>0$,
$$c\,\mathrm{KL}(P^u\Vert\pi P_B)+\mathrm{KL}(P^u\Vert P_{\rm phys})=(1+c)\,\mathrm{KL}(P^u\Vert M)+\text{const},\qquad M\propto(\pi P_B)^{\frac c{1+c}}\,P_{\rm phys}^{\frac1{1+c}} .$$
   So, for a *fixed* $P_B$, the combined KL is the expectation under $P_F$ of ONE residual:
$$\delta=(1+c)\log P_F-c\,(\log\tilde\pi+\log P_B)-\log P_{\rm phys}-\log Z .$$
   [PROVED on the OU chain: the identity holds to 6e-11 on values up to 1065, for a random Gaussian chain and $c=0.5$, 3, 50.]
   - **Fixed $P_B$.** The optimum is $M$. Its bridges blend those of $P_B$ and of the physics, and its endpoint is only partly Boltzmann.
     - Reviser: for $P_B$ the Brownian bridge to $A$, $M$ is itself an h-transform, so it can be computed on a grid in 1D (branch O0). Its reference drift is $(1-a)b$ with $a=c/(1+c)$, its killing rate is $a(1-a)\lvert\nabla U\rvert^2/(2\sigma^2)$, and its terminal reward is $(\pi/w_T(\cdot\mid A))^a$, where $w_T$ is the Brownian transition density. The reason is that the geometric mixture of the Brownian and physical path measures is the drift-$(1-a)b$ diffusion times $e^{-a(1-a)\int\lvert b\rvert^2dt/(2\sigma^2)}$. [PROVED on the OU chain: exact match.]
     - OU, Brownian-bridge arm: endpoint KL to $\pi$ 0.0296, 6.0e-4 and 6.3e-6 nats at $c=1$, 10, 100, while the bridge KL to physics rises, 0.144, 0.573 and 0.711, toward 0.729 nats. The endpoint goes Boltzmann and the bridges go Brownian.
   - **Learned helper $P_B$ pinned at $A$, combined KL minimised jointly** [PROVED on the OU chain].
     - Minimising over $P_B$ turns the TB term into $c\,\mathrm{KL}(p^u_T\Vert\pi)$.
     - The reference KL splits into an endpoint part and a bridge part. The bridges go physical, and the endpoint goes to
$$\rho_c\propto\pi^{c/(1+c)}\,p_T(\cdot\mid A)^{1/(1+c)} .$$
     - The drift is the h-transform with $g=(\pi/p_T)^{c/(1+c)}$, and $c\to\infty$ gives the minimum-bias object.
     - At the optimum, $\log Z^*=-(1+c)\log N_c-c\log Z_U$ with $N_c=\int\pi^{c/(1+c)}p_T(\cdot\mid A)^{1/(1+c)}$, and $-(1+c)\log N_c\to\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$, the minimum of item 3 (reviser).
     - OU: over a family of OU-bridge helpers ($k'/k$ = 0, 0.5, 0.9, 1, 1.1, 1.5, 2), the $c=1$ objective is 0.429, 0.257, 0.097, 0.088, 0.097, 0.313 and 0.923 nats, minimised at the physical bridge ($k'=k$). There $\rho_c$'s mean and variance match to 6e-14, and $-(1+c)\log N_c$ = 0.0883, 0.1610, 0.1755 and 0.17723 at $c=1$, 10, 100 and 1e4, against $\mathrm{KL}(\pi\Vert p_T)=0.17725$.
     - **$\log Z$ is not the bias-cost readout.** The $-c\log Z_U$ term diverges as $c$ grows and needs the unknown $Z_U$. Read the per-start cost $\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$ as the model's own $\frac\beta4\mathbb E\int_0^T\lvert u\rvert^2dt=-\mathbb E_{P_F}\log w$ at large $c$ (re-deriver; equal to $\mathrm{KL}(\pi\Vert p_T)$ at the optimum on OU).
   - **The helper must be trained on the expected residual, not the squared one** [REFUTED as a training statement for the squared residual; OU]. For a helper, the term $(1+c)\log M(P_B)-c(\log\tilde\pi+\log P_B)-\log P_{\rm phys}$ is the constant $-(1+c)\log N(P_B)-c\log Z_U$ on every path. So every pair $(M(P_B),P_B)$ — any helper with its own blend as the forward — has zero squared residual once the free $\log Z$ absorbs that constant, and the constant is exactly the normaliser that the KL form uses to select the physical bridge.
     - OU: Var $\delta\le$ 2.3e-23 at $M(\text{Brownian bridge})$, $M(\text{OU}(k/2)\text{ bridge})$ and $M(\text{physical bridge})$, for $c=1$ and 10, while their combined-KL values at $c=1$ are 0.429, 0.257 and 0.088 nats.
     - The squared-residual gradient for the helper, $-2c\,\mathrm{Cov}(\delta,\nabla_\phi\log P_B)$, vanishes at every such point. The expected-residual (KL) gradient, $-c\,\mathbb E_{P_F}[\nabla_\phi\log P_B]$, does not: it is maximum likelihood of the helper on the forward's own paths, i.e. ASBS corrector matching.
     - So the optimum statements above hold only when the helper is trained that way. The forward drift may still use the squared residual on-policy (Lemma D). The endpoint-only certificate (item 11) detects the wrong zeros: conditional $\log w$ variance 1.49 nats² at $M(\text{BB})$ ($c=10$), against about 1e-27 at the optimum.
   - **A sum of separate terms.** A squared or Huber TB term plus a separate KL term does not reach $\rho_c$ at finite $c$.
     - Reviser's refinement: on-policy with detached samples, the forward gradient of the squared TB term is twice the reverse-KL gradient (Lemma D), so the forward part acts like $c\to2c$.
     - But the helper's gradient is weighted by the TB residual, which is not constant at the blended optimum. Off-policy, or with a Huber cap, the forward part departs as well.
     - Re-deriver [DERIVED, not checked]: with a learned helper, its $c\to\infty$ limit is still the minimum-bias object, because TB² = 0 forces $\pi\otimes P_B$ and the separate KL then selects physical bridges. A separate KL term is not absorbed by $\log Z$, so the sum is better identified than the folded squared residual.
   - **Conditioning.** Large $c$ weakens the physics selection: pure TB with a learned backward is non-identifiable (item 2). So training becomes ill-conditioned as $c$ grows; the selection margin stays roughly constant in absolute terms while the TB curvature scales with $c$ (re-deriver).

10. **The helper does not redefine the bias at convergence — if it is trained by maximum likelihood on the forward's paths.** With that update (ASBS corrector matching) the target is defined by the physics, and the helper's unique optimum is the physical bridge (OU: the helper family above is minimised at $k'=k$). Before convergence its training error leaks into the bias. A helper trained by the squared TB residual has no unique optimum (item 9).
    - ASBS corrector matching (Liu et al. 2025, arXiv:2506.22565) is the published machinery for this helper: a regression on forward samples.
    - Its theory allows a physical base drift and a point source. With a Dirac source its corrector is $\nabla\log p^{\rm base}_1$, which here is $\nabla\log p_T(\cdot\mid A)$ (novelty verifier).
    - Its experiments use a Brownian reference.

11. **A certificate that needs no helper.** At the exact optimum,
$$\log w(\tau):=\log\frac{dP_{\rm phys}}{dP_F}(\tau\mid A)=\log p_T(x_T\mid A)-\log\pi(x_T),$$
    a function of the endpoint only. It can be computed from the model's own path by Lemma A, without $p_T$.
    - Conversely, an endpoint-only log-weight together with Boltzmann endpoints implies physical bridges, and hence the minimum-bias object.
    - Test: regress $\log w$ on $x_T$. The residual spread at fixed endpoint measures the non-physical part of the bias.
    - The regression detects non-physical bridges only. The endpoint must be checked separately: $\rho_c\otimes$ physical bridges also has an endpoint-only $\log w$ (re-deriver). At finite dt with fixed-variance kernels the spread has a kernel-mismatch floor (Table 2), which O1 allows for.
    - Reviser: the fitted function also returns $\log p_T(\cdot\mid A)$ up to a constant, which normalisation fixes.
    - [PROVED on the OU chain.] Residual variance of $\log w$ after an exact quadratic regression on $x_T$, 40,000 paths each: 8.5e-28 at $\pi\otimes$ physical bridge, 8.4e-28 at $\rho_c\otimes$ physical bridge ($c=10$), 1.95 nats² at $\pi\otimes$ BB, and 1.49 nats² at $M(\text{BB})$, $c=10$.

12. **Rates by reweighting.** $P_{\rm phys}(X_T\in B\mid A)=\mathbb E_{P_F}[\mathbf 1_B(X_T)\,e^{\log w}]$ is unbiased for any trained $u$ with full support (Girsanov). Training quality sets only the variance. [PROVED on the OU chain; Table 5b.]
    - Reviser, qualified by the re-deriver: at the optimum the per-sample variance is $\mathrm{Var}_\pi(\mathbf 1_B\,p_T/\pi)$. When $p_T/\pi$ is constant on $B$, the relative variance is $(1-\pi(B))/\pi(B)$, independent of how rare $A\to B$ is; that is the gain. It is not small in absolute terms: for a non-rare $B$ it can exceed direct sampling.
    - The minimum-bias object is not the variance-optimal proposal for any single $B$: on OU a non-optimal proposal did better.

**Table 5b.** Reweighted estimates of $P_{\rm phys}(X_T>0.5\mid A)$ on the OU EM chain ($A=2$, $T=2$; exact 0.29911), $10^5$ paths per proposal. Compare the estimates with the exact value, and the per-sample variances with direct sampling; the predicted optimum variance is $\mathrm{Var}_\pi(\mathbf 1_B\,p_T/\pi)=0.653$.

| proposal | estimate ± SE | per-sample variance |
|---|---|---|
| minimum-bias optimum ($\pi\otimes$ physical bridge) | 0.29982 ± 0.00256 | 0.655 |
| $\pi\otimes$ Brownian bridge | 0.2913 ± 0.0056 | 3.16 |
| $M(\text{BB})$, $c=1$ | 0.29704 ± 0.00245 | 0.598 |
| direct sampling of the physics | – | 0.210 |
    - A rate constant needs this at two or more horizons, as $dC/dT$ on the plateau (§3.3; owner-form rate readout, §3.18).

13. **A hot reference.**
    - (a) *Across training.* A hot reference makes barriers cheap, so a helper's data reach every basin. Raise $\beta_{\rm ref}$ to the physical $\beta$ (cool the reference) on a fixed, finite schedule, with warm starts. The schedule is bounded, because it ends at the physical $\beta$.
      - A fixed hot reference with a cold Boltzmann endpoint is the same variational problem as the "fast cooling" of Chen, Georgiou & Pavon 2015 (arXiv:1411.1323; read by the novelty searcher, not re-verified), up to the time change $t'=(\beta_{\rm ref}/\beta)\,t$. CGP's problem is underdamped and starts from a given initial law, not a point.
      - While the reference is hot, the bias mostly does cooling everywhere [PROVED on the OU chain]. The minimum bias then never vanishes: as $T$ grows it tends to $\mathrm{KL}(\pi\Vert\pi_{\rm ref})=\tfrac d2(r-1-\ln r)$ nats with $r=\beta_{\rm ref}/\beta$, extensive in $d$, instead of to 0. OU, dt = 0.05: at $r=0.25$ the minimum-bias KL is 1.596, 0.334 and 0.3205 nats at $T$ = 2, 10, 50 (limit 0.318); at $r=0.5$, 0.817, 0.0999 and 0.0997 (limit 0.0966); at $r=1$, 0.164 and then 1.6e-4 (the EM floor).
      - [CONJECTURE] The barrier-relevant bias appears only once the reference is at the physical $\beta$. There is no derivation: at an intermediate $\beta_{\rm ref}$, a reference that is not equilibrated by $T$ still needs inter-basin transport in the minimum-bias drift. The three-hole rung O3 tests it.
      - The three-hole toy tests this, because its channel preference flips with temperature: the upper-channel share is 0.348 at $\beta=1.67$ and 0.629 at 6.67 (Table 7).
    - (b) *Within each path.* A reference that cools along the trajectory, as in simulated annealing. Its path density is tractable, but $p_T$ is still unknown. This joins the annealed-transport family (§3.13).
    - Reviser, from Lemma C(c, d): the reference and the model must share $\sigma$. So a hot reference is run at the model's $\sigma$ as the drift $-(\beta_{\rm ref}/\beta)\nabla U$, which is a time-rescaled hot Langevin. Mixing noise levels makes the path measures mutually singular.

14. **Practicalities.**
    - The Boltzmann endpoint is a training target, not a guarantee.
    - The fraction of paths that land in $B$ is $\pi(B)$, which is fine for low-energy $B$. For example, starting in the deeper well of the 1D asymmetric double well at $\beta=5$, the shallow *basin* $B=\{x>0.0627\}$ (beyond the barrier top) has $\pi(B)=0.0852$ by quadrature, matching $k_{A\to B}/(k_{A\to B}+k_{B\to A})$ from Table 7. Table 7's stopped set $B=\{x\ge0.9671\}$ has $\pi(B)=0.0354$ (re-deriver). Branch O0–O1 uses the basin.
    - Amortisation over $A$, and over systems, is the main cost.

**What the value claims would rest on** (reviser's reading of items 3, 5 and 12).
- *(1) The whole landscape in one object.* For each start $A$ and horizon $T$, the object gives:
  - Boltzmann samples, hence basin populations and the free-energy map. Any equilibrium sampler gives these.
  - Lag-$T$ transition probabilities $P_{\rm phys}(X_T\in B\mid A)$, by reweighting (item 12).
  - Physical $A\to B$ paths (item 5).
  - The bias cost $\mathrm{KL}(\pi\Vert p_T(\cdot\mid A))$ (item 3), read as the model's own $\frac\beta4\mathbb E\int_0^T\lvert u\rvert^2dt=-\mathbb E_{P_F}\log w$ at large $c$, not from $\log Z$ (item 9).

  Across starts that cover the basins, the transition probabilities form a lag-$T$ transition matrix between basins: an MSM-like object, for which TITO's checks can be borrowed. The claim fails where the starts miss a basin, or where the reweighting variance is large because the model is far from its optimum.
- *(2) Conditional training over many systems.* This is the same amortisation as ASBS (over molecular graphs) and TITO (over molecules). It is untested here.

**Novelty search.** The search was run as workflow wf_f3b71b61-fd2. It found 92 unique papers across four families, by web search; 8 were read in full by verifiers.
- **Topline.** No one was found building the whole thing: a sampler trained from the energy alone that starts from a given state, lands on Boltzmann at a finite time, keeps its bias close to the physical dynamics, and is then read for paths and rates.
- Chen, Georgiou & Pavon 2015 pose the same variational problem (underdamped Langevin reference, a given initial law, a colder Boltzmann target at finite time) and solve it only for linear systems. ASBS is the training machinery. EntangledSBM puts a learned bias on physical Langevin but pins $B$. TITO is the transferable, MD-trained competitor. ITO, BoPITO, Timewarp, MDGen and KC-Maps are conditioned on the start but imitate unbiased MD.

**Table 5.** The closest works to §3.19 found by the novelty search. Each cell answers its column's yes/no question for that paper. "Full text" rows were read by a verifier; "searcher only" rows were not re-verified. The owner's object would answer yes in the first five columns, so compare each row against all-yes.

| work (id) | trained from the energy, without MD? | conditioned on a start state? | Boltzmann endpoint at finite $T$? | bias relative to the physical dynamics? | paths or rates read out? | how checked |
|---|---|---|---|---|---|---|
| ASBS, Liu et al. 2025 (arXiv:2506.22565; NeurIPS 2025) | yes; amortised over molecular graphs | no: Gaussian or harmonic sources (the theory allows a Dirac source) | yes | theory allows a base drift; every experiment uses $f_t=0$ | no | full text |
| Chen, Georgiou & Pavon 2015, "fast cooling" (arXiv:1411.1323) | no learning; posed for nonlinear underdamped Langevin, solved for linear systems only | no: a given initial law (Gaussian in the worked example) | yes | yes | no | searcher only |
| EntangledSBM, Tang et al. 2025 (arXiv:2511.07406) | path-measure loss (data source not re-verified) | no: one fixed start; the bias is conditioned on the target | no: conditioned on a target basin | yes: a learned bias on physical Langevin | paths; no rates; one system per model | searcher only |
| TITO, Viguera Diez et al. 2025 (arXiv:2510.07589; Sci. Adv. 2026) | no: MD pairs, no energy term | yes, plus lag and molecular graph | tested by JSD, not imposed | no: an ODE in generative time | timescales (implied timescales, VAMP-2); no path ensembles | full text |
| BoPITO 2024 (arXiv:2410.10605; ICLR 2025) | no: MD, plus a pretrained equilibrium model | yes | only as the lag grows without bound | no | per-system kinetics | full text |
| KC-Maps 2026 (arXiv:2611.12880) | force field plus equilibrium samples, no trajectories | yes | no: it learns the unbiased kernel, lags up to about 100 fs | no bias: imitates the physical kernel | MFPT on AL2 only | full text |
| MDGen 2024 (arXiv:2409.17808; NeurIPS 2024) | no: MD data only | yes; paths by pinning both end frames | no | no | data-trained paths on unseen tetrapeptides | full text |
| ITO 2023 (arXiv:2305.18046), Timewarp 2023 (arXiv:2302.01170), PLaTITO 2026 (arXiv:2602.11216) | no: MD (Timewarp fine-tunes on an acceptance ratio that contains $U$) | yes | no | no | ITO: MSM MFPTs; PLaTITO: MFPTs 2–5× faster than MD | full text |

- **Expected reviewer question.** Why not a TITO-style learned propagator plus a Boltzmann prior? Answer: this object is trained from the energy and needs no MD that already crosses the barriers.
- **Borrowable pieces.**
  - ASBS corrector matching for the helper (item 10): maximum likelihood on forward paths, the update item 9 requires.
  - BoPITO: draw training starts $A$ from an equilibrium sampler (here, the existing crystal GFN), not from one structure. This bears on O2's amortisation over $A$.
  - The horizon $T$ as a network input, as ITO and TITO condition on the lag.
  - TITO's checks: implied timescales, VAMP-2, and Chapman–Kolmogorov self-consistency.
  - Marginal Girsanov reweighting for rates (Wang, Wu & Olsson 2025, arXiv:2509.25872).
  - KC-Maps' architecture, which builds the physical short-time step into the network so the bias vanishes at short times.
  - PLaTITO as a cautionary data point: kinetics 2–5× too fast against MD.
- **What looks open.** A bridge with the physical dynamics as reference, from a point, amortised over starts, in high dimension. Reading paths and rates from it by reweighting. Any of this for molecular crystals; the only periodic generative kinetics found were LiFlow and SupraTITO. A system-conditional version.
- **Coverage limits.**
  - Four families were searched with web search; there was no citation-graph crawl.
  - Eight papers were checked in full text; the rest rest on abstracts or listings.
  - Langevin Flow Maps (arXiv:2610.05998) could not be read beyond its abstract, and a Gloy & Olsson preprint was found only inside a thesis.
  - The gaps above come from the search; they are not proof that nobody has done this.
  - The novelty searcher also states item 5 (B-landing paths of an exact half-bridge are physical paths) as not found in any paper.

**Pending owner decision (D5).** Fixed backward or learned helper, for this object. The proposed first step needs no training and has not been run. On a 1D double-well grid, compare the exact minimum-bias drift with three alternatives:
- (i) a fixed Brownian bridge plus the KL term;
- (ii) the physical bridge with the exact $p_T$;
- (iii) the optimum with a learned helper trained by maximum likelihood on forward paths.

This is branch O0 of §6. The v3 OU checks bear on it but do not replace it: they are on a single-well reference, where no barrier is crossed.

**What pins the helper (v4).** The owner asked what makes the learned helper unique and physically meaningful rather than an arbitrary number. An independent deriver and an adversarial refuter checked six claims on closed-form Gaussian OU Euler–Maruyama chains. Scripts and outputs are in `energy_sampling/SCRATCH/nsde_design_nov24/helper_check/deriver/` and `.../helper_check/refuter/`.

**Table 5c.** Claims about the helper and their status after the check. Two OU cases were used: "mild" ($A=2$, $k=1$, $T=1$) and "far" ($A=3$, $\sigma^2=0.1$, with $\pi$ 5.2 standard deviations of $p_T$ away from $p_T$). Nothing here was run on a double well, so barrier crossing is untested.

| claim | status | evidence on the OU chain |
|---|---|---|
| H1. For a fixed forward, maximum likelihood of the forward's paths under the helper has a unique minimiser on the forward's path support: the forward's own bridges | PROVED, when the helper family contains those bridges | five random restarts agree to 1e-14 in the objective; natural-parameter distance to the bridge 2.7e-14 |
| H2. At the joint optimum, for any finite $c$, the helper's reversed-time drift is $-b+\sigma^2\nabla\log p_t(x\mid A)$, free of $\pi$ and $c$. The forward's marginal is $p_t(x\mid A)\,h_c(x,t)$; it is $p_t h$ only as $c\to\infty$ | PROVED (the marginal corrected from $p_t h$) | helper against the physical bridge: 3.6e-14 to 5e-12 relative over nine ($\pi$, $c$) cases; drift error $O(\Delta t)$ |
| H3. Exact alternation (helper set to the forward's bridges; forward set to the argmin of $J$) shrinks the log-ratio of the bridges to the physical ones by exactly $a=c/(1+c)$ per round, up to a function of the endpoint. The only fixed point is physical bridges with endpoint $\rho_c$. At $c=\infty$ every helper is a fixed point | PROVED | measured per-round ratios 0.2000, 0.5000, 0.8000, 0.9500 against $a$ = 0.2, 0.5, 0.8, 0.95 |
| H4. From $u_\theta=0$, one exact round reaches the fixed point | PROVED only with the helper frozen while the forward is minimised to convergence | KL to the fixed point after one round ≤ 1e-14 |
| H5. A helper written as a scalar $\ell_\phi\approx\log p_t(\cdot\mid A)$ and penalised by the log-form Fokker–Planck residual has a unique zero | CORRECTED: unique only with the singular-start form imposed globally (including far from $A$ at small $t$), the residual enforced on the whole space, and positivity | $\log\pi$, $\log p_t(\cdot\mid A')$, mixtures, constant shifts and solutions injected across a collocation boundary are all exact zeros. One mixture matches the start form on the forward's small-$t$ support to within $10^{-108}$, yet sends 52% of the endpoint mass back to $A'$ |
| H6. In practice the risk is coverage of the transition bridges the helper must represent | CONJECTURE, sharpened | in the far case the maximum-likelihood helper's error at $\pi$'s endpoints is 4–5 times its error under $p_T$ (1,000 and 10,000 physics paths) |

What this means for training:
- **Physics, not the forward, pins the helper.** Inside $J$ the helper is identified only through $\mathrm{KL}(P_F\Vert P_{\rm phys})$; the TB-like term alone leaves every helper a fixed point ($c=\infty$). So the anchoring weakens as $c$ grows. The loop is not circular, but it is weakly conditioned at large $c$.
- **Soft valley: alternate, do not train jointly.** Joint gradient descent and refits of the helper to a partly trained forward fall into a valley where the forward's and the helper's bridges move together. Its curvature is about 0.16 at $c$ = 1, 10 and 100, while the largest curvature grows in proportion to $c$, so convergence slows by a factor of order $c$. Iterations to reach a KL of 1e-4 to the optimum in the mild case: frozen helper 50, 100, 100 at $c$ = 1, 10, 100; joint descent 150, 1,740, and not reached in 5,000. Freeze the helper while the forward converges, then refit it.
- **Helper step variance near the pin.** With the helper's step variance fixed at $\sigma^2\Delta t$ (the usual Euler–Maruyama choice) it cannot represent the bridges next to the pin. The floor is about $(1-\gamma)/2\approx0.21$ nats per dimension and does not shrink with $\Delta t$; the joint optimum then misses $\rho_c$ with physical bridges by 0.28–0.32 nats at $c=10$. A Brownian-bridge variance schedule, $\sigma^2\Delta t\,(n-1)/n$ for the step from $x_n$ back to $x_{n-1}$, restores $O(\Delta t)$ accuracy.
- **A helper frozen at the physical reversal makes the problem one solve.** Then $c\,\mathrm{KL}(P_F\Vert\pi B_{\rm phys})$ alone has $Q_A$ as its unique minimiser, and the KL to physics only tilts the endpoint toward $\rho_c$ (DERIVED from the $J$ identity). Two routes give such a helper: maximum likelihood on physics paths from $A$, which physics rarely carries into other basins; or the Fokker–Planck residual with the global start condition of H5, which is stiff where $p_t$ is tiny.
- **Leaks hide themselves.** Where the helper under-assigns probability to paths ending at $A$, the forward's optimal endpoint density drops by a power of the helper's return probability, so the forward avoids exactly the endpoints where the helper is wrong. Under maximum likelihood on forward paths such a leak shows up as a per-endpoint log-normaliser deficit (checked to 1e-14). The Fokker–Planck residual cannot see that deficit, and maximum likelihood cannot see the helper's shape off the forward's support. Using both covers each one's blind spot (suggestion, untested).

---

## 4. Prior work and novelty

**Table 6.** Prior work most relevant to the owner's design. "Verified" means the survey fetched the arXiv or publisher page. Venue caveats are in the cells. Rows marked "named by the reviewer" were not fetched by any survey. Work closest to the owner's proposal is in Table 5 (§3.19). Several 2026 arXiv ids rest on a single survey fetch: Guo et al. 2609.28010, the only prior art for §3.9's bias optimum; Pigeon et al. 2607.21425; Hou et al. 2610.10054, Petersen et al. 2608.02536 and Negedly et al. 2609.20461 (one row below). Re-fetch each before it carries a novelty argument.

| work | id (existence verified?) | what it does, in the common notation | relation to the owner's idea |
|---|---|---|---|
| Hartmann & Schütte 2012 | arXiv:1208.3232 (yes) | log-transformed value function; optimal forcing is conservative, bias $2F$; cross-entropy | origin of "optimal drift = Doob = bias potential" |
| Lu & Nolen 2015 | arXiv:1303.1744 (yes, abstract only) | reactive segments are the h-transform with $h=q$ | target law of §3.4 |
| Das, Rose, Garrahan & Limmer 2021 | arXiv:2105.04321 (yes) | reverse KL / RL to tilted path ensembles; value function linked to the committor | early reverse-KL form of the design |
| Nüsken & Richter 2021 | arXiv:2005.05409 (yes) | log-variance divergence; moment loss with learned $y_0=-\log Z$ | the TB / log-variance loss family |
| Singh & Limmer 2023 | arXiv:2302.14857 (yes) | forward KL on TPS data, $u=(2/\beta)\nabla\log q_\theta(x,t)$ with a time-dependent $q$ at fixed $t_f$; bound on $\ln(kt_f)$ | forward-KL arm; needs path data; does not train toward the stationary committor drift |
| Singh & Limmer 2024 | arXiv:2402.05414 (yes; journal not confirmed) | fixed-horizon $q_B(x,\tau)$ reduces to the splitting probability in the spectral window | theory behind §3.3 |
| Holdijk et al. 2023 (PIPS) | arXiv:2207.02149 (yes; the whole row is abstract-level, and the loss direction comes from garbled equations) | path-integral control of a bias potential; fixed horizon; soft endpoint | $u_{\rm phys}+u_\theta$, no $P_B$ |
| Yuan, Shah, Bentz & Cameron 2023 | arXiv:2305.17112 (yes) | committor from a PDE loss, plugged into the control; only transition paths; rates | control from a non-path learner; no drift trained by a path loss |
| Seong et al. 2025 (TPS-DPS) | arXiv:2405.19961 (yes; ICLR 2025) | log-variance / TB with learned $\log Z$; off-policy replay; temperature annealing; fixed horizon; state-only equivariant bias | closest to the TB leg; no backward kernel, no committor readout, no transfer |
| Du et al. 2024 (Doob's Lagrangian) | arXiv:2410.07974 (yes; NeurIPS 2024) | simulation-free action on Gaussian marginals; $v^*=\nabla\log h$; Dirac endpoints | the force-call comparison against shooting |
| Earle & Van Koten 2025 | arXiv:2502.04605 (yes; SIAM journal name unconfirmed) | reverse KL to the exact reactive process, $u=\sigma^2\nabla\log\tilde q$ | reverse KL of §3.4 |
| Du, He, Zhang, Vanden-Eijnden & Domingo-Enrich 2026 (REACT) | arXiv:2604.13213 (yes; v4 is dated 23 Nov 2026, one day before this note; Appendix F, which compares log-variance, moment and FBSDE losses, not read) | stopped; $\Phi=-\log q_\xi$; off-policy value matching; uniqueness theorem; finite-element toys; alanine; chignolin with about 100× less simulation than the semigroup loss | most complete prior art for §3.4 and §3.5 |
| Hartmann, Joester, Schütte, Sikorski & Weber 2026 | arXiv:2601.01489 (yes) | compares control and importance-sampling formulations for committors and exit rates with unbounded stopping times | stopping-time control theory; not a TB method |
| Kang, Trizio & Parrinello 2024; Khoo, Lu & Ying 2019; Li, Lin & Ren 2019 | arXiv:2401.05279; 1802.10275; 1906.06285 (yes; the venue of Khoo et al. is unconfirmed: Res. Math. Sci. per one survey, none per the other) | Dirichlet-energy committor networks; static bias $-(\lambda/\beta)\log(\lvert\nabla q\rvert^2+\epsilon)$ | non-path baselines (§3.15) |
| Lorpaiboon, Weare & Dinner 2025 | arXiv:2509.03539 (yes, abstract) | lag-time variational committor exact only at one step; the stopping-time form is exact | supports stopped formulations |
| Guo, Hartmann, Berger & Nüske 2026 | arXiv:2609.28010 (yes) | ergodic control with KL-rate cost; bias $-(2/\beta)\log\varphi$ by eigenproblem | the bias optimum of §3.9; not trained on trajectories |
| Valsson & Parrinello 2014; Bonati, Zhang & Parrinello 2019 | arXiv:1407.0477; 1904.01305 (yes) | the convex VES functional | §3.11 |
| Darve & Pohorille 2001 (ABF) | J. Chem. Phys. 115, 9169 (no arXiv; verified by search) | mean force on CVs | §3.10 is this |
| Zhang, Wang & E 2018 ("reinforced dynamics") | not in the survey inputs; named by a verifier; unverified. The reviewer recalls J. Chem. Phys. 148, 124113 (2018), arXiv:1712.03461; confirm before citing | neural mean-force regression | neural §3.10 |
| Vaikuntanathan & Jarzynski 2008 | arXiv:0804.3055 (yes) | escorted switching; the perfect escort | object of §3.14 |
| Vargas, Padhy, Blessing & Nüsken (CMCD) | arXiv:2307.01050 (yes; ICLR 2024) | annealed Langevin, both directions containing a learned $\nabla\phi$; controlled Crooks | §3.14 |
| Albergo & Vanden-Eijnden (NETS) | arXiv:2410.02711 (yes) | $u_{\rm phys}(t)+b_t$; PINN loss; learned $F_t$; no backward kernel | the $u_{\rm phys}+u_\theta$ split for transport |
| He et al. 2025 (FEAT) | arXiv:2504.11516 (yes; NeurIPS 2025) | learned escorts give $\Delta F$ estimators and bounds | current machine-learning standard for $\Delta F$ |
| Wu, Köhler & Noé 2020 (SNF); Doucet et al. 2022 (MCD) | arXiv:2002.06707; 2208.07698 (yes) | fixed backward kernels for stochastic layers; learned backward | fixed-$P_B$ precedent |
| Sendera et al. 2024 | arXiv:2402.05098 (yes; the current listing gives a final author, "Esmeralda S. Whitammer", where arXiv v1 lists Nikolay Malkin, probably a fetch artefact; check the PDF or the NeurIPS 2024 proceedings before citing) | TB with learned $\log Z$; Langevin-parameterised forward; fixed **Brownian-bridge** $P_B$ | lineage of this repository's samplers. Checked against the code in v2: this repository's $P_B$ is a Brownian bridge pinned at $x_0=0$, with a learned bounded correction (§3.19 item 7) |
| Zhang et al. 2024 (DGFS); Berner et al. 2025 | arXiv:2310.02679; 2501.06148 (yes) | sub-trajectory balance with a learned log-flow; discrete-to-continuous equivalences | $x$-dependent $\log Z$; finite-dt theory |
| Nüsken & Richter 2023, "Interpolating between BSDEs and PINNs" | arXiv:2112.03749 (yes, abstract) | the "diffusion loss": Itô increments supply the Laplacian; window-length trade-off; committor examples | Lemma F and the local-loss discussion |
| Richter et al. 2020 (VarGrad); Malkin et al. 2023, "GFlowNets and variational inference" | named by the reviewer | on-policy TB or log-variance gradient equals the reverse-KL gradient; off-policy it does not | Lemma D's on-policy result |
| Hartmann, Schütte & Zhang 2016 (Nonlinearity); Legoll & Lelièvre 2010 | listed in survey 1; Legoll & Lelièvre named by the bias verifier; neither fetched | optimal control reduced to CVs; effective dynamics | prior art for novelty claim 5 |
| Pigeon, Stoltz & Lelièvre 2026 | arXiv:2607.21425 (yes, abstract, single fetch) | compares committor objectives on Müller–Brown | bears on novelty claim 4 |
| Hou, Wang, Ambekar, Raginsky & Milenkovic 2026; Petersen, Lichtinger & Covino 2026; Negedly, Falkner, Coretti & Dellago 2026 (GenAIMMD) | arXiv:2610.10054; 2608.02536; 2609.20461 (yes, abstracts only, single fetch) | Hou: exit-time optimal control, committor from Koopman eigenfunctions, controller by an RKHS quadratic program, not a neural path loss. Petersen: variational committor and rate from trial functions that violate the boundary conditions. Negedly: AIMMD committor plus a conditioned Boltzmann generator for shooting points | Hou: a stopped-control comparator for §3.4 (no path training). Petersen: a non-path baseline (§3.15). Negedly: background for rung 4's shooting comparison |
| Strahan et al. 2023; Mitchell & Rotskoff 2024 | listed in survey 1, not fetched | semigroup committor losses | REACT's comparator; rung 7 |
| Chetrite & Touchette 2015; Thin et al. 2021; Geffner & Domke 2021; Zhang et al. 2021 | named by the reviewer | generalised Doob transform; annealed importance sampling with unadjusted-Langevin backward kernels | §3.9; the EM defect $C_{dt}$ (novelty claim 1) |

**Already done.**
- *Committor and transition paths.* The optimum ($\sigma^2\nabla\log h$, and $\sigma^2\nabla\log q$ when stopped), together with reverse-KL, forward-KL, log-variance/TB, value-matching and Lagrangian training (rows 3–13).
- *Bias and CVs.* The ergodic-control bias potential (as an eigenproblem); VES; ABF.
- *Transport.* The escort and shared-backward constructions; Jarzynski and Crooks.
- *Theory.* The conservative Doob bias.

**Where the owner's framing adds something.**
1. *Hygiene of the fixed Langevin backward kernel.* The $\pi$ factor, the last-step factor, the exact EM defect $C_{dt}$, and the stopping caveat. Useful, but not a new optimum. The ULA-backward annealed-importance-sampling literature already works with the same EM time-reversal defect (Table 6).
2. *A per-start $\log Z$ as a second committor readout*, with a free consistency check against the drift. It is close to REACT's value function and DGFS flows; the equivalence is a CONJECTURE.
3. *A quantified finite-dt picture* (Table 2, Lemma D's off-policy bias, Lemma G). The surveys found no paper that quantifies these effects for committor or bias readouts.
4. *A matched force-call comparison*, narrowed to what no source covers: TB or log-variance under a fixed Langevin backward, together with non-path baselines, measured against barrier height. REACT's overdamped toy comparisons (value matching against detailed-balance, backward-Kolmogorov and semigroup losses, with finite-element ground truth) are matched on energy evaluations; simulated time is used only for alanine and chignolin. Pigeon, Stoltz & Lelièvre 2026 also compare committor objectives on Müller–Brown. This claim depends on REACT's Appendix F, which was not read.
5. *A CV learned as a bottleneck inside $u_\theta$ under a path loss.* Open as a trained method. The theory of CV-reduced optimal control exists (Hartmann, Schütte & Zhang 2016; Legoll & Lelièvre 2010).
6. *Transfer across $\beta$ and across systems.* Open; TPS-DPS states that it does not generalise across systems and state pairs. Generalisation across starts is standard and is not claimed.
7. *The owner's proposal (§3.19).* A start-conditioned sampler with a Boltzmann endpoint, its bias anchored to physical Langevin, read for physical paths and rates. Not found by the novelty search (Table 5), within the coverage limits stated there.

**Where it re-derives known work.**
- The committor and Doob optima.
- Mean-force regression (ABF).
- VES.
- The escort and controlled-Crooks transport.
- The ground-state bias.
- The tied stopped TB is probably REACT's value-matching identity used on-policy (CONJECTURE). Its identity is Nüsken & Richter's diffusion loss (Lemma F).
- The lumpability criterion of §3.2 (Dynkin; Rogers–Pitman).
- The on-policy equivalence of TB and reverse-KL gradients (VarGrad; Malkin et al. 2023).

---

## 5. Toy problems and ground truth

**Table 7.** Toys, what each tests, and the validated ground truth. Values are from `energy_sampling/SCRATCH/nsde_design_nov24/toys/gt_solver.py` and the derivers' grids. Rates follow the §2 notation ($\nu_{AB}$ flux, $k_{A\to B}$ rate constant). "Solver error" is the measured discretisation error of the reference values.

| system | dims | parameters | tests (Table 1 rows) | ground truth (key values) | solver error |
|---|---|---|---|---|---|
| symmetric double well $U=(x^2-1)^2$ | 1 | $\beta\in\{3,5,8\}$; $A=\{x\le-1\}$, $B=\{x\ge1\}$ | null, fixed horizon, window, stopped, tied, regression, rates, Dirichlet | $\beta=5$: $q(-0.5)$ = 2.5518e-2; $\nu_{AB}$ = 2.741e-3; $\lambda_1$ = 1.110e-2, $\lambda_2$ = 3.289; MFPT(−1→1) = 182.4 | quadrature; finite-difference order 2.0, 1.06e-7 at $h$ = 1.25e-3 |
| asymmetric double well $U=(x^2-1)^2+0.25x$ | 1 | $\beta=5$; $A=\{x\le-1.0299\}$, $B=\{x\ge0.9671\}$ | as above, without symmetry coincidences | $q(0)$ = 0.37942, $q(-0.3)$ = 6.015e-2, $q(-0.5)$ = 1.124e-2, $q$ at barrier top = 0.4835; $\nu_{AB}$ = 1.4197e-3; $k_{A\to B}$ = 1.552e-3, $k_{B\to A}$ = 1.666e-2 | 2.6e-7 at $h$ = 1.25e-3 |
| OU $\tfrac12\sum k_ix_i^2$ | any | $k,\beta$ | null, fixed horizon, rates, transport | eigenvalues $-\sum n_ik_i$ (Hermite); hit-0 law $\mathrm{erfc}(x/\sqrt{2v})$; closed-form $u^*(x,t)$; EM stationary variance $1/(\beta k(1-k\,dt/2))$ | analytic; Crank–Nicolson hitting solver 2.8e-6 |
| annulus or shell | 2, 3, any | $U=0$ or radial; $r_1=1$, $r_2=2$ | curved $\partial A$ with singular drift | $q=\ln(r/r_1)/\ln(r_2/r_1)$ in 2D | $O(h)$ with staircase sets (8.5e-3); $O(h^2)$ with exact boundary data |
| rotated double well | 2 to 1000 | catalogue: $(s^2-1)^2+\tfrac12\sum k_iy_i^2$, $k=4$, $\beta=4$, plus a variant with one $k_i=0.01$ (slow non-reactive mode). The CV family used $V=(\xi^2-1)^2+0.2\xi$, $\kappa=4$, 30° | dimension scaling; bottleneck CV; slow mode versus committor | $q=q_{1D}(e\cdot x)$ exactly; $F(s)$ exact; $\lambda_1$ = 2.9780e-2 ($\beta=4$, catalogue) | 2D: $\rho_R$-weighted RMS 7.3e-5 at $h$ = 0.01 |
| Müller–Brown | 2 | standard; $\beta\in\{0.05,0.1,0.2\}$; discs $r=0.1$ at the two deep minima | stiff forces; intermediate basin; non-factorisable CV | $\beta=0.1$: $q$(saddle S1) = 0.325; $\nu_{AB}$ = 4.915e-3; $\lambda_1$ = 0.2000, $\lambda_2$ = 41.3 (gap); MFPT = 198.3; EM stable only for dt < 4.9e-4 | converged to 4 digits ($h$ = 0.02 to 0.005); current conserved to 1e-13 |
| three-hole | 2 | Metzner–Schütte–Vanden-Eijnden; $\beta\in\{1.67,3,6.67\}$ | channel coverage; transfer across $\beta$ | upper-channel share 0.348, 0.401, 0.629; $\nu_{AB}$ = 1.877e-2, 7.782e-4, 9.136e-8. The $\beta=1.67$ flux coincides with the symmetric well's at $\beta=3$ (1.8769e-2); the reviewer's independent finite-volume code (1.8770e-2 at $h$ = 0.01, `energy_sampling/SCRATCH/nsde_design_nov24/critic/threehole_check.py`) confirms it is real, not a copy error | flux equals Dirichlet form to 1e-14 |
| two-channel ring | 2 | $\kappa(r^2-1)^2+2y^2+cy$, $\kappa=5$, $\beta=3$ | mode collapse | split 0.5000 at $c=0$ (exact by symmetry); upper share 0.187 at $c=0.3$ | grid |
| entropic neck | 2 | $(x^2-1)^2+\tfrac12\omega(x)^2y^2$, $\rho=5$, $w=0.2$, $\beta=3$ | bias versus committor; entropic barrier | analytic $F(x)$; $\Delta F$ = 1.5365; $\nu_{AB}$ = 9.31e-3 | 1–2% (neck resolution) |
| periodic 2D toy | 2 | $2\cos2x_1+0.5\cos x_1+2(1+0.5\sin x_1)(x_2-\cos x_1)^2+0.25x_2^4$, $\beta=2$, $s=x_1$ | mean-force regression, VES, ergodic flat bias | $F(x_1)$ by quadrature on 2048×4001 points; $\beta$ range 9.87 | quadrature; $F'$ to 1.7e-12 |
| Gaussian pairs | 1, 2 | 1D: $x^2/2\to2(x-1.5)^2$, $\beta=2$; 2D: changing covariance plus a pull | transport rows | $\log Z_1/Z_0$ = −0.693 (1D), 2.97582 (2D); closed-form moments | analytic |
| alanine dipeptide (vacuum, 300 K) | 66 | standard force field | final rung | no analytic truth; shooting committors (±0.05 at $q=0.5$ with 100 shots), MSM rates. Recollection, not checked | MD |
| 1D underdamped double well (needed by rung 14; not in the catalogue) | 1 position + 1 velocity | to be chosen | stopped committor, in phase space (rung 14) | phase-space committor by a finite-difference solve (as in REACT, per the reviewer); not built | – |
| entropic neck with $d-1$ orthogonal coordinates (needed by rung 9; not in the catalogue) | $d$ | $(x^2-1)^2+\tfrac12\omega(x)^2\lvert y\rvert^2$ | stopped committor; bottleneck CV (rung 9, non-separable $d$-scaling) | $q(x,\lvert y\rvert)$ by a 2D grid in $(x,r)$ with the radial term $(d-2)/(\beta r)\,\partial_r$; not built | – |

**Solver.** `gt_solver.py` has sections `1d | 2dval | 2dtoys | sde`. Its output folder is hard-coded to the original scratchpad path, so edit `OUT` before rerunning.
- It builds the reversible square-root-approximation (SqRA) generator $L_{ij}=e^{-\beta(U_j-U_i)/2}/(\beta h^2)$ with reflecting walls.
- It solves committor and MFPT problems directly with sparse solvers, finite-horizon $H$ by Crank–Nicolson with a Rannacher start, and eigenpairs by shift-invert.
- It also computes the CV-restricted variational rate and the reactive-current split.
- Validations:
  - order 2 against quadrature (1D);
  - $k$ and MFPT to 1e-6 relative;
  - OU eigenvalues at order 2;
  - closed-form hitting at order 2;
  - annulus at $O(h^2)$ with exact boundary data;
  - rotated well against the exact 1D committor;
  - Müller–Brown grid convergence and current conservation;
  - a Doob importance-sampling check with lowest variance at factor 1;
  - OU bridge correction: −11.5% bias falls to +0.5%.
- Runtime: under 50 s per section, peak 0.6 GB.
- Arrays are in `gt_*.npz` in the same folder.

**Metrics.** Use these, and always score against the floor the exact optimum reaches at the same dt.
- Committor error weighted by the reactive density $m_R=\pi q(1-q)$ (plus the $\pi$-weighted and uniform versions).
- Variational rate excess $k(\hat q)/k-1=\beta^{-1}\int\pi\lvert\nabla(\hat q-q)\rvert^2/\nu_{AB}\ge0$.
- Path KL of the drift, $\frac\beta4\mathbb E_Q\int\lvert u_\theta-u^*\rvert^2$.
- ELBO gap $\log h_\epsilon(x_0)-\mathbb E[w]$ (nats), rung 1's main criterion.
- Relative weight variance and effective sample size per force call.
- $\log Z$ error, and consistency between the $\log Z$ readout and the drift readout.
- B-exit fraction against $q/(q+\epsilon(1-q))$.
- Iso-committor histogram on $\hat q=0.5$, and $q$ at the saddle (0.325 on Müller–Brown, not 0.5).
- CV scores (§3.16).
- RMS free-energy error over the region within 10 kT of the minimum, after removing the best constant.
- Channel split.
- Error in $\log k$.
- Generalisation gap at held-out $\beta$, $x_0$ and $d$.

---

## 6. Experiment ladder

**Conventions.**
- **One addition per rung.** Each rung adds one thing to the method. Comparators (baselines, exact controls) are part of a rung's measurement, not additions to the method.
- **Reference at the training dt.** Every trained rung first evaluates the exact control at its training dt (rung 0, check (ii)). Each criterion is scored against the floor that the exact control reaches there, not against zero or the continuum.
- **Two-dt protocol.** Every trained rung runs at dt and dt/2. A criterion passes only if it passes at both dts, each against that dt's exact-control reference. The shift of each readout between the two dts must also agree with the exact control's shift within 2 standard errors (SE).
- **Budget.** Each rung has a training force-call budget (paths × steps). Evaluation calls are reported separately and not counted. If a rung's criteria are not met within its budget, the rung counts as failed: stop and diagnose before adding the next rung. The budgets are proposals (working assumptions) sized from the single-seed runs of §3.5–3.6; the owner may reset them.
- **Evaluation size.** ELBO and exit-fraction criteria use at least 2e4 fresh paths. At rung-1 settings that gives an SE of $\mathbb E w$ of about 6e-4 nats.
- **Drop rule from rung 7 on.** A path arm that does not beat the best non-path baseline at matched force calls on a toy is dropped for that toy (§3.15).

**Table 8.** The committor ladder. Each row is a rung: what it adds, the toy, the criterion it must meet, the training budget in force calls, and what happens if it fails. Toy values are from Table 7. "Floor" is the exact control's value at the same dt.

| rung | adds (one thing) | toy | must show | training budget (force calls) | if it fails |
|---|---|---|---|---|---|
| 0 | pre-flight checks, no training | OU; 1D asymmetric well | checks (i)–(v) below | none (seconds of CPU) | fix the code; nothing later can be interpreted |
| 1 | on-policy TB; free drift $u_\theta(x)$; scalar $\log Z$; point start | 1D asymmetric well, $\beta=5$, $x_0=0$, $\epsilon=0.01$ | ELBO gap ≤ 0.007 nats; B-exit fraction within 2 SE of 0.984; Var $\delta$ ≤ 2× floor | 2e7 at dt = 2e-3; 4e7 at dt = 1e-3 | stop |
| 2 | spread start law with a per-start $\log Z_\phi$ head (free arm) | same; starts uniform on $D$ | $\max\lvert\hat q-q\rvert\le0.01$ where $q>0.01$, on held-out starts; $\sigma^2\partial_x\log Z_\phi=u_\theta$ within 2 SE | 4e7 | stop |
| 3 | tied single network $h_\theta$ (§3.5) in place of the free arm; two-sided tied committor (§3.18) as an optional second arm | same | rung 2's accuracy; force calls against rung 2's, over 3 seeds | rung 2's actual spend | keep the cheaper arm |
| 4 | barrier height, $\beta\in\{3,5,8\}$ | 1D asymmetric well | rung 2's accuracy at each $\beta$ | at each $\beta$: the cost of brute-force unbiased shooting to the same accuracy, computed from the exact $q$ and the grid's mean exit time | stop at the first $\beta$ where it loses to brute force |
| 5 | replay buffer (off-policy) | 1D asymmetric well, $\beta=5$ | free arm: measured bias matches Lemma D's law within 2 SE. Tied arm: no bias beyond noise at both dts | rung 3's spend | report the law; the failing arm stays on-policy |
| 6 | learned forward-kernel covariance (§3.18) | same | a lower variance floor, or a lower rung-5 bias, at ≤ 1.2× the calls | rung 3's spend | drop the covariance |
| 7 | competing losses at matched calls | 1D asymmetric well, $\beta\in\{5,8\}$ | a loss ordering stable over 3 seeds and both $\beta$ | rung 3's spend, per arm | if REACT-style value matching wins at every $\beta$, revisit D1 |
| 8 | dimension, separable | rotated well, $d$ = 2, 10, 100, 1000 | angle < 2° and rung 2's $q$ accuracy; fitted exponent $\alpha$ in calls $\propto d^\alpha$, with $\alpha\le1$ (proposed); non-path CV baselines at matched calls | $4\times10^7\cdot d$ per arm at each $d$ | stop the dimension claim if no arm reaches $\alpha\le1$ within the cap |
| 9 | dimension, non-separable | entropic neck with $d-1$ orthogonal coordinates (Table 7; solver not built), $d$ = 2, 10, 100 | rung 2's $q$ accuracy; $\alpha\le1$ as in rung 8 | $4\times10^7\cdot d$ per arm at each $d$ | stop the dimension claim |
| 10 | curved boundaries: distance-function boundary conditions, bridge killing on curved sets | 2D annulus (exact $q$) | $\max\lvert\hat q-q\rvert\le0.01$; readout bias $O(dt)$ under the two-dt protocol | 4e7 | enlarge $A$ by $O(\sigma\sqrt{dt})$ instead of bridge killing |
| 11 | several channels | ring at $c=0$, then $c=0.3$ | channel split within 2 SE of 0.5000 and of 0.187 | 4e7 per $c$ | fix mode coverage before rung 12 |
| 12 | nonlinear $s_\theta$, on a toy whose $q$ no linear CV represents | Müller–Brown, $\beta=0.1$, disc sets, dt < 4.9e-4 | Dirichlet ratio ≤ 0.05; $q$(S1) within 0.02 of 0.325; iso-committor histogram on $\hat q=0.5$ centred within 0.05; beats the best non-path baseline at matched calls (drop rule) | 4e7 per arm | keep linear CVs only |
| 13 | $\beta$ as a network input | three-hole: train at $\beta\in\{1.67,6.67\}$, test at 3 | error at the held-out $\beta$ ≤ 2× the in-distribution error | 2× rung 12's actual spend, at most 8e7 | drop the transfer claim |
| 14 | underdamped dynamics | 1D underdamped double well (Table 7; solver not built) | rung 2's criteria against the finite-difference phase-space $q$ | 4e7 | stop before the real system |
| 15 | first real system | alanine dipeptide in vacuum, underdamped Langevin | $\hat q=0.5$ histogram centred within 0.05; rate within 2× of a reference fixed before the run; ground-truth-free certificate (Lemma G) passes | 1.6e8 energy evaluations (10× TPS-DPS's 1.6e7) | stop |
| B1 | bias branch (only if D1 includes it) | periodic 2D toy, a slow-orthogonal-barrier variant, entropic neck | the per-force-call advantage of mean-force regression over VES, OPES and metadynamics, at matched calls, survives a slow $x_\perp$ | 1.28e6 per method per toy (the largest budget in Table 4b) | report it; promote nothing |

**Rung 0: pre-flight checks** (no training; seconds each).
- (i) Null case with the exact OU kernel: $\lvert w-\log Z_U\rvert\le10^{-13}$.
- (ii) Evaluate the analytic $u^*_\epsilon$ at each training dt (2e-3 and 1e-3). This gives the residual-variance floor, the EM value of $\log h_\epsilon$, the ELBO gap and the exit fractions that rung 1 is scored against.
- (iii) Log the owner-form residual next to the forward one. With $\tilde R=R_\epsilon\tilde\pi$ they must differ by $\beta U(x_0)+C_{dt}$, path by path.
- (iv) Measure the effect of bridge killing on $q_{dt}$ near $A$ with the exact control and the lattice solve. This test moves here from v1's rung 2: a trained readout cannot resolve a 4% effect, since the single-seed tied run's readout error at $x=-0.8$ was −8% (0.00269 against 0.00293).
- (v) Evaluate the spurious controls against the rung-1 criteria. The reviewer has done this for $u_0$ (Table 9).

**Table 9.** The rung-1 criteria at the exact control and at the spurious control $u_0=\sigma^2\nabla\log q$. Asymmetric well, $\beta=5$, $x_0=0$, $\epsilon=0.01$, dt = 2e-3, $10^5$ EM paths, one seed (reviewer's check: `energy_sampling/SCRATCH/nsde_design_nov24/critic/rung1_check.py`, results in `rung1_check.json`). The reference values are $\log h_\epsilon(0)=-0.9529$ nats and a continuum B-exit fraction of 0.9839. Compare the rows within each column. With bridge killing, only the ELBO gap and the B-exit fraction separate $u_0$ from $u^*_\epsilon$; under plain monitoring Var $w$ also separates them (310.7 against 0.0059 nats²).

| control (boundary monitoring) | IS $\log Z$ (nats) | $\mathbb E[w]$ (nats) | ELBO gap $\log h_\epsilon(0)-\mathbb E[w]$ (nats) | Var $w$ (nats²) | B-exit fraction |
|---|---|---|---|---|---|
| exact $u^*_\epsilon$ (bridge killing) | −0.9536 | −0.9565 | 0.0036 | 0.0059 | 0.98388 (A: 0.0161 ± 0.0004) |
| exact $u^*_\epsilon$ (plain monitoring) | −0.9543 | −0.9572 | 0.0043 | 0.0059 | 0.98417 (A: 0.0158 ± 0.0004) |
| spurious $u_0$ (bridge killing) | −0.9547 | −0.9726 | 0.0197 | 0.0085 | 0.99995 |
| spurious $u_0$ (plain monitoring) | −0.9631 | −1.0302 | 0.0773 | 310.7 | 0.99998 |

At $u_0$ with bridge killing, the IS $\log Z$ is within 0.002 nats of the target and the variance is 1.4× the floor. So the v1 criteria (IS $\log Z$ within 0.01 nats, Var within 2× the floor) would have passed it. The plain-monitoring bias of the exit fraction is not resolved at this $x_0$, so the 0.984 criterion is safe here.

**Rung 1: the literal simplest experiment.**
- **Setup.**
  - Toy: 1D asymmetric double well, $\beta=5$, stopped on entering $A$ or $B$.
  - Start: point start $x_0=0$ (non-rare: $q=0.379$).
  - Reward: $R_\epsilon=\mathbf 1_B+0.01\cdot\mathbf 1_A$.
  - Model: free MLP drift $u_\theta(x)$, time-independent, with a scalar $\log Z$.
  - Training: on-policy TB with detached samples and the forward-reference ratio, at dt = 2e-3 and again at dt = 1e-3. Cap the path length at $T_{\max}$, and report the fraction of capped paths.
  - Boundary monitoring: bridge killing on, so the B-exit reference is 0.9839 at dt = 2e-3 (with plain monitoring the exact control gives 0.9842; Table 9).
- **Must demonstrate, at both dts:**
  - (a) ELBO gap $\log h_\epsilon(0)-\mathbb E[w]$ ≤ 0.007 nats. That is about 2× the exact control's gap, and it rejects $u_0$ (0.0197) by a margin of about 20 SE;
  - (b) B-exit fraction within 2 SE of the exact control's value at that dt (0.9839 at 2e-3);
  - (c) Var $\delta$ ≤ 2× the floor.
- **Recorded, not tested:**
  - the IS $\log Z$ (a readout);
  - the Q-occupation-weighted drift error against force calls;
  - the two ground-truth-free certificate quantities of Lemma G. With bridge killing, the ratio $(\log\hat Z_{\rm IS}-\mathbb E w)/(\tfrac12\mathrm{Var}\,w)$ computed from Table 9's run was 1.00 at $u^*_\epsilon$ and 4.2 at $u_0$ (about 4e-4 at $u_0$ under plain monitoring). Rung 1 tests whether it tracks convergence, before rung 15 relies on it.
- **Decision:** the force calls needed to meet (a)–(c). Beyond 2e7 (4e7 at dt = 1e-3) the rung has failed.

**Rung 2.** The bridge-killing test moved to rung 0 (iv). The tied arm moved to rung 3. For scale: the tied arm reached a maximum error of 0.009 at 2.2e7 calls on the symmetric well (§3.5).

**Rung 4.** "Brute force" is unbiased shooting from the same start law. Its cost to the target accuracy follows from the exact $q$ (binomial variance) and the grid's mean exit time. This is the owner's first-priority number, and it is the comparison not found in the literature.

**Rung 5.** The prediction is split by arm. Lemma D's bias, about $\sigma^2dt\,H(u^*-u_b)$ per step, applies to the free drift with a $\log Z$ head. The tied arm's zero set does not depend on the behaviour law in continuous time (Lemma D), so for it any bias must shrink with dt.

**Rung 7 arms.**
- forward-KL regression on accumulated B-hits (§3.7);
- a semigroup committor loss (REACT's baseline);
- REACT-style value matching, if it can be implemented from the paper;
- a Dirichlet baseline (§3.15) fed by umbrella windows along $x$ with WHAM weights, with every force call counted.

**Rung 8.** This rung was mis-specified in v1.
- In the separable rotated well, $q=q_1(e\cdot x)$ for every $k_i$, and for $u\parallel e$ the EM residual is exactly the 1D one. So the finite-dt floor and the per-path force calls at $u^*$ do not depend on $d$, and the rung measures only the cost of learning to ignore $d-1$ directions.
- The slow-orthogonal variant is dynamically decoupled from $\xi$ and cannot change the stopped committor. It discriminates only the fixed-horizon and spectral formulations (§3.3, §3.9), so it moves to the D1(b) branch.
- Arms: an unstructured drift, and the linear bottleneck $\sigma^2\nabla\log f(w\cdot x)$.
- Non-path CV baselines at matched calls: the Dirichlet committor with a linear $s$, the restricted Rayleigh quotient, and TICA or VAMP on unbiased trajectories.
- Deciding measurement: force calls against $d$ per arm, and $S(s)$. The bottleneck stays only if it lowers calls or improves held-out error.

**Rung 9.** Take $U=(x^2-1)^2+\tfrac12\omega(x)^2\lvert y\rvert^2$ with $y\in\mathbb R^{d-1}$. The committor depends on $\lvert y\rvert$, because the force along $x$ contains $\omega\omega'\lvert y\rvert^2$, so $L[q(x)]\ne0$ (reviser). The ground truth needs a 2D grid in $(x,r=\lvert y\rvert)$, which has not been built.

**Rung 10.** The exactly solvable annulus comes before Müller–Brown. That way the curved-boundary pieces (risk 6, and bridge killing on discs) are first exercised where the answer is exact and the forces are not stiff.

**Rung 13.** Training at two temperatures and testing at the intermediate one (upper-channel share 0.401) is interpolation. Extrapolation across the channel-preference flip — train at $\{1.67,3\}$, test at 6.67, where the share goes from 0.348 to 0.629 — is reported as a stretch, not as a pass criterion.

**Rung 14: underdamped dynamics.** Rung 15 will almost certainly use underdamped Langevin, so this rung must come first.
- The noise and the control act only on velocities, so Girsanov is hypoelliptic: the path density ratio uses velocity increments only.
- Euler–Maruyama position transitions are degenerate. Use a splitting integrator whose stochastic part acts on velocities only, and score only those increments.
- The time reversal used as $P_B$ needs a momentum flip.
- The committor is defined on phase space.
- REACT's uniqueness guarantee does not cover this case.

**Rung 15.**
- Specify the friction and the integrator before the run.
- Name the rate reference before the run, for example an MSM built on long unbiased MD with the same force field and temperature.
- Use published training budgets as a scale, not a benchmark, because the targets differ: TPS-DPS 1.6e7, Doob's Lagrangian 3.8–5.1e7.

**B1: bias branch.** Mean-force regression (§3.10) inside the $u_{\rm phys}+u_\theta$ frame, against VES, OPES and metadynamics at matched force calls. The slow orthogonal barrier is where mean-force methods are expected to fail silently. Deciding measurement: the $\beta$-RMS error of $F$ against force calls, with the metric floor (0.011 kT) stated.

**Table 10.** The branch for the owner's proposal (§3.19). It is independent of the committor ladder until rung 14. Each row has the same columns as Table 8. Nothing in it has been run, and the owner has not given a go.

| rung | adds (one thing) | toy | must show | training budget (force calls) | if it fails |
|---|---|---|---|---|---|
| O0 | grid comparison, no training | 1D asymmetric well, $\beta=5$, $A$ = the deeper minimum ($x=-1.0299$), $T\in\{2,5\}$ | per arm: endpoint KL to $\pi$ (nats); Q-weighted L² distance of the drift to the exact minimum-bias $u^*$; KL of the B-landing bridges to the physical bridges (nats) | none (CPU, seconds to minutes) | not a pass/fail rung; its outcome decides D5 by the rule below |
| O1 | smallest trained version: one fixed start $A$, D5's arm (for D5(a): the helper updated by maximum likelihood on forward paths, never by the squared residual), $u_\theta(x,t)$ with a time input | same, $T=2$; $B$ = the basin $\{x>0.0627\}$ | B-landing fraction within 2 SE of the grid $\pi(B)=0.0852$; IS estimate of $P_{\rm phys}(X_T\in B\mid A)$ within 2 SE of the grid $p_T(B\mid A)$; spread of the endpoint-regressed $\log w$ (item 11) ≤ 2× its value at the exact optimum at that dt | 5e7 at dt = 2e-3 (1,000 calls per path); 1e8 at dt = 1e-3 | stop the branch |
| O2 | the start $A$ as a network input | same; training starts across $[-1.2,1.2]$ | O1's criteria at 5 held-out starts; fewer calls in total than 5 separate O1 runs | 2.5e8 (5 × O1's cap at dt = 2e-3) | the amortisation claim fails on this toy |
| O3 | hot-reference anneal, $\beta_{\rm ref}$ raised from 1.67 to 6.67 on a fixed schedule, at the model's $\sigma$ | three-hole | O1-type criteria at $\beta=6.67$; fewer calls than direct training at 6.67 | 1e8 over the whole schedule (proposed) | drop the anneal |

**O0 arms** (all on the same grid).
- *Exact.* The minimum-bias $h$, from a backward Kolmogorov solve with terminal data $\pi/p_T(\cdot\mid A)$. Here $p_T$ comes from a forward Crank–Nicolson solve started from a narrow Gaussian at $A$.
- *(i)* A fixed Brownian bridge plus the KL term, at $c\in\{1,10,100\}$. The optimum $M$ is an h-transform (§3.19, item 9), so the same solver applies.
- *(ii)* The physical bridge with the exact $p_T$. It must reproduce the exact arm, which makes it a check on the code.
- *(iii)* The learned-helper optimum at $c\in\{1,10,100\}$: $g=(\pi/p_T)^{c/(1+c)}$. This assumes the helper reaches the combined-KL optimum, i.e. is trained by maximum likelihood on forward paths. O0 computes optima on a grid, so it cannot detect that a helper trained by the squared residual has no unique optimum (§3.19 item 9); O1 must fix the helper update.

**O0 decision rule** (proposed thresholds). If arm (i) at $c=100$ reaches an endpoint KL ≤ 0.01 nats and a B-landing bridge KL ≤ 0.01 nats at both horizons, a fixed backward is enough: D5(b). Otherwise, if arm (iii) reaches both at some $c\le100$: D5(a). If neither does, report the measured values and stop the branch for the owner's adjudication. Item 7 predicts that arm (i) fails the bridge criterion, because its B-landing paths tend to Brownian bridges. The OU checks are consistent with that (bridge KL 0.711 nats at $c=100$), but OU has no barrier, so they do not decide D5. $B$ is the basin $\{x>0.0627\}$, $\pi(B)=0.0852$.

Two-state estimates (reviser) place both horizons inside the spectral window. Take $1/\lambda_2\approx0.3$, the symmetric-well value, and $1/\lambda_1\approx1/(k_{A\to B}+k_{B\to A})\approx55$.

**O3.** The TPT upper-channel share at $\beta=6.67$ (0.629) is the stationary reactive-path value. The share among finite-$T$ physical paths that land in $B$ needs its own grid computation, which has not been built.

Multi-system conditioning, value claim (2), stays out of scope until O2 passes. The branch would join rungs 14–15 after O2.

---

## 7. Risks and failure modes that could sink the project

1. **Low novelty of the core result.** The committor optimum, stopped training with off-policy reuse (REACT), and TB with learned $\log Z$ for transition paths (TPS-DPS) are all published. If rungs 4–13 show no force-call, finite-dt or transfer advantage, the project re-derives known methods. Rung 7 must include REACT-style and semigroup baselines. REACT is still changing (v4 is dated 23 Nov 2026), and its Appendix F was not read.
2. **Bootstrapping rare events.** From realistic starts, $q(x_0)$ is exponentially small. At $u=0$, batches without B-exits give exactly zero gradient under log-variance, and zero-mean noise under TB with a free scalar $\log Z$ (Lemma G). The runs that escaped were 1D with $q_0\approx10^{-3}$. Beyond start curricula (interface splitting, untested), no remedy is shown.
3. **Spurious on-policy solutions** (Lemma G). A run can converge to the right CV with the wrong profile and readout while its loss looks converged. The IS $\log Z$ does not detect this (Table 9). Without the ELBO and exit-fraction checks it will be misread. The two-sided tied committor (§3.18) might remove the problem.
4. **Finite-dt bias in every readout.** The "converged" answer is the EM chain's: $O(\sqrt{dt})$ at boundaries, the curvature law for sharp targets, heavy tails near $A$. The two-dt protocol (§6) puts a dt comparison into every trained rung. With the forward-reference ratio, dt must shrink with $d$ only when $\lVert\nabla^2\log h\rVert_F$ grows with $d$ (e.g. product targets), not for the separable committor targets; with an EM backward, $C_{dt}$ also grows with $d$ through $U$ (Table 2).
5. **Off-policy reuse at finite dt** (Lemma D). For the free-drift arm the bias follows a known law, measured once at 14% on an OU check far from the optimum. For the tied arm the zero set does not depend on the behaviour law in continuous time. Its size for a realistic buffer is unmeasured (rung 5). If it is large, staying on-policy costs force calls, the owner's first priority.
6. **Boundary construction in $d>1$.** Hard boundary conditions and the floor $\epsilon$ need distance functions to $A$ and $B$, and $\lvert\nabla q(\partial A)\rvert$, which is unknown a priori. This is close to needing the CV before learning it.
7. **CV identifiability.**
   - A CV is identified only where paths go.
   - Classes that cannot represent $q$ give a CV that depends on the policy under reverse KL.
   - Multi-state systems need $k>1$.
   - The slowest mode is not the reactive coordinate when a soft orthogonal mode exists.
8. **Bias branch.** The natural objective is ABF re-derived. Hidden orthogonal barriers bias it without a visible symptom. Its measured advantage comes from one easy toy and holds only from 6.4e4 calls on (the crossover with umbrella sampling plus WHAM lies between 3.2e4 and 6.4e4; Table 4b).
9. **Transport branch.** Jarzynski was cheaper than training on the toy, so only amortisation justifies a trained transport.
10. **Circular evaluation.** Scores weighted by the policy's own occupation pass wrong policies. Use target marginals and held-out starts.
11. **Gap between toys and real systems.** Per-force-call advantages in 1D and 2D may not survive MLIP costs. Alanine has no exact committor; shooting noise is about ±0.05.
12. **Underdamped dynamics.** The real system needs them. Hypoelliptic Girsanov, degenerate position transitions and the momentum flip in $P_B$ are untested here (rung 14), and REACT's uniqueness result does not cover them.
13. **The owner's proposal (§3.19).**
    - The owner expects it to be very expensive. Amortisation over starts and systems is the cost, and it is unmeasured (O2).
    - A learned helper leaks its training error into the bias before convergence; the endpoint-only certificate (item 11) is the only detector.
    - A helper trained by the squared TB residual converges to a wrong zero-residual point, since every helper has one. The endpoint-only certificate flags it (OU: conditional $\log w$ variance 1.49 nats² at $M(\text{BB})$, $c=10$, against about 1e-27 at the optimum). Train the helper by maximum likelihood on forward paths.
    - Large $c$ ill-conditions training.
    - The Boltzmann endpoint is a target, not a guarantee.
    - Reweighted rates are unbiased, but their variance grows with the distance from the optimum. Even at the optimum the relative variance is about $(1-\pi(B))/\pi(B)$, and it can exceed direct sampling for a non-rare $B$ (Table 5b).
    - A hot reference leaves an extensive cooling cost in the bias while it is hot (item 13).
    - Novelty rests on a web search with no citation crawl (Table 5), and TITO-style propagators are the comparison reviewers will ask for.

---

## 8. Owner decisions needed

**D1. Which target first?**
- **Recommended:** (a) the committor by stopped conditioning (Table 8). It has exact toys, a time-independent drift, the per-start readout, and the clearest open comparison.
- Alternatives:
  - (b) **The literal fixed horizon** (terminal reward, fixed $T$). Here $P_B$ is a genuinely normalised backward kernel, and the method yields $\varphi_1$ and rates.
    - It needs a time input and gives only a soft committor control.
    - No trained-model evidence exists for it: §3.3 and §3.9 are grid and spectral computations only.
    - Its waiting cost has not been compared at matched start, target and accuracy. v1's "5,000 against about 1,100 calls per path" compared different problems (§3.2).
    - An Exp($\gamma$) horizon (§3.18) would remove the time input.
  - (c) **Free energy / bias first.** Mean-force regression is quick and had the lowest error per force call on one toy from 6.4e4 calls on (Table 4b). It is ABF, so novelty is low, and its open test is the hidden barrier.
  - (d) **$\Delta F$ transport.** Covered by CMCD, NETS and FEAT; Jarzynski was cheaper on the toy.
  - (e) **The owner's proposal** (Table 10). Its first step, O0, needs no training and does not compete with (a) for compute. Training (O1 onward) waits on D5 and on the owner's go.
- Default if unanswered: (a).

**D2. What should "the CV" mean?**
- **Recommended:** (a) the committor's level sets for questions with $A$ and $B$; the slow mode $\varphi_1$ when there are no $A$ and $B$.
- Alternatives:
  - (b) **Always $\varphi_1$.** It needs no $A$ and $B$, and it is what the fixed horizon reaches.
    - It differs from affine($q$) by 2.06e-3 in sup norm on the 1D well at $\beta=5$, by 1.8e-3 as a relative $L^2(\pi)$ residual on Müller–Brown, and by 3.0e-2 (relative $L^2(\pi)$) on the rotated-well $\xi$ chain at $\beta=1$. The gap shrinks as metastability grows.
    - It is not reactive when a soft orthogonal mode is slower.
  - (c) **Always the committor.** It needs $A$ and $B$, which in $d>1$ already require a coordinate.
- For the owner's proposal, the $A\to B$ drift is "a" CV-bias object: finite-time iso-committor level sets (§3.19 item 5). It is consistent with (a) inside the spectral window.
- Default: (a).

**D3. Is a non-learned backward kernel a hard constraint?** In every option below, the reference process itself is never learned.
- **Recommended:** (a) yes, for the targets of §3.1–3.15, split by readout:
  - *h-type readouts* (committor, hitting, rates): the forward-reference density ratio with a per-start $\log Z$. For stopped targets the owner's $P_B$ enters only as this density ratio. It is not a normalised backward policy over paths of variable length, and its normalisation is never used (Lemma C).
  - *Z-type readouts* ($\Delta F$): the normalised Langevin backward, which gives Lemma E's exact importance-sampling identity at every dt.
  - *The owner's proposal* (§3.19): here the constraint has a price. With any fixed, tractable backward, the minimum-bias object is out of reach unless $\log p_T(x_T\mid A)$ is supplied (items 6–7). A Brownian bridge gives the PIS-type transport, whose B-landing paths are not physical. Keeping the backward fixed there means D5(b) or D5(c); reaching the minimum-bias object needs D5(a).
- Alternatives:
  - (b) **The literal reversed-EM kernel for every readout.** Same optimum, plus the $O(dt)$ term $C_{dt}$ in h-type readouts, and one extra force call per path.
  - (c) **Allow a learned backward** (shared escort) for transport. Exact at finite $T$, but it leaves the owner's frame, and the escort has a large gauge (a 97% gauge deviation was observed).
- Default: (a); for the owner's proposal, D5.

**D4. Where should the project live?** Two facts bear on this.
- *A memory lead.* A session memory records an owner statement of 2026-10-27: "a central training script with modular components is highly preferable". It was said after a parallel INN trainer kept turning up `train.py` mechanisms it lacked. This is a navigation lead, not repository authority, and the owner should confirm whether it applies here.
- *The code* (checked in v2). This repository's sampler uses a Brownian-bridge $P_B$ pinned at a deterministic $x_0=0$, with a learned bounded correction (`model.learn_pb: true`, `model.pb_drift_range: 0.4`). It has a replay branch (`replay_loss_coeffs`). Rungs 0–4 and 6–7 need the opposite: a physical Langevin reference, on-policy sampling, and stopped paths of variable length (rung 5 adds a plain replay buffer).

Options:
- **Recommended:** (a) split by stage.
  - Rungs 0–13 and O0–O1 are standalone CPU scripts with their own tests and no imports from crystal code, in their own repository or a top-level directory of this one (v1's options (a) and (c)). They are measurement instruments on toys with exact answers, with at most a plain replay buffer (rung 5) and a scalar $\beta$ input (rung 13), but no LR control, buffer servo or system conditioning, so they are not a parallel trainer. Their cost is a small TB loop, logging and run naming, about the size of the scripts already written.
  - From the first rung that needs `train.py`'s machinery — O2 onward, rung 15, or any crystal use — the winning loss goes into `train.py`'s policy and reference slots. Only what is intrinsically new is adapted there: a physical reference, stopped variable-length paths, a helper backward pinned at $A$.
  - Consequence: the scripts must not grow trainer features. When they would, port instead.
- Alternatives:
  - (b) **Inside `train.py` from the start.** It reuses months of crystallised fixes. But its defaults are the opposite of what rungs 0–7 need, and that code is crystal-specialised and governed by `mk_dev.yaml` and the repository's change discipline. Toy experiments would then carry that discipline's cost.
  - (c) **A standalone package for everything, in its own repository.** Clean and CPU-first. But once training machinery is needed, it becomes the parallel trainer that the owner's 2026-10-27 statement argues against.
- Default: (a). If the owner's central-trainer preference is meant to cover toy measurement scripts too, choose (b).

**D5. For the owner's proposal: a fixed backward or a learned helper?**
- **Recommended:** (a) run O0 first. If O0 confirms item 9, use a learned helper pinned at $A$, trained in alternation with the drift on the combined reverse KL (Table 5c). Freeze the helper while the forward converges, then refit it; joint gradient training falls into the soft valley and slows by a factor of order $c$. The helper's update is maximum likelihood on the forward's own paths (ASBS corrector matching), never the squared residual, which has a zero-residual point for every helper (item 9). Give the helper the Brownian-bridge variance schedule near the pin. The drift may use the squared residual on-policy. The physical reference stays fixed and non-learned, and $c$ rises on a fixed, finite schedule. Anchoring the helper to physics directly (H5 with the global start condition, or pre-training on physics paths) is the alternative to relying on the weak KL anchoring at large $c$.
  - It is the only option whose optimum is the minimum-bias object with a Boltzmann endpoint (that optimum PROVED on the OU chain).
  - The target stays defined by the physics, since the helper's unique optimum under that update is the physical bridge.
  - It leaves the strict non-learned-backward frame the owner preferred. Helper error leaks into the bias before convergence, which the endpoint-only certificate (item 11) detects, and large $c$ ill-conditions training.
- Alternatives:
  - (b) **A fixed Brownian bridge pinned at $A$, plus the KL term.** The backward stays non-learned and tractable.
    - The optimum is the blend $M$: its bridges mix free diffusion and physics, and its endpoint is only partly Boltzmann.
    - As $c\to\infty$ it becomes the PIS-type transport, whose B-landing paths are Brownian bridges.
    - $A\to B$ probabilities still come out unbiased by reweighting (item 12), but the paths are not physical. Its reweighting variance need not be worse: the minimum-bias object is not the variance-optimal proposal for a given $B$ (on OU, $M(\text{BB})$ at $c=1$ gave 0.598 against 0.655 per sample; Table 5b).
  - (c) **Drop $\log p_T$**, so the target is $\pi$ times the physics path density. Non-learned and tractable, but the endpoint is $\propto\pi\,p_T(\cdot\mid A)$, so the rare-event problem returns.
  - The physical bridge with the exact $p_T$ is not an option outside toys. It is O0's reference arm.
- Default: build nothing until the owner gives a go. O0 is proposed first because it is cheap and decides between (a) and (b).

---

## Appendix A. Changes from v1 (v2) and from v2 (v3)

- **Added:**
  - the design summary (placed after §1 in v3);
  - the owner's proposal (§3.19), with its Table 1 row, ladder branch (Table 10) and decision D5;
  - formulations named but not developed (§3.18);
  - the off-policy validity of the tied form (Lemma D) and a certificate without ground truth (Lemma G);
  - the reviewer's rung-1 check (Table 9);
  - an underdamped rung, a two-dt protocol, budgets and failure actions for every rung, and non-path CV baselines;
  - uncited prior work (Table 6).
- **Corrected:**
  - the topline's scope ("every path formulation with a fixed Langevin reference"), the finite-dt qualifier on the fixed backward, the 2°/0.4° angle numbers, and the bias-potential claim, now narrowed;
  - "generalisation across starts is open", now transfer across $\beta$ and systems only;
  - the Doob's Lagrangian wording;
  - REACT's matched-call comparisons;
  - the OU bridge-correction parameters;
  - the $-\beta\Delta F$ labels in §3.13;
  - Novikov for stopping times;
  - the stopping-time ratio, now up to $\beta U(x_0)+C_{dt}$;
  - D3's split by readout;
  - the Müller–Brown $\lambda_2$ sign;
  - the transport-row start-term conventions;
  - the §3.3 control-horizon cost;
  - the zero-gradient claim of Lemma G.
- **Demoted:**
  - "stay on-policy", from a rule to a law to measure;
  - the impossibility of an $F$-free flat-bias cost, from PROVED to DERIVED;
  - "cheapest per force call", now "lowest from 6.4e4 calls", on one toy (v2 said "from about 1.28e5"; corrected against the raw 5-seed table in v3);
  - Brownian-bridge killing, now 1D with point boundaries only;
  - "best linear CV", now the best of three scored.
- **Ladder:**
  - rung 1 uses the ELBO criterion instead of IS $\log Z$;
  - the bridge test moved to pre-flight;
  - v1's rung 4 was re-specified (now rung 8: $d$-independence in the separable well);
  - v1's rung 5 was split into rungs 10–13;
  - a non-separable high-$d$ toy was added.
- **Paths:** the old scratchpad paths were replaced by `energy_sampling/SCRATCH/nsde_design_nov24/`.
- **Checked against the code in v2:** this repository's $P_B$ (`models/gfn.py`, `configs/mk_dev.yaml`).

### Changes in v3

- **§3.19 re-derived and checked** in closed form on an OU Euler–Maruyama chain (`check_owner_idea/ou_checks.py`): items 3, 7, 9 (KL-form parts), 11, 12 and 13's cooling cost relabelled PROVED; Tables 5a and 5b added.
- **Corrected in §3.19:**
  - a learned helper reaches the minimum-bias optimum only when trained by maximum likelihood on forward paths; under a squared residual with free $\log Z$ every helper has a zero-residual partner (REFUTED as a training statement). Propagated to topline 6, Table 1, items 9–10, O0, O1, risk 13 and D5(a);
  - the bias cost is read as $\frac\beta4\mathbb E\int\lvert u\rvert^2=-\mathbb E\log w$, not from $\log Z$;
  - $\beta_{\rm ref}$ rises to $\beta$; the CGP match holds up to a time change; "barrier-relevant bias only at physical $\beta$" is now CONJECTURE;
  - the reweighting-variance claim is qualified; $B$ is defined for O0–O1 ($\pi(B)=0.0852$ for the basin, 0.0354 for Table 7's set);
  - Table 5's start-conditioning cells for CGP and EntangledSBM; the CGP wording; BoPITO's start-drawing tip added.
- **Corrected elsewhere:**
  - topline 7's budget (2e7 at dt = 2e-3, 4e7 at dt = 1e-3) and bridge killing in rung 1;
  - force-call caps for rungs 8, 9, 12, 13, B1, O2 and O3; criteria moved out of the budget column;
  - the Lemma B evidence list;
  - the mean-force crossover (6.4e4, not 1.28e5) in five places; Table 4b replaces the mixed bias rows and adds VES at 3.2e4 and the binned variant at 1.28e5;
  - the T = 0.5 OU bridge run credited to the simple deriver;
  - D4(a)'s description of rungs 0–13;
  - the $d$-scaling of dt in Table 2 and risk 4;
  - Table 9's caption, Lemma G's certificate ratio and rung 1's ratio, now scoped to bridge killing.
- **Style:** the topline now comes first; Table 7's test cells and the three-hole path; Table 6 row for Hou, Petersen and Negedly; "arXiv v1" in the Sendera row; ELBO gap added to the metrics.

### Changes in v4
- §3.19: new "What pins the helper" with Table 5c (claims H1–H6 checked by an independent deriver and refuter on OU chains) and its training consequences: alternate rather than train jointly, Brownian-bridge helper variance near the pin, the frozen-physical-helper route, self-hiding leaks.
- §3.19 Status: the OU check scripts are now in the durable SCRATCH folder.
- D5(a): alternation with a frozen helper, the variance schedule, and direct anchoring of the helper to physics.
