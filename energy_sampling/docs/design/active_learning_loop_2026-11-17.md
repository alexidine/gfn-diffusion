> Design proposal from owner discussion, 2026-11-17. Nothing here is built or run. It refines quiver E1/F2
> (`contingency_quiver.md`) and borrows from `phase2_active_learning_2026-11-10.md`; where they differ, this note is
> the later statement of intent. **[D]** derived from definitions, **[T]** seen in a 1-D toy only, **[U]** untested.

# Active learning of the training set

## Goal

A CSP search tool that finds all low-energy modes with as few energy calls as possible. The Boltzmann distribution
is the training target, used as a proxy for that goal. The claim to test: an active-learning loop reaches a given
cost-to-coverage with fewer energy calls than random acquisition. The lever is generalisation across conditions:
no single landscape has to be mapped. Large, flexible or high-Z′ molecules never saturate, even on their own.

## Loop

1. **Seed** D with training conditions. Hold a **fixed set of held-out conditions** that is never acquired.
2. **Round 0:** MLE warm start on D.
3. **Each round:**
   - train with TB for a fixed number of steps; the held-out gate (`heldout_gate`) may bring the next acquisition
     forward, and the step cap always ends the round;
   - draw a batch of conditions (§Acquisition);
   - for each condition, take on-policy draws and score them *with the model as it was before this condition's rows
     are trained on* (these are the probe), then add states from the state sampler;
   - score everything and append it to D.
4. **Budget axis:** every energy call counts: relaxations (about 200 calls per minimum), noised children, on-policy
   scoring and acquisition statistics. The benefit should show where calls are expensive (MLIP), not with eLJ.

## Loss

- **TB.** Its answer is π on the support of D, whatever route the data took to get there [D]. Acquisition's job is
  therefore to provide support.
- **On-policy draws are required.** They are the only check on mass q puts where π has none, and once stored they
  cover hot tails that search never reaches. They are part of acquisition, at about 1 in 20 training steps.
- **Old hot draws.** Temper their replay weight by energy, or let them age out. A young q's draws are the
  high-energy rows that tilted stored-row TB in the Nov 10 toy (about 3.9 nats between ledger skews at a pinned
  level) [T]. The level-free pairwise loss is the fallback [T].

## Acquisition

- **Per-condition error = the TB residual log w − log Ẑ(c) on the acquisition batch itself, split by source** [D]:
  - on q draws, its mean is −KL(q‖π): q is too hot, or puts mass where π has none;
  - on search or kicked starts (not drawn from q), a positive residual means q misses good states. These states
    find dropped modes, which q draws cannot.
  - On stored rows the residual can be recomputed with the current model at no energy cost.
- **Learned head beside log Ẑ(c)** (the open innovation). Outputs: mean and variance of log w on q draws, and the
  positive residual on search states. It is trained online from the residuals every TB batch already computes,
  plus first-visit residuals of acquired conditions. An ensemble gives the uncertainty.
- **The head's bias.** Trained-condition labels underestimate error on unseen conditions, and an ensemble cannot
  see this: an unseen near-twin of a trained molecule gets a confident, low prediction [D]. So the head takes as
  inputs the rows of c in D and the distance to the nearest trained condition, and gets labels for both cases.
- **Batch selection.** Draw a uniform candidate set from the pool and select by UCB or Thompson sampling on the head
  (BoTorch-style, with optional re-draws). Diversity comes from a fixed molecular fingerprint, not the conditioner
  embedding, which moves during training.
- **Probe stratum.** A fixed share of each batch goes to the nearest untrained neighbours of recently trained
  conditions, chosen by distance rather than by the head.
- **Hard conditions are allowed.** A per-round cap on each condition's share bounds the spend; a condition may
  return in later rounds.
- **Condition generator:** parked; a separate project.

## State sampler

- Per condition: q draws, plus local optimisation from uniform starts, prior starts, transplants from nearby
  conditions, and kicks around D's minima. It can run asynchronously on a separate GPU stream.
- An exp-6 rough optimisation with MLIP polish depends on the exp-6 recall test (quiver F3).
- Starts seeded by the model cross the Oct 17 default that search is not seeded by the trained sampler (loosened on
  Nov 4). Keep a fixed share of uniform or prior starts so the support is not self-referential.
- A separate state generator: parked until there is an idea for training it.

## Overfitting

| Measurement | Drawn from | Question it answers | Feeds | Extra energy calls |
|---|---|---|---|---|
| Fixed held-out conditions (`test_molecules_path`, logged as `eval_test/*`) | a set that is never acquired | is the loop as a whole generalising? | stopping, `heldout_gate`, the paper's metrics | already paid at each eval |
| First visit to an acquired condition | each round's batch, scored before training | how wrong is the model on an unseen condition? | labels on unseen conditions for the head | none: the planned on-policy draws |
| Neighbour gap | first-visit residual minus that of the nearest trained condition | is the model memorising there, or under-sampled? | the choice between adding conditions and adding states | none: computed from the row above |
| Held-out rows within trained conditions (row store `held_out`, replay `val_frac`) | rows of conditions in D | are there memorised spikes within a condition? | rebuilding or widening children | none: rows already stored |

*Which overfitting measurement serves which decision. Proposal only; the first and last rows exist in code today,
the middle two are new. The atlas's `neighbour_matched` (`eval/cond_panel/trends.py`) computes the neighbour gap
offline from checkpoints.*

- **Measure before you train.** That ordering makes on-policy draws double as probes.
- **The remedy is more data of the right kind.** A large neighbour gap at short distance means add conditions
  around c. A high residual on both sides means add states on c.
- **The fixed held-out set never informs acquisition.** Once it chooses conditions, it is training data by
  selection.

## Termination

- **The loop:** a round count or energy budget.
- **A round:** a step cap; the gate can only shorten it.
- **A condition:** a per-round cap on its share of the batch.
- **Saturation:** never relied on, since hard conditions need not saturate.

## Staged plan

1. **Pool-based, retrospective, on pool8.** Use the P-1 search over 72k molecules with pre-built K = 4 rows.
   Acquire conditions only and reveal their rows from the pool; compare against random reveal at equal row count.
   The only energy calls are eLJ scoring of on-policy draws. If choosing conditions does not beat random here,
   choosing states probably will not rescue the result.
2. **Live state acquisition with eLJ on norbornane,** whose P-1 and P2_1/c searches give reference landscapes.
3. **MLIP.**

**Readouts:** cost-to-coverage (`eval/cond_panel/csp.py`) on the fixed held-out panel, against an independent
reference search, with energy calls on the x-axis; and the generalisation radius (neighbour gap against distance)
per round.

## Open

- The design of the head: online labels, first-visit labels, and how to weight between them [U].
- Whether stored-row TB tilts on crystals, and whether the level-free fallback is needed [U].
- Exp-6 recall (quiver F3).
