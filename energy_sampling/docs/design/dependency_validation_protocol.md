# Validating an external dependency: method and template

Status: **ACTIVE METHOD**
Scope: any place we reimplement, accelerate, or restructure a third-party
computation and then depend on the number it returns. Written from the UMA and
MACE validations (`docs/mlip_validation.md`, F-053/F-055); the method is not
MLIP-specific.

This is a reusable procedure, like `design/comment_audit.md`. It is not a record
of a particular validation — those live in their own write-up and in
`findings.md`.

---

## 1. The failure this exists to prevent

A suite can be large, green, and prove nothing about correctness.

The pattern: we take an upstream computation, rebuild its input path for speed
(batched construction, device-side neighbour lists, hoisted builders), and then
test the fast path against our own slow path. Every such test compares **two of
our routes**. It is a *regression* gate — it catches drift between our own
versions — and it is worth having. But a defect that sits upstream of the fork,
in the part both routes share, is invisible to every test of that shape, no
matter how many there are.

The UMA suite had nine such gates and could not have caught a systematically
wrong energy. Neither could MACE's four.

**The distinguishing question for any gate: if this assertion fails, what did I
learn? And if it passes, what have I ruled out?** Internal parity rules out
"our two paths disagree". It does not rule out "we are both wrong".

## 2. The three levels

| Level | Compares | Rules out | Cost |
|---|---|---|---|
| **L0 · internal parity** | two routes that share their upstream derivation | drift between routes that fork below the shared part | cheap, CPU-able |
| **L1 · stock-workflow ground truth** | our production path vs the dependency's own documented workflow | our construction, conventions, and setup being wrong | one test file, needs the real model |
| **L2 · published reference** | the dependency's workflow vs numbers its authors published | our *install* or checkpoint differing from theirs | needs upstream data |

**The axis is derivational, not about who wrote the code.** A test comparing our
code against an upstream helper is L1 for whatever it covers: `test_pbc_neighbours.py`
checks our production neighbour list against matscipy on exact edge sets and is a
genuine external check of the GRAPH, even though it sits beside the parity tests.
Classify by what is shared, not by authorship, or you will file a real external gate
as internal and claim less coverage than you have.

L0 is not optional and not sufficient. **L1 is the one that converts belief into
evidence** and is the subject of this document. L2 is cheap to *wire* and often
impossible to *populate*; build the mechanism, skip cleanly without data, and do
not fabricate a reference — a reference we generated ourselves is L1 with extra
steps.

## 3. Required elements of an L1 gate

A gate missing any of these is weaker than it looks. Each has a failure it
prevents.

**1 · Hand off as early as possible.** Build the stock side from the
*checkpoint path*, not from our already-loaded model object, and with the
dependency's own default settings — no overrides, no monkey-patching, no
post-load surgery. Sharing our object still tests construction but silently
excludes the load. *Prevents: validating everything except the part we changed.*

**2 · Build the comparison input independently.** Construct the stock side's
input from primitive state (positions, cell, elements), not by calling our own
converter. If both sides route through the same helper, a bug in it cancels.
*Prevents: a converter bug proving itself correct.*

**3 · Measure the control in the same test, on the same data.** Never assert
exact equality against nondeterministic hardware, and never hard-code a
tolerance. Run one side twice, take that spread as the control, and require the
cross-stack delta to sit within a small multiple of it. *Prevents: measuring the
GPU's reduction order instead of the code; and a pinned tolerance rotting
silently.*

**4 · State the bar in interpretable units, sized between two known scales.**
The bar belongs between the noise floor and the smallest defect worth catching,
and the write-up must name both numbers. Raw units are for the assertion;
*decision* units (here, kJ/mol per molecule against a kT of 2.5) are for the
message. *Prevents: a bar nobody can argue with because nobody knows what it
means.*

**5 · A negative control that must fail.** Perturb the input by less than a real
bug would and require the comparison to reject it. A gate never observed to fail
has unknown power. *Prevents: bars so loose they cannot catch anything.*

**6 · A not-accidentally-identical guard.** Assert the two sides are *not*
bitwise equal. Two independent stacks on a GPU never are; equality means the
fixture is comparing something against itself. *Prevents: the whole file passing
vacuously after a wiring mistake.*

**7 · Assert preconditions, do not assume them.** Anything that would
desynchronise the two sides — a density guard that mutates cells, a element
table the model does not cover, a batch-size ceiling — gets its own assertion or
an honest skip. *Prevents: a comparison that is quietly invalid rather than
loudly wrong.*

**8 · Attribute the residual; do not tolerate it.** If the two sides differ by
more than the control, find out why before widening the bar. The MACE residual was
traced to our own fractional round-trip perturbing positions by 1.9e-6 Å — provable
by feeding the stock side the same round-tripped positions and watching the gap
collapse 4-12x. **An attributed residual is a result; an unattributed one is a
tolerance.**

**8a · Attribute the CAUSE and the MECHANISM separately, and say which you have.**
The intervention above establishes *what* the residual depends on. It does not
establish *how*. The first draft of that write-up asserted a mechanism — edges
flipping at the model's neighbour cutoff — which our own source refutes, because our
graph is built before the round trip is applied and cannot flip. An intervention is
strong evidence of cause and no evidence of mechanism; do not let a plausible story
ride along on it. A cause without a mechanism is still a result. A mechanism you have
not checked against the code is a liability, and it will be quoted back at you.

**9 · State the regime and why it is the right one.** Every L1 gate is scoped.
Name what is excluded and the mechanism that forces the exclusion (e.g.
fairchem's internal graph truncates at `max_neighbors=300`, so degenerate cells
would measure its neighbour cap rather than our code).

**10 · Name what both sides still share.** The fork you opened has its own
upstream. Whatever layer feeds *both* routes is still untested and will cancel
exactly rather than show up as a disagreement. In the MLIP case both stacks consumed
the same `unit_cell_pos` and re-derived the same tiling, so a layout inversion in the
symmetry expansion would have been invisible to every test while leaving them green
on a chemically nonsensical structure. Element 2 is converter-scoped and element 9 is
regime-scoped; neither catches this. Write the shared layer down in the scope
section — then close it with a check that does not use the shared assumption.
`tests/test_unit_cell_layout.py` is the worked example: instead of comparing element
labels (the assumption under test) it compares GEOMETRY, requiring each block to carry
the asymmetric unit's distance matrix, which is true under the right layout and false
under any other. *Prevents: a validated pipeline resting on an unvalidated
foundation.*

## 4. Anti-patterns

- **Making the two sides identical to get agreement.** If our path is unwrapped
  and theirs wraps, do *not* wrap by hand: that deletes the difference the test
  exists to probe. Reason about why the difference is physically immaterial, and
  let the test cross it.
- **A bar hugging the measurement.** A bar at the observed maximum fails on the
  next fixture and teaches everyone to ignore the file. Leave headroom and say
  how much.
- **Reading the pass count.** Coverage loss shows up as a *rising* pass count
  with a rising skip count. Always run `-rs` and read skips first. See F-054.
- **Module-scope global mutation in a test file.** `os.environ`,
  `set_default_dtype`, backend flags — pytest imports every collected module
  before running anything, so one file reconfigures the session. See F-054 and
  the `set_default_dtype` precedent.
- **Skips that read as passes.** A gate that skips without its checkpoint,
  without a GPU, or behind a busy card must say which, and the run must be read
  with that in mind.

## 5. Write-up template

One document per dependency family, at `docs/<family>_validation.md`, status
`ACTIVE`. Sections, in order:

1. **Verdict** — two sentences, numbers included, at the very top.
2. **What was compared** — the two stacks, named concretely enough to rebuild.
3. **Why the comparison is fair** — the substantive argument, including any
   deliberate difference between the sides and why it does not invalidate the
   result. This is the section a sceptical reader attacks first.
4. **Results** — a table, in both raw and decision units, with the control
   beside every cross-stack number. Say how many repeats.
5. **Residual attribution** — what is left, and what it is. `unattributed` is a
   legitimate entry; a missing section is not.
6. **Scope and limits** — regime tested, what is excluded, and explicitly what
   this does *not* establish.
7. **How to run it** — the exact command, the expected green line, and the skip
   reasons that mean "did not actually run".
8. **What would invalidate this** — the conditions under which the result stops
   holding.

Then: an entry in `findings.md` with a grade (`REPLICATED` needs ≥2 conditions
or seeds and an effect exceeding measured noise), and a router line in
`docs/README.md`. The write-up explains; the ledger entry is the evidence; the
test is the proof. Do not let the document be the only home of a number.

## 6. Checklist

Before calling a validation done:

- [ ] Stock side built from the path, with upstream defaults
- [ ] Comparison input built independently of our converter
- [ ] Control measured in the same test, same data
- [ ] Bar between the noise floor and the smallest real defect, both named
- [ ] Negative control present and passing
- [ ] Not-accidentally-identical guard present
- [ ] Preconditions asserted or skipped honestly
- [ ] Residual attributed, or explicitly recorded as unattributed
- [ ] Cause and mechanism distinguished; any mechanism checked against the source
- [ ] The layer both sides still share named in the scope section
- [ ] Scope and exclusions stated with their mechanism
- [ ] Repeated ≥2 times if claiming `REPLICATED`
- [ ] Run with `-rs`; skip reasons read, not just the pass count
- [ ] `findings.md` entry filed with a grade
- [ ] `docs/README.md` router updated
