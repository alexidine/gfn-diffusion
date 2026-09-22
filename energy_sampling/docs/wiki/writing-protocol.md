# Writing protocol

How a page of this knowledge base is produced. Every drafter and reviewer reads this first, then [index.md](index.md) for the page conventions, then two existing pages as exemplars.

## 1. Draft, from the code and the derivations

- The page is about one thing. If a neighbouring object wants to expand, name its page in one sentence and leave it out.
- Sources are the code at the stamped commit, the derivation notes under `docs/design/` and `docs/*.tex`, and the canonical config. Agent memory files may be used to locate things and never as evidence. Where a memory says the code does X, check the code; the page states what the code does today.
- Every symbol cited exists. Check by grep before citing. Notation: `cfg:block.key` for a config key, `file.py::Class.method` or `file.py::function` for a symbol, paths relative to the repo that owns them.
- Prose for a graduate student or a new agent. Define terms. Equations in `$...$` and `$$...$$`. A Mermaid diagram only where it shows a real mechanism. Plain punctuation: no em or en dashes.
- Sections: introduction without a heading; headings as the content needs; then `Owner choices` as the placeholder line; `Config keys`; `Could be tooling` on code-bound pages; `Sources`.
- Length follows the subject: a page is as long as its mechanics and derivations need and no longer. A config key with three consumers may be three hundred words; a page carrying derivations may be several times that. There is no target length, and padding toward one or trimming to one is a defect.
- First line after the title: `*Drift: **T|C|M** (...). Verified against commit \`<hash>\`, <date>. Sources at the end.*`

## 2. What the page says and does not say

The page says **what**: what the code does, what the theory derives, and where the two differ. It does not say **why** anything was chosen, what was tried, what a run showed, or what to do.

- No verdicts. Not "this shows", "rules out", "the lever is", "full stop". A single run is never a conclusion.
- No doctrine. No "principle", "rule", "we hold that". A description of the code is not a rule the project has adopted.
- No owner voice. No rationale for a choice ("for this reason", "which is why the config ships"). Rationale is the owner's to write, under `Owner choices`.
- No narrative. No "we tried", no chronology, no battery or run names as evidence for a claim in the body.
- No insight commentary. Not "the whole page is", "half true", "the reframing".
- No second-person procedure. Not "do not read X from Y", "always seed after". If a fact makes a procedure necessary, state the fact; the reader draws the procedure.
- Numbers only as tagged calibrations: `**[calibration, run, date]**` for a constant that is used operationally, with no interpretation attached. Any other number in prose needs its source in the same sentence or comes out.
- Where a derivation and the code disagree: one plain sentence each, what the derivation says and what the code does. No recommendation.
- A dated retirement of a key or mechanism is a fact and stays. A message the code prints is a fact and stays.

## 3. Verify

A verifier, which may be the integrator or a separate agent that reads the sources, reads the draft in full and checks at least five material claims against the code by grep before installing. A claim that fails is corrected on the page to what the code does, or removed if it cannot be stated from the code. The verifier also normalises cross-links to the canonical page names in [index.md](index.md), sets the commit stamp to the commit the code was read at, installs the page, and adds its index line. The verifier is never the reviewer of the same page.

## 4. Review

A separate agent that has not seen the draft's sources reads the installed page against the list in section 2 and reports, per page: at most five findings, each with the item, a quoted phrase under fifteen words, and one sentence. It also reports any recurring pattern the list does not name. It does not edit.

## 5. Fix

The integrator applies the findings it accepts and records the ones it rejects with a reason. Known over-strict readings, rejected by default: a dated retirement flagged as history; a printed message flagged as reassurance; a `Could be tooling` section flagged as roadmap; a definition or identity flagged as doctrine because it uses the word "principle" (fix the word, keep the content).

## 6. Record

Each installed page is listed in [index.md](index.md) with a one-line scope. The commit hash on the page is the one it was verified against, and is updated only by re-verification.
