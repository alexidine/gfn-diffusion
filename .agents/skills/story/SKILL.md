---
name: story
description: Retell a work item as a story — the steps taken in the order they happened, what each one showed, and what that meant for the next step — ending with where it leaves things. Use when the user invokes /story, or asks to "walk me through what happened", "tell me how we got here", or "narrate the work". An optional argument names the span when it isn't the most recent work item. Not a status update (that is /summ) and not the end-of-task report (AGENTS.md Reporting); both of those deliberately leave out the path.
---

# story — how the work went, step by step

/summ and the AGENTS.md report give outcomes and drop the path. /story is the path: the steps in the order they happened, what each one showed, and what that changed. The reader is the owner, who wants to follow the reasoning — why each step was taken, which turns were wrong, how the end state was reached — without reading the transcript.

## Scope

- Default: the most recent **work item** — from the request that opened it to now, across however many exchanges it took. One work item, not the session.
- With an argument, tell the span it names (`/story the batch sizer fix`, `/story the whole session`).
- Retell from the record; do not start new work to fill gaps. Where the record is thin, say so: beats that rest on a compacted-context summary rather than the transcript are marked as such, and work done by a subagent or another session is told as theirs and rests on their report unless I checked it.

## Format

**Topline** — one or two sentences: where the work ended, with its key number if it has one.

**The story** — numbered beats, in the order they happened. Each beat is a short prose paragraph, two to four sentences, carrying three things:

- the step, and why it was taken — the question it was meant to answer, or the earlier result that prompted it;
- what came back — the result itself, with its number, unit, and the reference it is read against;
- the upshot — what it ruled in or out, and so what came next.

Beats chain: the upshot of one is the motive for the next. When a step doesn't follow from anything before it (a new instruction from the owner, a side discovery), say what started it.

Illustration of form only:

> 2. I reran the same 200 steps on CPU, to rule out GPU nondeterminism as the source of the seed-to-seed spread. The CPU curves differed by the same 3% at step 200, so nondeterminism was not it. That moved the search to the data loader (beat 3).

**Open** — bullets: what is unverified, deferred, or owed, each actionable cold and split by owner where it matters ("you: …", "me: …"). A single "none" if none. Do not restate the topline.

## Rules

- **Actual order, not tidied order.** Tell it as it happened, including wrong turns, my own mistakes, and reversals; say what was wrong and how it was caught. A clean logical reconstruction that hides a detour misreports the work.
- **Hindsight is marked, not back-filled.** An early beat says what was believed then. When a later result overturns it, the later beat says so explicitly ("this overturned the reading in beat 2"), and the early beat carries a short forward pointer ("— overturned in beat 5") so the reader never carries a wrong belief to the end.
- **A beat earns its place by changing what was known or what was done next.** Fold mechanics — reading files, fixing typos, rerunning a flaky command — into the beat they served, or drop them. Most work items need 3–7 beats. Past ~10, the span is really several work items: tell the most recent and name the others in one line.
- **No invented motives.** The "why" is the reason stated or acted on at the time. If none was recorded, write "next I tried X" without one rather than supplying a plausible rationale after the fact.
- **Plain register.** Past tense; first person for my actions, "you" for the owner's. No dramatic framing ("the breakthrough", "surprisingly"); if a result contradicted an expectation, state the expectation instead.
- **Evidence status lives in the beat.** Measured, inferred, or assumed — say which when it isn't obvious. A change that was made but not tested is said to be untested.
- **Names parse cold.** No shorthand coined during the session. Meaning first, identifier in parentheses: "the batch-size selector (`train.select_batch_size`)". Run names, config keys and file paths go in parentheses.
- **Prose, not tables.** If a comparison genuinely needs a table, keep it to a few rows, labelled and captioned as AGENTS.md requires. Conformer measurements carry their tier.
- No preamble, no closing paragraph, no offer of follow-ups. Topline, story, open — that is the whole response.
