---
name: summ
description: Produce a very brief point-form status summary of the most recent work in the session — changes made, findings, and action items from the last exchange or two. Use when the user invokes /summ, or asks for a "quick summary", "status update", "where are we", or "what changed" mid-session. Not for handoff documents or end-of-task reports; those follow the Reporting section of AGENTS.md.
---

# summ — instant status on what just happened

Produce a point-form status summary of the **most recent work**, not the whole session. The reader is the owner, who stepped away for a few minutes: they need to know what just landed, not relive everything since the session opened.

## Scope

- Default window: the **last exchange** — the most recent user request and everything done in response to it.
- Extend to the **exchange before it** only when the current one is a direct continuation (a follow-up fix, a verification of the previous change, a correction). Two exchanges is the ceiling; never sweep the session.
- Earlier work appears only in two cases: an action item from earlier that is still open, or a recent finding that **overturns** something reported earlier. Say explicitly that it overturns it.
- If the window contains nothing worth reporting (a question answered, a file read, no change made), say so in one line instead of padding the three sections.

## Format

Exactly three sections, bullets only, no prose paragraphs:

**Changes**
- One bullet per changed file or contract. Lead with the *meaning* of the change; put the code identifier in parentheses. Example: "batch sizer now holds B×accum constant across the occupancy ladder (`train.select_batch_size`)". Omit files touched only mechanically (imports, formatting).

**Findings**
- One bullet per thing learned that the owner didn't know before — measurements, root causes, disproved hypotheses. State the finding, not the investigation. If a finding is unverified, say so in the bullet.
- Omit the section entirely if there are none.

**Action items**
- One bullet per open item, each phrased so it could be pasted into a to-do list and acted on cold. Split by owner if relevant ("you: relaunch the battery", "me: still owed the boundary test").
- If none: a single bullet, "none".

## Rules

- Hard cap ~6 bullets total; most invocations warrant 2–4. Selectivity, not compression — drop bullets that wouldn't change what the reader does next, rather than abbreviating every bullet into fragments.
- Recency is not importance. Within the window, a bullet still has to earn its place.
- No process narration ("first I searched...", "then I ran..."). Only outcomes.
- No shorthand coined during the session — a codename or label invented mid-work must not appear. Would each bullet parse for someone who read none of the transcript? If not, rewrite it.
- Distinguish "done" from "done and verified". A change whose test wasn't run gets flagged in its own bullet, not silently folded in.
- No preamble, no closing paragraph, no offer of follow-ups. The three sections are the entire response.
