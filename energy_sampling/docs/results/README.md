# Battery results

Status: **RECORDS, NOT FACTS**
Scope: `energy_sampling/docs/results/`

One file per closed test battery. A record keeps what a designed battery measured so the work is not lost. It holds datasets, trends and correlations. It is not a finding, it proves nothing about the code, and a wiki page does not cite it as evidence.

## What qualifies

A battery that was designed as a comparison: named arms, a stated question, and enough steps to read. A single run, a smoke test or a debugging session does not get a record.

## What a record holds

- First line after the title: `*Battery record. Closed <date>. Code at commit \`<hash>\`. Not a finding.*`
- **Design.** The question, the arms and what differs between them, the base config, seeds per arm, where the runs ran.
- **Data.** Where the logs, checkpoints and figures are, by path or run name, and which of those still exist.
- **Measurements.** Tables, each with labelled columns (quantity and unit), rows named by arm, and a caption saying what was measured, on what, from how many seeds or samples, and the resolution below which a difference is noise.
- **Trends.** What moved with what, stated as an observation with its range: "across the four arms X rose as Y fell". No cause is asserted.
- **Not covered.** What the battery did not vary and what was not checked.

## What a record does not hold

- Verdicts. Not "this shows", "rules out", "the lever is". A trend on one battery is a trend on one battery.
- Recommendations or next steps.
- Resume state. A record is written when the battery is closed.

## Lifetime

A record is written once and then frozen. It is not updated when the code moves, and `wiki/check_refs.py` does not read this directory. A symbol or config key in a record is the name it had at the stamped commit.

## Promotion

A trend that holds across batteries and has a mechanism can become wiki content. Promotion is new work: the mechanism is derived or verified against the code under `wiki/writing-protocol.md` and written on the page for that mechanism. The record stays as it was. The terminal reward force on `wiki/gradient-routes-and-terminal-force.md` is the example of a result that was promoted.
