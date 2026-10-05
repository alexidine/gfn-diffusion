# Research notes

Status: **THE OWNER'S RESEARCH RECORD, NO AUTHORITY OVER CODE**
Scope: `energy_sampling/docs/research/`

One file per studied system or research question (a molecule, a landscape, a paper's set of runs). These are the detailed system-specific analyses: what a landscape looks like, which packings a search found, how two energy models rank the same structures. They bear on the research projects, not on the code, so they are kept out of the wiki.

## What a note holds

- First line after the title: `*Research note. Not repository policy and not a description of the code.*`
- Dated entries or dated sections. Each result carries its date, the model or checkpoint, the energy function and its currency, the number of structures or draws, and the path of the artefact or report it came from.
- Tables with labelled columns, named rows and a caption.
- Owner decisions about the system (which conformer, which model, which cut), marked as the owner's with their date.
- Open questions.

## How it differs from the neighbours

- A **wiki page** says what the code does and what the theory derives. A note says what was seen on one system.
- A **battery record** under `../results/` is frozen when a battery closes. A note is living: a later analysis amends it, with the earlier entry kept and marked superseded when a number changes.
- A note is evidence about a system. It does not set a default, an invariant or a workflow. Where a note and the code disagree about what the code does, the code is right.

## Sources

Reports and data live outside the repository, mostly under `D:/crystal_datasets/`. A note points at them by path and says what the file is. A path that no longer exists is marked as such, not deleted from the note.
