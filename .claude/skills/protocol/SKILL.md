---
name: protocol
description: Render a training config's protocol as a scannable HTML card (one panel per stage, a row per branch with where its rows come from, its loss terms, what it trains and its share, then cadence, draw, balance, LR, entry and exit) and show it to the owner. Use when the user invokes /protocol, optionally with a config path or a run name, or asks to "show the protocol", "what does this run's training protocol do", or "card for <config>". Not for workflow diagrams of code paths (that is /workvis).
---

# protocol — the training protocol as a card

The card is generated from the config by `energy_sampling/protocol_card.py` through `config_snapshot --card`, from the same stage parse and loss-coefficient overlay the trainer uses, so it shows what a run does rather than what a comment says. Never hand-write or hand-edit a card.

## Steps

1. **Pick the config.**
   - No argument: `energy_sampling/configs/mk_dev.yaml` (the canonical control surface).
   - A path: that YAML.
   - A run name (e.g. `norb_p2_nov08_tbhead_s6k`): the config that launched it. Look first beside the run's log or in the battery folder that generated it (`configs/<battery>/`, or a scratch folder named in the session); if it cannot be found, ask rather than guess.
2. **Render.** From `energy_sampling/`, with the project interpreter and PYTHONPATH the session already uses:

   ```
   python -m config_snapshot <config> --card docs/workvis/protocol_card_<stem>.html
   ```

   It runs the config contract first. On `contract FAILED`, report the listed issues and stop; do not render around them.
3. **Look at it before sending.** Screenshot it headlessly (Edge or Chrome `--headless=new --screenshot=<png> --window-size=1120,1300 file:///<html>`) and read the PNG. Check that every stage panel is present, chips do not wrap mid-label, and nothing says `base coefficients unavailable`.
4. **Show it** with SendUserFile (`display: render`). Under it, in two to four lines, name what the card makes visible that matters for the question at hand, such as an inert branch, an OFF balance, a stage skipped by `skip_if`, or rows that differ between stages. Do not restate the card.

## Rules

- The card reads only the config, not data files. The prior-buffer line describes a pre-noised build; when the run's prior file is not one, say so beside the card.
- Generated cards in `docs/workvis/` are viewing copies; do not commit them.
- A wrong or unclear card is a defect in `protocol_card.py`: fix it there, add a case to `tests/config/test_protocol_card.py`, and update the card paragraph of `docs/wiki/config-validation.md` in the same change.
- To compare two runs, render both and point at the differing rows; there is no diff view yet.
