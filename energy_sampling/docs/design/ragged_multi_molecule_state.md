# Multi-molecule (mixed-k) conformer state: the carrier

**Status: implemented 2026-09-11, smoke-tested on a 6-molecule set (C N CO C=C C=O CN at
`full`, k = 6..15, K = 15). Not committed; the protocol is the owner's to choose.**

This replaces the 2026-09-10 plan here, which proposed a fully ragged `[sum_k]` state
through the rollout, the losses and the buffers. What shipped keeps the rollout DENSE and
confines raggedness to the one place it buys something: the policy.

## The problem

Every molecule has its own `k = 3N - 6` at level `full`, and its own periodic mask. The GFN
fixes both at construction (`GFN(dim=...)`, `angular_mask`), `split_params` slices contiguous
`[mean(dim), logvar(dim)]` blocks, and the buffers store fixed-width rows. Before this,
`MultiConformerTorsions` refused a member whose `data_ndim` or `periodic_dims` differed.

## The design

**State: a block-padded carrier** (`energies/conformer_carrier.py::CarrierLayout`).

    carrier = [ r block (R_max) | theta block (T_max) | phi block (P_max) ],  K = sum

A member's state column j goes to `offset[block(j)] + rank of j within its block`; every other
column is PAD. Consequences, each by construction:

- the periodic mask is column-constant (the phi block), so one `angular_mask` stays true;
- a set whose members share block counts is the IDENTITY layout, byte-identical to before;
- buffers, `split_params`, the flat backward policy and every `[B, dim]` path are unchanged.

**Pads are not coordinates.** Four guards, and none of them is a mask that can be forgotten
somewhere and read as a real coordinate at 0:

| where | what |
|---|---|
| `ConformerGFN._pin_dead` | pads written to exactly 0 at every wrap point and both endpoints (`torch.where`, so a NaN becomes 0) |
| `ConformerGFN.gauss_logprob`, `_pb_logprob` | per-row `state_mask` on the final sum; the exact P_B mixture via `_pb_mixture_ang_terms`, a replica of the shared-file method minus its sum (tested equal) |
| `MultiConformerTorsions._member_rows` | REFUSES a row whose pads are nonzero, or whose `state_mask` disagrees with the layout, before slicing it to its member's columns |
| `MultiConformerTorsions` chart methods | on a carrier `self` is a dispatcher; `potential_energy`, `sample_prior_states`, `dof_from_state`, ... are refused by name |

**Policy: ragged over valid columns** (`models/ragged_set_policy.py::RaggedConditionalSetPolicy`).
Tokens are gathered flat (`flat_idx`, `dof_batch`, bound once per trajectory), pooled with
segment ops, and scattered back to `[B, 2K]` blocks with pad outputs 0. Per-column identity is
`dof_static` (handcrafted, on the batch) plus the learned `DoFCorrelator` over per-atom
embeddings; the pooled molecular embedding joins `rho`'s context.

**Conditions** (`build_conformer_conditions.py --carrier`). Each graph is re-expressed in the
carrier: `ctree_{r,th,ph}_col` remapped (so `state_to_dof` works on a mixed batch),
`n_torsions = K`, `state_mask [1, K]`, `dof_static [1, K*F]`, `dof_atoms`/`dof_mask` padded to
K. The reconstruction is re-checked against each member's chart AFTER padding
(`check_carrier_convention`, 0.00e+00 A on all six). The run rebuilds the same layout from
the same member set, and `_member_rows` checks the two agree on every energy call.

**Prior** (`ConformerModeller._draw_carrier_prior`). Equal rows per member, each drawn,
optionally relaxed, and baked in its own chart, then placed in the carrier. Used by both
`init_prior_dataset` and the churn path `sample_from_prior`.

## Not done / known gaps

- **Physical eval stats** (`log_physical_properties`' `cm.*` block, basin coverage, tier
  minimum) read one chart; on a carrier the block is ABSENT with a one-time notice. Per-member
  versions are the next eval item.
- **Transverse columns** (`_free_block == 3`) have no carrier block; refused at layout build.
- **`policy_kind: set` cannot be resumed** (unchanged): any carrier set-policy run starts fresh.
- **Flow head** at >1 condition is a `scalarMLP`: `z_level_fill` cannot pin it and `lr_flow`
  must be sized for ~1.6M params, not a scalar.
- **`models/ragged_gfn.py`** (a `[sum_k, 2]` split_params) is superseded by the carrier and
  unused; kept pending the owner's call.
- **Atom orders differ** between the encoder and conformer paths; the builder aligns on
  `spec.perm`, unchanged.
- Water and HCN have no `full` chart; acetylene is refused as an incomplete chart.
