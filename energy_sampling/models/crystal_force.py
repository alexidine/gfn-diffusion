"""Force on the crystal latent from a pre-trained atom trunk, in the form GFN's force terms read.

`TrunkForce` loads a checkpoint written by `pretrain_atom_trunk.py` and is installed on a
sampler with `GFN.install_drift_force`. For a batch of trajectories the caller builds one
context from the crystal batch the rows' crystals are built from (`TrunkForce.context`) and
hands it to the trajectory functions as `drift_context`; the sampler then calls the
provider with every state it needs a force at:

    force = provider(state [B, dim], context, create_graph)        # -dE~/dx, kT per latent unit

E~ is the trunk's energy: the per-atom-compressed eLJ energy in kT at ONE temperature and
`lj_coeff`, the checkpoint's (`check_energy` refuses a run whose energy differs). The
molecule-only parts of the geometry (image tables, intramolecular pairs) are built once
per context; a state costs the latent map, the image selection, the pair distances inside
the trunk's cutoff, and one forward and one backward pass of the trunk.

MEMORY IS BOUNDED WHATEVER THE STATE. A cell squeezed far below any physical density (what
an untrained policy produces) has thousands of times the pairs of a crystal: 2.2 million
inside 5 A against at most 21 thousand on the trunk's training states (QM9, measured
2026-10-06). `max_images` and `max_pairs` cap what one crystal may hold (its nearest images,
its shortest pairs), and `max_pairs_per_call` splits a batch of rows across trunk calls, so
no state can ask for more than a fixed amount. A capped crystal's force is the force of its
truncated pair list; `capped_rows` counts them.
"""
import math
from dataclasses import dataclass, replace
from typing import List

import torch

from .atom_trunk import AtomTrunk, crystal_density, intramolecular_edges
from mxtaltools.analysis.vdw_analysis import exponential_edgewise_lj_energy
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.image_pairs import ImageTables, build_image_tables, select_images, pair_distances


def compressed_elj_per_atom(pairs, vdw, lj_coeff: float, temperature: float, compress_at: float, n_nodes: int,
                            envelope=None):
    """The trunk's target, per atom: eLJ pair energies in kT summed onto their reference atom, then
    log-compressed above `compress_at` kT. `pairs` is a `pair_distances` result; `envelope`, if given,
    multiplies each pair energy (the pre-training script's short-range variant)."""
    e_pair = exponential_edgewise_lj_energy(
        vdw, {'intermolecular_dist': pairs['dist'],
              'intermolecular_dist_atoms': [pairs['z_src'], pairs['z_tgt']]}, 2.5) * (lj_coeff / temperature)
    if envelope is not None:
        e_pair = e_pair * envelope
    raw = torch.zeros(n_nodes, dtype=e_pair.dtype, device=e_pair.device).index_add_(0, pairs['node_ref'], e_pair)
    return log_rescale_positive(raw, compress_at)


@dataclass
class _Chunk:
    rows: slice                 # rows of the batch
    tables: ImageTables         # this chunk's tables, node pointers counted from its first atom
    z: torch.Tensor             # [n] atomic numbers of its atoms
    node_graph: torch.Tensor    # [n] crystal index within the chunk
    intra_index: torch.Tensor   # [2, Ei]
    intra_dist: torch.Tensor    # [Ei]


@dataclass
class TrunkContext:
    """What one batch of rows needs to turn a latent into trunk inputs. Built by `TrunkForce.context`."""
    crystals: object            # the rows' crystal batch, owned by the context: its cell is overwritten per state
    chunks: List[_Chunk]
    num_rows: int


def _slice_tables(tables: ImageTables, lo: int, hi: int, node_lo: int) -> ImageTables:
    return replace(tables, p=tables.p[lo:hi], amask=tables.amask[lo:hi], z=tables.z[lo:hi], nat=tables.nat[lo:hi],
                   W=tables.W[lo:hi], w=tables.w[lo:hi], kmask=tables.kmask[lo:hi], radius=tables.radius[lo:hi],
                   ptr=tables.ptr[lo:hi] - node_lo)


class TrunkForce:
    """A frozen `AtomTrunk` as a force provider.

    Parameters
    ----------
    checkpoint : path of a `pretrain_atom_trunk.py` checkpoint (arm 'trunk', target 'full').
    device : where the trunk and every context live.
    chunk : rows per geometry build; a context's rows are split into chunks of this size.
    max_images, max_pairs : per-crystal caps on image molecules and on atom pairs inside the
        trunk's cutoff (`image_pairs.select_images`, `pair_distances`). The defaults are about
        twice the largest counts on the trunk's training states.
    max_pairs_per_call : pairs one trunk call may be handed; rows over it go to further calls.
    """

    def __init__(self, checkpoint: str, device, chunk: int = 1000, max_images: int = 2000,
                 max_pairs: int = 40_000, max_pairs_per_call: int = 4_000_000):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        a = ck['args']
        if a.get('arm') != 'trunk' or a.get('target', 'full') != 'full':
            raise ValueError(f"{checkpoint} is arm {a.get('arm')!r}, target {a.get('target')!r}; a force provider "
                             f"needs a trunk fitted to the full target")
        self.checkpoint = str(checkpoint)
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.max_images, self.max_pairs = int(max_images), int(max_pairs)
        self.max_pairs_per_call = int(max_pairs_per_call)
        if min(self.max_images, self.max_pairs, self.max_pairs_per_call) < 1 or self.max_pairs > self.max_pairs_per_call:
            raise ValueError(f"max_images {max_images}, max_pairs {max_pairs} and max_pairs_per_call "
                             f"{max_pairs_per_call} must be positive, and one crystal's pairs must fit in a call")
        self.cutoff = float(a['feature_cutoff'])
        self.label_cutoff = float(a['label_cutoff'])
        self.compress_at = float(a['compress_at'])
        self.temperature = float(a['temperature'])
        self.lj_coeff = float(a['lj_coeff'])
        self.step = int(ck.get('step', -1))
        self.row_scale = ck['row_scale'].to(self.device)
        self.trunk = AtomTrunk(node_dim=a['node_dim'], message_dim=a['message_dim'], num_convs=a['num_convs'],
                               cutoff=self.cutoff, folded=not a['unfolded']).to(self.device)
        self.trunk.load_state_dict(ck['model'])
        self.trunk.requires_grad_(False).eval()
        self.calls = 0          # force evaluations (one per state batch)
        self.rows = 0           # crystal states evaluated
        self.capped_rows = 0    # of those, states whose image or pair list was capped
        self.extra_calls = 0    # trunk calls beyond one per chunk, made to stay under max_pairs_per_call

    def check_energy(self, temperature: float, lj_coeff: float, rtol: float = 1e-6):
        """Refuse a run whose energy is not the one the trunk was fitted to. The target is
        compressed per atom in kT, so it is not a rescaling of the energy at another
        temperature or coefficient."""
        for name, mine, theirs in (('temperature', self.temperature, float(temperature)),
                                   ('lj_coeff', self.lj_coeff, float(lj_coeff))):
            if abs(mine - theirs) > rtol * max(abs(mine), 1.0):
                raise ValueError(f"the trunk in {self.checkpoint} was fitted at {name} = {mine:g}; this run has "
                                 f"{name} = {theirs:g}")

    def context(self, crystal_batch) -> TrunkContext:
        """Context for rows whose crystals are built from `crystal_batch` (row i <-> graph i).
        The batch is cloned: the caller's object is not touched afterwards."""
        cb = crystal_batch.clone().to(self.device)
        tables = build_image_tables(cb)
        node_ptr = cb.ptr.to(self.device)
        chunks = []
        for lo in range(0, cb.num_graphs, self.chunk):
            hi = min(lo + self.chunk, cb.num_graphs)
            n_lo, n_hi = int(node_ptr[lo]), int(node_ptr[hi])
            sub = _slice_tables(tables, lo, hi, n_lo)
            intra_index, intra_dist = intramolecular_edges(sub, self.cutoff)
            chunks.append(_Chunk(rows=slice(lo, hi), tables=sub, z=cb.z[n_lo:n_hi].long(),
                                 node_graph=cb.batch[n_lo:n_hi] - lo,
                                 intra_index=intra_index, intra_dist=intra_dist))
        return TrunkContext(crystals=cb, chunks=chunks, num_rows=cb.num_graphs)

    def energy_and_force(self, state, ctx: TrunkContext, create_graph: bool = False):
        """(E~ [B] in kT, -dE~/dstate [B, dim]). `state` is the sampler's latent, one row per context row."""
        if ctx is None:
            raise ValueError("TrunkForce needs a context (TrunkForce.context of the rows' crystal batch) passed "
                             "to the trajectory function as drift_context")
        if state.shape[0] != ctx.num_rows:
            raise ValueError(f"{state.shape[0]} states for a context of {ctx.num_rows} rows")
        with torch.enable_grad():
            x = state if state.requires_grad else state.detach().requires_grad_(True)
            cb = ctx.crystals
            cb.latent_to_cell_params(x.to(self.device))
            T_fc, T_cf, centroid, orientation = cb.T_fc, cb.T_cf, cb.aunit_centroid, cb.aunit_orientation
            energy = torch.zeros(ctx.num_rows, dtype=T_fc.dtype, device=self.device)
            grad = torch.zeros_like(x)
            for c in ctx.chunks:
                r = c.rows
                sel = select_images(c.tables, T_fc[r], T_cf[r], centroid[r], orientation[r], self.cutoff,
                                    max_images=self.max_images)
                pairs = pair_distances(c.tables, sel, self.cutoff, max_pairs=self.max_pairs)
                self.capped_rows += int((sel['capped'] | sel['image_capped'] | pairs['pair_capped']).sum())
                density = crystal_density(c.tables, T_fc[r])
                for rows, inputs in self._calls(c, pairs, density):
                    e = self.trunk(*inputs)['energy']
                    g, = torch.autograd.grad(e.sum(), x, retain_graph=True, create_graph=create_graph)
                    grad = grad + g
                    energy[r][rows] = e.detach()
        self.calls += 1
        self.rows += ctx.num_rows
        return energy, -grad

    def _calls(self, c: _Chunk, pairs, density):
        """The trunk calls of one chunk: (rows of the chunk, trunk inputs) for the whole chunk when its
        pairs fit `max_pairs_per_call`, else for consecutive groups of crystals that each do. Each
        group's energy depends on its own crystals alone, so the caller may take its gradient and
        let it go before the next."""
        n_rows = c.rows.stop - c.rows.start
        if int(pairs['dist'].numel()) <= self.max_pairs_per_call:
            yield slice(0, n_rows), (c.z, c.node_graph, c.intra_index, c.intra_dist,
                                     pairs['node_ref'], pairs['node_img'], pairs['dist'], density)
            return
        counts = torch.bincount(pairs['graph'], minlength=n_rows).tolist()
        bounds, held = [0], 0
        for g, k in enumerate(counts):
            if held and held + k > self.max_pairs_per_call:
                bounds.append(g)
                held = 0
            held += k
        bounds.append(n_rows)
        node_ptr = c.tables.ptr.tolist() + [int(c.z.numel())]
        self.extra_calls += len(bounds) - 2
        for g0, g1 in zip(bounds[:-1], bounds[1:]):
            n0, n1 = node_ptr[g0], node_ptr[g1]
            mine = (pairs['graph'] >= g0) & (pairs['graph'] < g1)
            intra = (c.intra_index[1] >= n0) & (c.intra_index[1] < n1)     # an edge never leaves its molecule
            yield slice(g0, g1), (c.z[n0:n1], c.node_graph[n0:n1] - g0, c.intra_index[:, intra] - n0,
                                  c.intra_dist[intra], pairs['node_ref'][mine] - n0, pairs['node_img'][mine] - n0,
                                  pairs['dist'][mine], density[g0:g1])

    def __call__(self, state, ctx: TrunkContext, create_graph: bool = False):
        return self.energy_and_force(state, ctx, create_graph)[1]

    def target_force(self, state, ctx: TrunkContext):
        """(-d(target)/dstate [B, dim], capped [B]): the force of the energy the trunk was fitted to, the
        compressed eLJ energy inside `label_cutoff`, computed exactly; what the trunk's force is an
        estimate of. The per-crystal caps apply here too, scaled by the volume ratio of the two cutoffs,
        and `capped` marks the rows they cut: there this is the force of a truncated list, not the target."""
        vdw = torch.tensor(list(VDW_RADII.values()), device=self.device)
        wider = math.ceil((self.label_cutoff / self.cutoff) ** 3)
        capped = torch.zeros(ctx.num_rows, dtype=torch.bool, device=self.device)
        with torch.enable_grad():
            x = state.detach().to(self.device).requires_grad_(True)
            cb = ctx.crystals
            cb.latent_to_cell_params(x)
            T_fc, T_cf, centroid, orientation = cb.T_fc, cb.T_cf, cb.aunit_centroid, cb.aunit_orientation
            grad = torch.zeros_like(x)
            for c in ctx.chunks:
                r = c.rows
                sel = select_images(c.tables, T_fc[r], T_cf[r], centroid[r], orientation[r], self.label_cutoff,
                                    max_images=wider * self.max_images)
                pairs = pair_distances(c.tables, sel, self.label_cutoff, max_pairs=wider * self.max_pairs)
                capped[r] = sel['capped'] | sel['image_capped'] | pairs['pair_capped']
                e_atom = compressed_elj_per_atom(pairs, vdw, self.lj_coeff, self.temperature, self.compress_at,
                                                 int(c.z.numel()))
                grad = grad + torch.autograd.grad(e_atom.sum(), x, retain_graph=True)[0]
        return -grad, capped

    def agreement(self, state, ctx: TrunkContext, step_variance: float):
        """How well the trunk's force matches its target at `state`: medians over the rows where both
        are finite and the target's pair list was not capped (`rows` of them; `capped` counts the rows
        left out for a cap). `step_variance` is one trajectory step's noise variance, so the two
        `*_sigma` numbers are the displacement of a full-strength force step, target and error, in
        units of that step's noise standard deviation (RMS over the latent's coordinates)."""
        force = self(state, ctx, False)
        target, capped = self.target_force(state, ctx)
        ok = torch.isfinite(force).all(1) & torch.isfinite(target).all(1) & ~capped
        if not bool(ok.any()):
            return {'cosine': float('nan'), 'cosine_p10': float('nan'), 'err_sigma': float('nan'),
                    'target_sigma': float('nan'), 'rows': 0, 'capped': int(capped.sum())}
        f, t = force[ok], target[ok]
        cos = torch.nn.functional.cosine_similarity(f, t, dim=1)
        scale = math.sqrt(step_variance)
        return {'cosine': float(cos.median()), 'cosine_p10': float(cos.quantile(0.1)),
                'err_sigma': float((scale * (f - t).square().mean(1).sqrt()).median()),
                'target_sigma': float((scale * t.square().mean(1).sqrt()).median()),
                'rows': int(ok.sum()), 'capped': int(capped.sum())}


class CrystalDriftForce:
    """The provider a crystal trainer installs on its samplers (`GFN.install_drift_force`).

    A trajectory function is given the rows' molecules as `mol_batch`; the reward turns them
    into crystals with the energy function's own builder, and so does this: `context` calls
    `energy_function.init_blank_crystal_batch(mol_batch)`, the batch
    `MolecularCrystal.instantiate_crystals` applies the latent to, so a state's force is for
    the crystal its reward would be computed on.
    """

    def __init__(self, trunk_force: TrunkForce, energy_function):
        self.trunk_force = trunk_force
        self.energy_function = energy_function

    def __deepcopy__(self, memo):
        # a copied sampler (the EMA model, a frozen prior model) shares the provider: it holds a
        # frozen trunk and the energy function, neither of which belongs to the model being copied
        return self

    def context(self, mol_batch) -> TrunkContext:
        if mol_batch is None:
            raise ValueError("a force term needs the rows' molecules: this trajectory call was given "
                             "mol_batch=None and no drift_context")
        return self.trunk_force.context(self.energy_function.init_blank_crystal_batch(mol_batch))

    def __call__(self, state, ctx: TrunkContext, create_graph: bool = False):
        return self.trunk_force(state, ctx, create_graph)
