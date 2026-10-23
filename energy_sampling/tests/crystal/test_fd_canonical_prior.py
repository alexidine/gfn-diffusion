"""data_processing/fd_canonical_prior.py: an sg 2 prior rewritten as canonical forms, one row per crystal.

Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_fd_canonical_prior.py

What each test would catch:

  * the images of one crystal not collapsing to one output row, or two crystals merged;
  * an image the writer moved at the box edge (a y centre clamped to 0.9999) becoming a second row, or taking its
    moved pose into the output instead of the unmoved image's;
  * the stored energy or any other per-row field (an extra stamp, the identifier) recomputed or dropped, and the
    top-level per-row tensors (source_row, image_id, is_anchor) not following the output rows;
  * a row whose canonical cell the trainer's latent clamp would move (owner D7) written instead of refused and
    counted;
  * a pose field left in the old basis when the cell reduction changes the basis (the checks rebuild atoms and
    eLJ);
  * an output file that the trainer's own prior loaders refuse or read differently: ``train.mirrored_rows``,
    ``Modeller._refuse_mirrored_rows``, ``Modeller._load_condition_file``, ``buffer.strip_lazy_sg_caches``,
    ``buffer._uniform_lj_coeff`` and ``Modeller.init_prior_dataset`` (the real MolecularCrystal of the canonical
    config, CPU), with the trainer-read eLJ of each output row equal to that of its source row;
  * a file without source grouping not deduplicated within tolerance, or one of its two batch keys dropped;
  * an existing output overwritten;
  * the file or manifest not saying that the rows fill one quarter of the trainer's sg 2 latent box, or the
    source's trainer-box rows not recovered by spreading each output row over its four images;
  * a D7 refusal that cannot be re-checked from the provenance alone (canonical pose and judged latents);
  * a top-level row count left at the source's value, or a count of something else overwritten; the count of the
    other file of a split (``--companion``) left at its source value;
  * ``--replace`` deleting the earlier output instead of moving it aside, replacing a set this script did not
    write from the same source, or overwriting an earlier superseded set;
  * output rows just beyond the one-crystal tolerance not counted by the near-pair census;
  * a source row whose stored centroid the builder clips, or a moved-image source, not reported against its
    stored energy;
  * a source whose every image the writer stored at its x edge not collapsing to one row, or not counted;
  * box fields (T_fc, T_cf, cell_volume) left from the old cell.

CPU, a synthetic molecule, no data drive. The checks run in this process here (the sources are tiny); the
child-process route (``--checks-child``, for sources above ``CHILD_CHECKS_ABOVE``) is exercised by the real
qm9full build, not by this file.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import energy_sampling.train as train
from data_processing import fd_canonical_prior as fcp
from energy_sampling import buffer
from mxtaltools.crystal_fd import charts, ops
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

pytestmark = pytest.mark.fast   # torch imported; CPU, synthetic rows of one 5-atom molecule, no data drive

TSF = 0.3635836825309169
SHIFTS = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5))   # pool_anchors.Chart.images, sg 2


def _molecule():
    """A rigid five-atom molecule at the orient_molecule('std') fixed point, as the prior files store it."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [2.1, 1.3, 0.2], [-0.7, 1.2, -0.3], [0.3, -1.3, 0.5]])
    z = torch.tensor([6, 6, 7, 8, 6])
    m = MolData(z=z, pos=pos, x=torch.zeros(5), identifier='syn', mol_volume=torch.tensor(70.0),
                mass=torch.tensor(68.0), radius=torch.tensor(1.6), z_prime=torch.tensor(1, dtype=torch.long))
    b = collate_data_list([m])
    b.orient_molecule(mode='std')
    pos = b.pos.clone()
    radius = float((pos - pos.mean(0)).norm(dim=1).max()) + 0.2
    return MolData(z=z, pos=pos, x=torch.zeros(5), identifier='syn', mol_volume=torch.tensor(70.0),
                   mass=torch.tensor(68.0), radius=torch.tensor(radius), z_prime=torch.tensor(1, dtype=torch.long))


MOL = _molecule()
R_EFF = float(MOL.radius)


def _row(cell, t, rv, elj, stamp):
    """One stored sg 2 row: cell (lengths in Angstrom, angles in degrees), centroid, rotation vector, energies."""
    return MolCrystalData(molecule=MOL.clone(), sg_ind=2, z_prime=torch.tensor(1, dtype=torch.long),
                          cell_lengths=torch.tensor(cell[:3], dtype=torch.float32),
                          cell_angles=torch.deg2rad(torch.tensor(cell[3:], dtype=torch.float64)).float(),
                          aunit_centroid=torch.tensor(t, dtype=torch.float32),
                          aunit_orientation=torch.tensor(rv, dtype=torch.float32),
                          aunit_handedness=torch.tensor([1.0]), identifier='syn', do_box_analysis=True,
                          elj=torch.tensor([elj]), lj=torch.tensor([0.0]), reduction_en=torch.tensor([0.0]),
                          energy_stamp=torch.tensor([stamp]))


def _images(cell, t, rv, elj, stamp, clamp=False):
    """The four Gamma+ images the pool writer stores for a source; ``clamp`` moves a y or z coordinate landing in
    (0.9999, 1) to float32(0.9999), as pool_anchors.Chart.describe does."""
    rows = []
    for dy, dz in SHIFTS:
        c = [t[0], (t[1] + dy) % 1.0, (t[2] + dz) % 1.0]
        if clamp:
            c = [c[0]] + [0.9999 if 0.9999 < v < 1 else v for v in c[1:]]
        rows.append(_row(cell, c, rv, elj, stamp))
    return rows


# crystal P: generic; Q: y at 0.49996, so the dy = 1/2 images land in the clip band and are clamped; D: c beyond the
# trainer's latent length box (sg 2: normed c = c / (2 r) <= 4), the D7 refusal; B: a crystal on P's lattice with
# another pose, stored in a non-reduced basis (b' = a + b), so reduce_cell changes the basis and every pose field
# must follow
P = ((5.9, 6.6, 7.6, 97.0, 101.0, 104.0), (0.13, 0.31, 0.27), (0.4, -0.3, 1.9), -61.5, 11.0)
Q = ((6.1, 6.8, 7.3, 95.0, 99.0, 102.0), (0.21, 0.49996, 0.18), (-0.2, 0.5, 2.3), -55.25, 22.0)
D = ((6.0, 6.7, 10.0 * R_EFF, 93.0, 97.0, 99.0), (0.3, 0.2, 0.1), (0.1, 0.2, 1.5), -40.0, 33.0)
B_POSE = (P[0], (0.37, 0.12, 0.44), (1.0, 0.2, 0.8), -48.0, 44.0)


def _non_reduced(cell, t, rv):
    d = CrystalDescription.from_crystal_params(2, torch.tensor([cell[:3]], dtype=torch.float64),
                                               torch.deg2rad(torch.tensor([cell[3:]], dtype=torch.float64)),
                                               torch.tensor([t], dtype=torch.float64),
                                               torch.tensor([rv], dtype=torch.float64))
    N = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    e = ops.change_basis(d, N, torch.zeros(1, 3, dtype=torch.float64))
    L, A, c, r, _ = e.crystal_params()
    return tuple(L[0].tolist()) + tuple(torch.rad2deg(A[0]).tolist()), tuple((c[0] % 1).tolist()), tuple(r[0].tolist())


def _b_row(elj=B_POSE[3], stamp=B_POSE[4], pose=B_POSE):
    """B (or another crystal given by ``pose``) stored in the non-reduced basis b' = a + b."""
    c, t, r = _non_reduced(*pose[:3])
    return _row(c, t, r, elj, stamp)


def _image_written_source(path):
    rows = _images(*P) + _images(*Q, clamp=True) + _images(*D)
    rows.append(_b_row())
    batch = collate_data_list(rows, max_z_prime=1)
    n = batch.num_graphs
    source_row = torch.tensor([100] * 4 + [200] * 4 + [300] * 4 + [400])
    image_id = torch.tensor([0, 1, 2, 3] * 3 + [0])
    is_anchor = torch.tensor([True] * 4 + [False] * 4 + [False] * 4 + [True])
    blob = {'prior': batch, 'equalized_prior': batch, 'thermal_scaling_factor': TSF, 'source_row': source_row,
            'image_id': image_id, 'is_anchor': is_anchor, 'folded': True, 'anchors_file': '/x/anchors.pt',
            'build': {'note': 'synthetic'}, 'n_anchors_in': 4078, 'n_structures': 13}
    assert n == 13
    torch.save(blob, path)
    return blob


@pytest.fixture(scope='module')
def written(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('fd1')
    src_path = tmp / 'syn_pooled_prior.pt'
    src = _image_written_source(src_path)
    man = fcp.rewrite_file(src_path, check=13, energy_check=13, probe=0)
    data_path, prov_path, man_path = fcp.output_paths(src_path)
    return dict(src=src, src_path=src_path, out=torch.load(data_path, weights_only=False),
                prov=torch.load(prov_path, weights_only=False), man=json.loads(man_path.read_text()),
                man_returned=man, paths=(data_path, prov_path, man_path))


def _out_rows(w):
    return w['prov']['keys']['equalized_prior']['out_row']


# ---------------------------------------------------------------- one row per crystal

def test_images_of_one_crystal_collapse_to_one_row(written):
    out_row = _out_rows(written)
    out = written['out']['equalized_prior']
    assert out.num_graphs == 3                                  # P, Q, B (D refused)
    assert out_row.tolist() == [0] * 4 + [1] * 4 + [-1] * 4 + [2]
    assert written['prov']['keys']['equalized_prior']['group_size'][:4].tolist() == [4] * 4
    assert written['prov']['keys']['equalized_prior']['code'][:4].tolist() == [fcp.KEPT] + [fcp.SAME_CRYSTAL] * 3
    assert written['man']['keys']['equalized_prior']['dedupe']['merged_duplicates'] == 0
    assert written['man']['keys']['equalized_prior']['image_groups']['sources_collapsed'] == 3   # P, D, B


def test_a_moved_image_does_not_make_a_second_row(written):
    prov = written['prov']['keys']['equalized_prior']
    out_row, code = prov['out_row'], prov['code']
    assert len(set(out_row[4:8].tolist())) == 1 and int(out_row[4]) >= 0
    assert code[4:8].tolist() == [fcp.KEPT, fcp.SAME_CRYSTAL, fcp.MOVED_IMAGE, fcp.MOVED_IMAGE]
    assert prov['moved_image'][4:8].tolist() == [False, False, True, True]
    cases = written['man']['moved_image_cases']['equalized_prior']
    assert len(cases) == 1 and cases[0]['moved_image_ids'] == [2, 3] and cases[0]['rep_image_id'] == 0
    assert all(s == 'clamp' for s in cases[0]['signatures'])
    # the output takes the unmoved image's canonical form: y = 0.49996, not the clamped 0.9999 - 1/2
    j = int(out_row[4])
    y = float(written['out']['equalized_prior'].aunit_centroid[j, 1])
    assert abs(y - 0.49996) < 1e-6, y
    mi = written['man']['checks']['equalized_prior']['moved_images']
    assert mi['output_vs_unmoved_image_A']['max'] < 1e-4 < mi['output_vs_moved_image_A']['max']


def test_stored_energy_and_per_row_fields_are_carried(written):
    out, src = written['out'], written['src']
    b, sb = out['equalized_prior'], src['equalized_prior']
    rep = written['prov']['keys']['equalized_prior']['source_row_of_output']
    assert rep.tolist() == [0, 4, 12]
    assert torch.equal(b.elj, sb.elj[rep]) and torch.equal(b.energy_stamp, sb.energy_stamp[rep])
    assert b.elj.tolist() == pytest.approx([P[3], Q[3], B_POSE[3]])
    assert b.energy_stamp.tolist() == pytest.approx([P[4], Q[4], B_POSE[4]])
    assert list(b.identifier) == ['syn'] * b.num_graphs
    assert out['source_row'].tolist() == rep.tolist()           # remapped to the source file's row index
    assert out['image_id'].tolist() == [0] * b.num_graphs
    assert torch.equal(out['is_anchor'], src['is_anchor'][rep])
    assert out['prior'] is out['equalized_prior']
    for k in ('thermal_scaling_factor', 'folded', 'anchors_file', 'build', 'n_anchors_in'):
        assert out[k] == src[k], k
    assert 'fd_canonical' in out and set(src) | {'fd_canonical'} == set(out)
    chk = written['man']['checks']['equalized_prior']
    assert chk['carried']['mismatched'] == [] and chk['carried']['energy_carried']
    assert torch.equal(b.aunit_handedness, torch.ones_like(b.aunit_handedness))


def test_a_d7_row_is_refused_and_counted(written):
    prov = written['prov']['keys']['equalized_prior']
    assert prov['code'][8:12].tolist() == [fcp.CODE['latent_clamp_cell_length']] * 4
    assert prov['out_row'][8:12].tolist() == [-1] * 4
    assert prov['refused_rows']['latent_clamp_cell_length'].tolist() == [8, 9, 10, 11]
    assert all(int(m) & (1 << 2) for m in prov['latent_clamp_mask'][8:12])     # latent entry c
    st = written['man']['keys']['equalized_prior']
    assert st['rows_by_code']['latent_clamp_cell_length'] == 4 and st['rows_refused'] == 4
    assert st['d7']['by_code']['latent_clamp_cell_length'] == 1                # one source, four rows
    # and every row that was written reads through the trainer's latent round trip unclamped
    chk = written['man']['checks']['equalized_prior']
    assert chk['trainer_build']['latent_round_trip_param_dev']['max'] < 1e-4


def test_a_d7_refusal_can_be_rechecked_from_the_provenance(written):
    """The provenance holds the refused representative's canonical pose and the latents it was judged on: rebuilding
    the latents from that pose (and the row's radius) gives the same values, outside the recorded clamp."""
    prov = written['prov']['keys']['equalized_prior']
    d7 = prov['d7']
    assert d7['source_rows'].tolist() == [8] and d7['code'].tolist() == [fcp.CODE['latent_clamp_cell_length']]
    lo, hi = d7['clamp_lo'], d7['clamp_hi']
    assert float(d7['latents'][0, 2]) > float(hi[2])
    desc = CrystalDescription.from_crystal_params(2, prov['canonical_cell'][8:9, :3].double(),
                                                  prov['canonical_cell'][8:9, 3:].double(),
                                                  prov['canonical_t'][8:9].double(), prov['canonical_rotvec'][8:9].double())
    radius = written['src']['equalized_prior'].radius[8:9].float()
    lat = charts.LatentV1Chart(2, radius).latents(desc)
    assert torch.allclose(lat, d7['latents'], atol=1e-5), (lat, d7['latents'])
    # and the file as written holds no row the clamp would move, nor a centroid in the clip band
    chk = written['man']['checks']['equalized_prior']['loader']['d7_on_written_rows']
    assert chk == dict(rows=3, latent_outside_live_clamp=0, centroid_in_clip_band=0)


def test_the_rows_fill_one_quarter_of_the_trainer_box_and_say_so(written):
    """Each output row is the y < 1/2, z < 1/2 description; its four trainer-box images recover every unmoved
    source row (basis unchanged) to float32 storage noise, and the file and the manifest carry the note."""
    assert written['out']['fd_canonical']['centroid_domain'] == fcp.CENTROID_DOMAIN_NOTE
    assert fcp.CENTROID_DOMAIN_NOTE.startswith('NOT A DROP-IN PRIOR FOR THE CURRENT TRAINER')
    assert 'spread each row over its 4 images' in fcp.CENTROID_DOMAIN_NOTE
    assert written['out']['fd_canonical']['k_box'] == fcp.K_BOX == 4
    assert written['man']['centroid_domain'] == fcp.CENTROID_DOMAIN_NOTE
    tb = written['man']['checks']['equalized_prior']['trainer_box']
    assert tb['output_share'] == [1.0, 0.0, 0.0, 0.0]
    assert tb['images_per_sampled_output_row'] == {'4': 3}
    rec = tb['source_rows_vs_nearest_image']
    assert rec['source_rows'] == 8 and rec['skipped_basis_changed'] == 1    # P 4 + Q 4; B's basis changed
    assert rec['unmoved_rows_dev_gt_1em6'] == 0 and rec['dev_unmoved']['max'] < 1e-6
    assert rec['moved_rows'] == 2 and rec['dev_moved']['p50'] > 1e-6        # Q's two clamped images


def test_top_level_row_counts_follow_the_output_and_other_counts_are_kept(written):
    out, man = written['out'], written['man']
    assert out['n_structures'] == 3 and out['n_anchors_in'] == 4078
    upd = man['counts']['updated']
    assert upd['n_structures']['source'] == 13 and upd['n_structures']['output'] == 3
    assert man['counts']['kept'] == {'n_anchors_in': 4078}
    assert out['fd_canonical']['counts']['updated']['n_structures']['output'] == 3


def test_a_source_clamp_and_the_box_fields_are_reported(written):
    st = written['man']['keys']['equalized_prior']
    # no source centroid sits in the clip band here (the writer clamped Q's images to 0.9999 itself)
    assert st['source_clip_band']['source_rows'] == 0
    assert set(st['stored_x_at_writer_edge']) == {'x_eq_f32_0.4999', 'x_eq_f32_0.9999_minus_half', 'x_eq_0'}
    chk = written['man']['checks']['equalized_prior']
    box = chk['recanonical']['box_fields_vs_written_cell']
    assert box['T_fc_max_abs'] < 1e-5 and box['cell_volume_max_rel'] < 1e-6
    mi = chk['moved_images']
    assert mi['cases'] == 1 and mi['moved_rows'] == 2 and 'stored_minus_rescored_output' in mi


def test_a_source_written_at_the_x_edge_on_every_image_is_one_row_and_counted():
    """Four images of a crystal whose x the writer stored at float32(0.4999): one output row, recorded."""
    cell, t, rv, elj, stamp = P
    rows = _images(cell, (0.4999, t[1], t[2]), rv, elj, stamp)
    batch = collate_data_list(rows, max_z_prime=1)
    r = fcp.rewrite_batch(batch, source_row=torch.tensor([7] * 4), image_id=torch.arange(4))
    assert r['stats']['rows_out'] == 1 and r['stats']['rows_refused'] == 0
    rec = r['stats']['stored_x_at_writer_edge']['x_eq_f32_0.4999']
    assert rec == dict(source_rows=4, output_rows=1, sources_every_image=1, sources_some_images=0,
                       output_rows_whose_source_has_it_on_every_image=1)


def test_a_basis_change_moves_every_pose_field_with_it(written):
    st = written['man']['keys']['equalized_prior']
    assert bool(written['prov']['keys']['equalized_prior']['basis_changed'][12])
    assert st['basis_changed_rows_out'] == 1
    b = written['out']['equalized_prior']
    sb = written['src']['equalized_prior']
    assert not torch.allclose(b.cell_lengths[2], sb.cell_lengths[12])        # B's cell was re-expressed
    vol = torch.linalg.det(b.T_fc.double())
    assert torch.allclose(vol, b.cell_volume.double(), rtol=1e-5)            # box fields follow the new cell
    assert torch.allclose(b.cell_volume[2].double(), sb.cell_volume[12].double(), rtol=1e-5)
    chk = written['man']['checks']['equalized_prior']
    assert chk['same_crystal']['min_over_gamma_plus_A']['max'] < 1e-4
    assert chk['same_crystal']['control_wrong_translation_min_A'] > 0.1
    assert chk['energy']['canonical_minus_source_rescored']['max'] < 1e-3
    assert chk['trainer_build']['posed_atoms_vs_description']['max'] < 1e-3
    assert chk['trainer_build']['control_R_transposed']['frac_gt_0p1'] == 1.0
    assert chk['recanonical']['rows_dev_gt_tol'] == 0 and chk['recanonical']['basis_changed'] == 0
    assert st['canonicalise']['shortcut_check']['N_equal']


# ---------------------------------------------------------------- the trainer's loaders, called

def test_the_output_passes_the_trainer_prior_loaders(written, capsys):
    path = str(written['paths'][0])
    data = torch.load(path, weights_only=False)
    prior = data['equalized_prior']
    zp1, zpn = train.mirrored_rows(prior)
    assert zp1.numel() == 0 and zpn.numel() == 0
    stub = SimpleNamespace(energy_function=SimpleNamespace(is_crystal=True, latent_energy=False))
    train.Modeller._refuse_mirrored_rows(stub, prior, path, 'prior_path')
    cond = train.Modeller._load_condition_file(path)
    train.Modeller._refuse_mirrored_rows(stub, cond, path, 'molecules_path')
    assert cond.num_graphs == prior.num_graphs
    buffer.strip_lazy_sg_caches(prior)
    assert buffer._uniform_lj_coeff(prior, 'test') is None     # unstamped, as every prior on disk; init stamps it

    ef, args = fcp._energy_function(fcp.DEFAULT_CONFIG)
    capsys.readouterr()
    m = train.Modeller.__new__(train.Modeller)
    m.energy_function, m.device = ef, 'cpu'
    m.args = SimpleNamespace(prior_path=path, energy_function=args.energy_function, energy_config=args.energy_config,
                             model=SimpleNamespace(hold_dead_latent_rows=True), space_groups=[2], z_primes=[1],
                             prior_model_name=None, checkpoints_dir=str(written['paths'][0].parent),
                             buffer_device='cpu')
    with torch.no_grad(), ef.allow_unreferenced():
        m.init_prior_dataset()
    pb = m.prior_dataset.batch
    assert len(m.prior_dataset) == prior.num_graphs == 3
    assert float(ef.lj_coeff) == pytest.approx(TSF)
    # what the trainer reads off each output row is its source row's crystal
    rep = written['prov']['keys']['equalized_prior']['source_row_of_output']
    e_src, _ = fcp.score(written['src']['equalized_prior'], rep)
    read = pb.elj.double().flatten() / float(ef.lj_coeff)
    assert torch.allclose(read, e_src, rtol=1e-4, atol=1e-3), (read, e_src)
    built = torch.cat([pb.cell_lengths, pb.cell_angles, pb.aunit_centroid, pb.aunit_orientation], 1)
    want = torch.cat([prior.cell_lengths, prior.cell_angles, prior.aunit_centroid, prior.aunit_orientation], 1)
    assert float((built - want).abs().max()) < 1e-4
    assert written['man']['loader_checks_passed']
    # the module's own run of the molecules_path loader, in its child process
    assert written['man']['checks']['_condition_file'] == dict(rows=3, zp1=0, zp_gt1=0, ok=True)


# ---------------------------------------------------------------- a file without source grouping

# P2: P with z moved by 2e-5, a separately stored crystal just beyond the one-crystal tolerance (the near-pair census);
# R: a crystal whose stored y lies in the builder clip band (0.9999, 1), as some nsym rows do (the clip-band report)
P2 = (P[0], (P[1][0], P[1][1], P[1][2] + 2e-5), P[2], P[3] + 0.01, 55.0)
R_BAND = (P[0], (0.31, 0.99995, 0.41), (0.7, -0.2, 1.1), -50.0, 66.0)


def test_a_file_without_images_dedupes_within_tolerance_and_keeps_both_keys(tmp_path):
    anchors = collate_data_list([_row(*P), _row(*Q)], max_z_prime=1)
    # P, a Gamma+ image of P, Q, D, P itself stored in a non-reduced basis, P2 and R
    noised = collate_data_list(_images(*P)[:1] + _images(*P)[2:3] + [_row(*Q), _row(*D), _b_row(pose=P),
                                                                     _row(*P2), _row(*R_BAND)], max_z_prime=1)
    src_path = tmp_path / 'syn_niggli.pt'
    torch.save({'prior': anchors, 'equalized_prior': noised, 'thermal_scaling_factor': TSF,
                'uma_energy_state': None}, src_path)
    fcp.rewrite_file(src_path, check=7, energy_check=7, probe=0)
    data_path, prov_path, man_path = fcp.output_paths(src_path)
    out = torch.load(data_path, weights_only=False)
    prov = torch.load(prov_path, weights_only=False)['keys']
    assert out['prior'].num_graphs == 2 and out['equalized_prior'].num_graphs == 4
    assert out['uma_energy_state'] is None and out['thermal_scaling_factor'] == TSF
    e = prov['equalized_prior']
    assert e['out_row'].tolist() == [0, 0, 1, -1, 0, 2, 3]      # both copies of P merged onto P; D refused
    assert e['code'].tolist() == [fcp.KEPT, fcp.SAME_CRYSTAL, fcp.KEPT, fcp.CODE['latent_clamp_cell_length'],
                                  fcp.SAME_CRYSTAL, fcp.KEPT, fcp.KEPT]
    assert bool(e['basis_changed'][4])
    man = json.loads(man_path.read_text())
    dg = man['duplicate_groups']['equalized_prior']
    assert len(dg) == 1 and dg[0]['leader'] == 0 and dg[0]['merged'] == [1, 4] and dg[0]['chart_dev'] <= fcp.TOL
    assert man['keys']['prior']['rows_out'] == 2 and man['keys']['prior']['rows_refused'] == 0
    st = man['keys']['equalized_prior']
    # P2 is kept apart from P (beyond tol) and counted by the near-pair census, with both stored energies
    census = st['near_pair_census']
    assert census['pairs_within']['1e-05'] == 0 and census['pairs_within']['0.0001'] == 1
    assert fcp.TOL < census['closest_chart_dev'] < 1e-4
    assert census['closest'][0]['out_rows'] == [0, 2]
    assert census['closest'][0]['energy'] == pytest.approx([P[3], P2[3]])
    # R's stored centroid is in the clip band: reported and rescored against its stored energy; its canonical form
    # (y = 0.49995) is not in the band, so it is written
    assert st['source_clip_band']['source_rows'] == 1
    assert st['source_clip_band']['output_rows_whose_representative_is_one'] == 1
    y = float(out['equalized_prior'].aunit_centroid[3, 1])
    assert abs(y - 0.49995) < 1e-6, y
    cb = man['checks']['equalized_prior']['source_clip_band']
    assert cb['output_rows'] == 1 and 'output_rescored_minus_stored' in cb and 'source_rescored_minus_stored' in cb


def test_an_existing_output_is_never_overwritten(written):
    data_path = written['paths'][0]
    before = data_path.stat().st_mtime_ns
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        fcp.rewrite_file(written['src_path'], check=1, energy_check=1, probe=0)
    assert data_path.stat().st_mtime_ns == before


# ---------------------------------------------------------------- a split's two files, and --replace

def _split_file(path, rows, n_structures, n_holdout_structures):
    """A file of a split as build_qm9_full_prior writes it: both files carry the counts of both."""
    b = collate_data_list([_row(*r) for r in rows], max_z_prime=1)
    torch.save({'prior': b, 'equalized_prior': b, 'n_molecules': 1, 'n_structures': n_structures,
                'n_holdout_structures': n_holdout_structures, 'n_anchors_in': 999}, path)


def test_companion_counts_and_replace(tmp_path):
    """train = [P, Q, D] (3 rows in, 2 out), holdout = [Q, D] (2 in, 1 out). The holdout file, rewritten with the
    train file as its companion, updates both counts; the train file, replaced with the holdout file as its
    companion, then does too, and its earlier output is moved aside, not deleted."""
    tr, ho = tmp_path / 'syn_train.pt', tmp_path / 'syn_holdout.pt'
    _split_file(tr, [P, Q, D], 3, 2)
    _split_file(ho, [Q, D], 3, 2)
    kw = dict(check=2, energy_check=2, probe=0)
    fcp.rewrite_file(tr, **kw)
    out_tr = torch.load(fcp.output_paths(tr)[0], weights_only=False)
    assert out_tr['n_structures'] == 2 and out_tr['n_holdout_structures'] == 2   # no companion: kept
    fcp.rewrite_file(ho, companion=tr, **kw)
    out_ho = torch.load(fcp.output_paths(ho)[0], weights_only=False)
    assert out_ho['equalized_prior'].num_graphs == 1
    assert out_ho['n_structures'] == 2 and out_ho['n_holdout_structures'] == 1 and out_ho['n_molecules'] == 1
    assert out_ho['n_anchors_in'] == 999
    comp = out_ho['fd_canonical']['counts']['companion']
    assert comp['source'] == str(tr) and comp['counts']['rows of syn_train[equalized_prior]'] == [3, 2]
    # replace the train file's output, now with the holdout file as its companion
    old = {p: p.read_bytes() for p in fcp.output_paths(tr)}
    man = fcp.rewrite_file(tr, companion=ho, replace=True, **kw)
    out_tr = torch.load(fcp.output_paths(tr)[0], weights_only=False)
    assert out_tr['n_structures'] == 2 and out_tr['n_holdout_structures'] == 1
    sup = tmp_path / fcp.SUPERSEDED_DIR
    for p, content in old.items():
        assert (sup / p.name).read_bytes() == content                        # moved aside intact
    assert [r['moved_to'] for r in man['replaced']] == [str(sup / p.name) for p in fcp.output_paths(tr)]
    assert json.loads(fcp.output_paths(tr)[2].read_text())['replaced'] == man['replaced']
    assert not any(p.with_name(p.name + '.partial').exists() for p in fcp.output_paths(tr))
    # a second replace would overwrite the superseded set: refused, and nothing moves
    with pytest.raises(SystemExit, match='earlier superseded set'):
        fcp.rewrite_file(tr, replace=True, **kw)
    assert all(p.exists() for p in fcp.output_paths(tr))
    # replace accepts only this script's output from the same source
    other = tmp_path / 'syn_other.pt'
    _split_file(other, [P], 1, 1)
    with pytest.raises(SystemExit, match='no earlier output'):
        fcp.rewrite_file(other, replace=True, **kw)
    for p, q in zip(fcp.output_paths(ho), fcp.output_paths(other)):
        q.write_bytes(p.read_bytes())                                         # the holdout file's set, renamed
    with pytest.raises(SystemExit, match='does not record this script'):
        fcp.rewrite_file(other, replace=True, **kw)
    assert not (tmp_path / fcp.SUPERSEDED_DIR / fcp.output_paths(other)[0].name).exists()


def test_the_d7_clamp_is_the_live_clamp():
    """The refusal reads the live latent_to_cell_params clamp, not a copy of it: D's canonical latent c is above
    the bound LatentV1Chart reads off the live code, P's is inside every bound."""
    rows = collate_data_list([_row(*P), _row(*D)], max_z_prime=1)
    cn = fcp.canonicalise(rows)
    mask, band, flagged = fcp.latent_clamp(cn, torch.arange(2), rows.radius)
    assert int(mask[0]) == 0 and int(mask[1]) & (1 << 2) and not bool(band.any())
    assert flagged['index'].tolist() == [1] and float(flagged['latents'][0, 2]) > float(flagged['hi'][2])
    lat = charts.LatentV1Chart(2, rows.radius).latents(fcp.canonical_description(cn, torch.arange(2)))
    lo, hi = charts.LatentV1Chart(2, rows.radius).clamp_bounds()
    assert float(lat[1, 2]) > float(hi[2]) and bool(((lat[0] >= lo) & (lat[0] <= hi)).all())
    assert np.isclose(float(hi[2]), 1.0) and np.isclose(float(lo[2]), -0.99)
