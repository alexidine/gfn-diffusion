"""Resume proof for the persisted P_B snapshot.

    python configs/local_pb_freeze/make_resume.py

Two configs off pbab_frozen.yaml:
  pbab_resume_a  no freeze key; the equilibration stage's on_enter gains the
                 `freeze_pb` action, so the freeze is taken at the stage
                 boundary (the measured place) and the 'running' checkpoint
                 must carry the snapshot. 320 steps (~120 frozen).
  pbab_resume_b  resumes A's running checkpoint (full load) to 420 steps. Its
                 log must say the snapshot came from the checkpoint, and the
                 snapshot tensors in B's final checkpoint must equal A's.
"""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
base = yaml.safe_load((HERE / 'pbab_frozen.yaml').read_text())
base.pop('freeze_backward_policy', None)

a = dict(base)
a['run_name'] = 'pbab_resume_a'
a['epochs'] = 320
eq = [s for s in a['protocols']['prod_eq']['stages'] if s['name'] == 'equilibration'][0]
eq['on_enter'] = list(eq.get('on_enter') or []) + ['freeze_pb']
(HERE / 'pbab_resume_a.yaml').write_text(yaml.safe_dump(a, sort_keys=False))

b = yaml.safe_load(yaml.safe_dump(a))
b['run_name'] = 'pbab_resume_b'
b['epochs'] = 420
b['load_weights_only'] = False
b['checkpoint_name'] = 'pbab_pbab_resume_a_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1_running.pt'
(HERE / 'pbab_resume_b.yaml').write_text(yaml.safe_dump(b, sort_keys=False))
print('wrote pbab_resume_a / pbab_resume_b')
