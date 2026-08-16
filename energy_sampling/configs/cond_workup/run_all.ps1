# cond_workup battery -- run sequentially from energy_sampling/.
# Comment out lines to skip runs. Each pair: bake artifacts, then train.
$env:PYTHONPATH = "C:\Users\mikem\Projects\mxt_gfn\mxtaltools;C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion"

# --- cw00: baseline: 2d interpolate, n1000, r4 ---
python data_processing\generate_toy_prior.py configs\cond_workup\0_toy.yaml
python train.py --config configs\cond_workup\0_train.yaml

# --- cw01: cond count: sparse ---
python data_processing\generate_toy_prior.py configs\cond_workup\1_toy.yaml
python train.py --config configs\cond_workup\1_train.yaml

# --- cw02: cond count: dense (tracker-sparsity probe) ---
python data_processing\generate_toy_prior.py configs\cond_workup\2_toy.yaml
python train.py --config configs\cond_workup\2_train.yaml

# --- cw03: noise: clean manifold ---
python data_processing\generate_toy_prior.py configs\cond_workup\3_toy.yaml
python train.py --config configs\cond_workup\3_train.yaml

# --- cw04: noise: broad ---
python data_processing\generate_toy_prior.py configs\cond_workup\4_toy.yaml
python train.py --config configs\cond_workup\4_train.yaml

# --- cw05: replicas: minimal (floor) ---
python data_processing\generate_toy_prior.py configs\cond_workup\5_toy.yaml
python train.py --config configs\cond_workup\5_train.yaml

# --- cw06: cond_dim 8 ---
python data_processing\generate_toy_prior.py configs\cond_workup\6_toy.yaml
python train.py --config configs\cond_workup\6_train.yaml

# --- cw07: richer GMM field ---
python data_processing\generate_toy_prior.py configs\cond_workup\7_toy.yaml
python train.py --config configs\cond_workup\7_train.yaml

# --- cw08: uniform conditions ---
python data_processing\generate_toy_prior.py configs\cond_workup\8_toy.yaml
python train.py --config configs\cond_workup\8_train.yaml
