import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

search_output_dir = r"D:\crystal_datasets\acridine"
identifiers = None # ['ACRDIN01', 'ACRDIN12']
energy_function = 'mace'
target_path = r"D:\crystal_datasets\acridine\canonical_structures_chunk_0.pkl"
mol_path = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
uma_model_path = r"D:\crystal_datasets\esen_s.pt"
mace_model_path = r"C:\Users\mikem\Downloads\acr_112025_mh1_stagetwo.model"
device = 'cuda'

if energy_function == 'uma':
    predictor = init_uma_crystal_predictor(uma_model_path, device)
elif energy_function == 'mace':
    predictor = load_mace_model(mace_model_path, device, torch.float32)

target_mols = torch.load(target_path, weights_only=False)
if identifiers is not None:
    target_mols = [elem for elem in target_mols if elem.identifier in identifiers]

#target_mols = [elem for elem in target_mols if elem.z_prime==2]
tbatch = collate_data_list(target_mols)

# standardize symmetry operations
tbatch.aunit_handedness = torch.abs(tbatch.aunit_handedness)  # for flat molecules this is relatively safe
tbatch, succeeded = tbatch.compute_standard_cell(confirm_transform=True)

mol = torch.load(mol_path, weights_only=False)
zp_sg_to_search = [[elem.z_prime, elem.sg_ind] for elem in target_mols]
max_zp = max(tbatch.z_prime)

samples_to_optim = []
ones3 = torch.ones(3, device='cpu')
ones1 = torch.ones(1, device='cpu')
print("Initializing crystals to optimize")
for zp, sg in zp_sg_to_search:
    opt_sample = MolCrystalData(
        molecule=[mol.clone() for _ in range(zp)] if zp > 1 else mol.clone(),
        # duplicate molecules here
        sg_ind=sg,
        aunit_handedness=ones1,
        cell_lengths=ones3,
        cell_angles=ones3,
        aunit_centroid=ones3,
        aunit_orientation=ones3,
        skip_box_analysis=True,
        max_z_prime=max_zp,
        z_prime=zp,
        do_box_analysis=True,  # need this just to instantiate the tensors
    )
    samples_to_optim.append(opt_sample)

batch = collate_data_list(samples_to_optim)

batch.latent_to_cell_params(tbatch.latent_params())
batch.identifier = tbatch.identifier

tbatch.analyze(['elj','rdf'], assign_outputs=True)
batch.analyze(['elj','rdf'], assign_outputs=True)

diff = (tbatch.elj - batch.elj).abs()/batch.elj.abs()
bins = torch.linspace(0, 10, 100)
rdiff = compute_rdf_distance(tbatch.rdf, batch.rdf, bins)
penalty = batch.compute_cell_reduction_penalty()

torch.save(batch,r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt")

aa = 1