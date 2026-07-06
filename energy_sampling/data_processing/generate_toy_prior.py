import os
from copy import deepcopy

import torch
import numpy as np

path = r'D:\crystal_datasets\conditional\priors'
os.chdir(path)
p1 = torch.load('14_1_elj.pt', weights_only=False)  # load a dummy file

enfunc = 'latent_harmonic'

p1['prior'].reset_sg_info(1)
p1['equalized_prior'].reset_sg_info(1)

prior = p1['prior']
prior.reset_sg_info(1)

eprior = p1['equalized_prior']
eprior.reset_sg_info(1)

big_prior = eprior.subsample_new_batch(np.random.choice(len(eprior), 100000, replace=True))

l1 = p1['prior'].latent_params()
l2 = big_prior.latent_params()

sbatch = deepcopy(prior)
condition = torch.zeros(8)
width = 0.25
target_temperature = 1.0

if enfunc == 'latent_harmonic':
    new_lat = sbatch.sample_latent_harmonic(n_samples=prior.num_graphs,
                                            width=width,
                                            target_temperature=target_temperature,
                                            )
    new_lat2 = sbatch.sample_latent_harmonic(n_samples=big_prior.num_graphs,
                                             width=width,
                                             target_temperature=target_temperature,
                                             )
elif enfunc == 'latent_multiharmonic':
    new_lat = sbatch.sample_latent_multiharmonic(n_samples=prior.num_graphs,
                                                 c=condition,
                                                 width=width,
                                                 target_temperature=target_temperature
                                                 )
    new_lat2 = sbatch.sample_latent_multiharmonic(n_samples=big_prior.num_graphs,
                                                  c=condition,
                                                  width=width,
                                                  target_temperature=target_temperature
                                                  )
prior.latent_to_cell_params(new_lat)
big_prior.latent_to_cell_params(new_lat2)

prior_energy = prior.analyze([enfunc],
                             width=width,
                             c=condition,
                             assign_outputs=True)[enfunc]
big_prior_energy = big_prior.analyze([enfunc],
                                     width=width,
                                     c=condition,
                                     assign_outputs=True)[enfunc]

p1['prior'] = prior
p1['equalized_prior'] = big_prior

import plotly.graph_objects as go

go.Figure(go.Histogram(x=prior[enfunc].cpu().detach().numpy(),
                       nbinsx=100)).show()
go.Figure(go.Histogram(x=big_prior[enfunc].cpu().detach().numpy(),
                       nbinsx=100)).show()
big_prior.plot_batch_cell_params(space='latent')
prior.plot_batch_staircase(space='latent')

del prior.fingerprint
del big_prior.fingerprint

torch.save(p1, f'{enfunc}_T{target_temperature}_sig{width}_cond0.pt')

aa = 1
