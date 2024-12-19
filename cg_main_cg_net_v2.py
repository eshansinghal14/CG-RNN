import numpy as np
import torch.nn as nn
import torch
import mdtraj as md
import random
import os
from scipy.special import rel_entr
from sklearn.metrics import mean_squared_error
from test_v2 import *

from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import RNN, GRU, LSTM, Conv1d, Conv2d, Conv3d

from cgnet.cgnet.feature import (MoleculeDataset, GeometryStatistics,
                                 GeometryFeature, ShiftedSoftplus,
                                 CGBeadEmbedding, SchnetFeature,
                                 FeatureCombiner, LinearLayer,
                                 GaussianRBF, SingleOutRNN)
from cgnet.cgnet.network import (HarmonicLayer, CGnet, ForceLoss,
                                 lipschitz_projection, Simulation)
from cgnet.cgnet.molecule import CGMolecule

import pyemma
import pyemma.plots
import pyemma.coordinates as coor

import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import GRO, XTC

# We specify the CPU as the training/simulating device here.
# If you have machine  with a GPU, you can use the GPU for
# accelerated training/simulation by specifying
# device = torch.device('cuda')
device = torch.device('cpu')

# nn_type = "Baseline_NN"  # Baseline NN
# nn_type = "GRU_LinearLayer"  # GRU+LinearLayer
nn_type = "Regular_RNN"  # Regular RNN

protein = 'CHIG'
n_files = 3740  # Chignolin = 3740 max files
num_epochs = 50

data_folder = '/Users/eshan/Desktop/Course Graining Garcia/'

if protein == 'ALA2':
    coords = np.load('Extra/data/ala2_coordinates.npy')
    forces = np.load('Extra/data/ala2_forces.npy')
    n_frames = n_files * 500
if protein == 'CHIG':
    coords, n_frames, traj_lens = read_chig('coords_nowater', n_files)
    forces = read_chig('forces_nowater', n_files)[0]
    # psf = 'data/aa_chignolin.psf'
    # pdb = 'data/aa_chignolin.pdb'
    # coords = np.load('chig')

if protein == 'ALA2':
    n_beads = 5
if protein == 'CHIG':
    n_beads = 10

coords = coords[:n_frames]
forces = forces[:n_frames]
# coords = map(float, coords[:n_frames])
# coords = np.double(coords[:n_frames])
# forces = np.double(forces[:n_frames])
print(coords)
print('Num Frames:')
print(n_frames)

print("Coordinates size: {}".format(coords.shape))
print("Force: {}".format(forces.shape))

if protein == 'ALA2':
    embeddings = np.tile([6, 7, 6, 6, 7], [coords.shape[0], 1])
if protein == 'CHIG':
    embeddings = np.tile([6] * 10, [coords.shape[0], 1])
print("Embeddings size: {}".format(embeddings.shape))

ala_data = MoleculeDataset(coords, forces, embeddings, device=device)
print("Dataset length: {}".format(len(ala_data)))

stats = GeometryStatistics(coords, backbone_inds='all', get_all_distances=True, get_backbone_angles=True,
                           get_backbone_dihedrals=True)

bond_list, bond_keys = stats.get_prior_statistics(features='Bonds', as_list=True)
bond_indices = stats.return_indices('Bonds')

angle_list, angle_keys = stats.get_prior_statistics(features='Angles', as_list=True)
angle_indices = stats.return_indices('Angles')
print("We have {} backbone beads, {} bonds, and {} angles.".format(coords.shape[1], len(bond_list), len(angle_list)))
print("Bonds: ")
for key, stat in zip(bond_keys, bond_list):
    print("{} : {}".format(key, stat))
print("Angles: ")
for key, stat in zip(angle_keys, angle_list):
    print("{} : {}".format(key, stat))

all_stats, _ = stats.get_prior_statistics(as_list=True)
num_feats = len(all_stats)
zscores, _ = stats.get_zscore_array()
print("We have {} statistics for {} features.".format(zscores.shape[0], zscores.shape[1]))
# print(all_stats)
print(num_feats)

n_beads = coords.shape[1]
geometry_feature = GeometryFeature(feature_tuples='all_backbone', n_beads=n_beads, device=device)

# hyperparameters
n_layers = 5
n_nodes = 128
activation = ShiftedSoftplus()
batch_size = 512 # Previously 512
learning_rate = 3e-4
rate_decay = 0.3
lipschitz_strength = 4.0
output_size = 8

# schnet-specific parameters
n_embeddings = 10
n_gaussians = 50
cutoff = 5.0
n_interaction_blocks = 3

save_model = True
directory = '.'  # to save model

embedding_layer = CGBeadEmbedding(n_embeddings=n_embeddings, embedding_dim=n_nodes)
rbf_layer = GaussianRBF(high_cutoff=cutoff, n_gaussians=n_gaussians)
schnet_feature = SchnetFeature(feature_size=n_nodes,
                               embedding_layer=embedding_layer,
                               rbf_layer=rbf_layer,
                               n_interaction_blocks=n_interaction_blocks,
                               calculate_geometry=False,
                               n_beads=n_beads,
                               neighbor_cutoff=None,
                               device=device)
distance_feature_indices = stats.return_indices('Distances')
layer_list = [geometry_feature, schnet_feature]
feature_combiner = FeatureCombiner(layer_list, distance_indices=distance_feature_indices)

# RNN Specific Modification Parameters
cell_type = 'GRU' # 'GRU' or 'LSTM'
n_rnn_layers = 5
rnn_dropout = 0
nonlinearity = 'relu'
n_rnn_nodes = 128
n_ll_nodes = 100
n_ll_layers = 3
activation = ShiftedSoftplus()

def baseline_nn():
    layers = []
    layers += LinearLayer(n_nodes, n_nodes, activation=activation)
    for _ in range(n_layers - 1):
        layers += LinearLayer(n_nodes, n_nodes, activation=activation)
    layers += LinearLayer(n_nodes, 1, activation=None)
    return layers


def reg_rnn():
    # 1, 3, 5 RNN layers
    # 1 and baseline method linear layers
    # Change rnn output nodes
    layers = []
    layers += [RNN(input_size=n_nodes, hidden_size=100, num_layers=3, dropout=0.2, nonlinearity='relu')]
    layers += [SingleOutRNN()]
    layers += LinearLayer(100, 1, activation=None)
    # layers += Line
    return layers


def config_num():
    layers = []
    if cell_type == 'RNN':
        layers += [RNN(input_size=n_rnn_nodes, hidden_size=n_ll_nodes, num_layers=n_rnn_layers, dropout=rnn_dropout)]
    if cell_type == 'GRU':
        layers += [GRU(input_size=n_rnn_nodes, hidden_size=n_ll_nodes, num_layers=n_rnn_layers, dropout=rnn_dropout)]
    if cell_type == 'LSTM':
        layers += [LSTM(input_size=n_rnn_nodes, hidden_size=n_ll_nodes, num_layers=n_rnn_layers, dropout=rnn_dropout)]
    layers += [SingleOutRNN()]
    for _ in range(n_ll_layers - 1):
        layers += LinearLayer(n_ll_nodes, n_ll_nodes, activation=activation)
    layers += LinearLayer(n_ll_nodes, 1, activation=None)
    return layers


if nn_type == 'Baseline_NN':
    layers = baseline_nn()
elif nn_type == 'Regular_RNN':
    layers = reg_rnn()
elif nn_type == 'ALA2 Results':
    layers = config_num()
else:
    layers = config_num()

priors = [HarmonicLayer(bond_indices, bond_list)]
priors += [HarmonicLayer(angle_indices, angle_list)]

ala2_net = CGnet(layers, ForceLoss(), feature=feature_combiner, priors=priors).to(device)

# print(ala2_net)

# Training tools
trainloader = DataLoader(ala_data, sampler=RandomSampler(ala_data),
                         batch_size=batch_size)
optimizer = torch.optim.Adam(ala2_net.parameters(),
                             lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50],
                        gamma=rate_decay)
epochal_train_losses = []
epochal_test_losses = []
verbose = True

# printout settings
batch_freq = 500
epoch_freq = 1

test_split = 0.05  # 0.05
val_split = 0.05  # 0.05
# save best model based on validation loss
# set random seed
train_test_arr = []
for i in range(n_files):
    if random.randint(0, 100) > val_split * 100:
        train_test_arr.append(True)
    else:
        train_test_arr.append(False)
print(train_test_arr)

best_test_loss = float('inf')
best_model = ala2_net
if not os.path.exists(data_folder + nn_type + '/model.pt'):
    print('model not found, starting new')
    best_model = ala2_net
else:
    print('model found')
    best_model = torch.load(data_folder + nn_type + '/model.pt')
    ala2_net = best_model
    print(ala2_net)

for epoch in range(1, num_epochs + 1):
    train_loss = 0.00
    test_loss = 0.00
    train_n = 0
    test_n = 0
    coord_counter = 0
    for i in range(n_files):
        optimizer.zero_grad()
        if protein == 'CHIG':
            embedding_property = torch.from_numpy(np.full((traj_lens[i], 10), 6))
            # print(embedding_property.shape)
        coord = torch.from_numpy(coords[coord_counter: coord_counter + traj_lens[i]])
        force = torch.from_numpy(forces[coord_counter: coord_counter + traj_lens[i]])
        coord.requires_grad = True
        # print(coord.shape)
        # print(i)
        coord = coord.float()
        energy, pred_force = ala2_net.forward(coord, embedding_property=embedding_property)
        batch_loss = ala2_net.criterion(pred_force, force)
        if train_test_arr[i]:
            batch_loss.backward()
            optimizer.step()

        # perform L2 lipschitz check and projection
        # lipschitz_projection(ala2_net, strength=lipschitz_strength)
        if verbose:
            if (i + 1) % batch_freq == 0:
                print(
                    "Batch: {: <5} Train: {: <20} Test: {: <20}".format(
                        i + 1, batch_loss, test_loss)
                )

        if train_test_arr[i]:
            train_loss += batch_loss.detach().cpu()
            train_n += 1
        else:
            test_loss += batch_loss.detach().cpu()
            test_n += 1
        coord_counter += traj_lens[i]

    if train_n != 0:
        train_loss /= train_n
    if test_n != 0:
        test_loss /= test_n
    if verbose:
        if epoch % epoch_freq == 0:
            print(
                "Epoch: {: <5} Train: {: <20} Test: {: <20}".format(
                    epoch, train_loss, test_loss))
    epochal_train_losses.append(train_loss)
    epochal_test_losses.append(test_loss)

    if test_loss < best_test_loss or test_loss == 0:
        best_model = ala2_net
        best_test_loss = test_loss
        print('Current Best Model')
        if save_model:
            model_location = data_folder + nn_type + '/model.pt'
            torch.save(ala2_net, model_location.format(directory))

        fig = plt.figure()
        plt.plot(np.arange(0, len(epochal_train_losses), 1),
                 epochal_train_losses, label='Training Loss')
        plt.legend(loc='best')
        plt.xlabel("Epochs")
        plt.xticks(np.arange(1, num_epochs))
        plt.ylabel("Loss")
        plt.savefig(data_folder + nn_type + '//Training Loss1')
        plt.close()

        fig = plt.figure()
        plt.plot(np.arange(0, len(epochal_test_losses), 1),
                 epochal_test_losses, label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epochs")
        plt.xticks(np.arange(1, num_epochs))
        plt.ylabel("Loss")
        plt.savefig(data_folder + nn_type + '//Testing Loss')
        plt.close()
    scheduler.step()

ala2_net = best_model

fig = plt.figure()
plt.plot(np.arange(0, len(epochal_train_losses), 1),
         epochal_train_losses, label='Training Loss')
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.xticks(np.arange(1, num_epochs))
plt.ylabel("Loss")
plt.savefig(nn_type + '//Training Loss')
plt.close()

fig = plt.figure()
plt.plot(np.arange(0, len(epochal_test_losses), 1),
         epochal_test_losses, label='Testing Loss')
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.xticks(np.arange(1, num_epochs))
plt.ylabel("Loss")
plt.savefig(nn_type + '//Testing Loss')
plt.close()

    # for num, batch in enumerate(trainloader):
    #     optimizer.zero_grad()
    #     coord, force, embedding_property = batch
    #     if nn_type == 'Baseline_NN':
    #         print(coord)
    #         print(embedding_property)
    #         print(num)
    #         energy, pred_force = ala2_net.forward(coord, embedding_property=embedding_property)
    #     else:
    #         print(coord.shape)
    #         energy, pred_force = ala2_net.forward(coord, embedding_property=embedding_property)
    #     batch_loss = ala2_net.criterion(pred_force, force)
    #     batch_loss.backward()
    #     optimizer.step()
    #
    #     # perform L2 lipschitz check and projection
    #     lipschitz_projection(ala2_net, strength=lipschitz_strength)
    #     if verbose:
    #         if (num + 1) % batch_freq == 0:
    #             print(
    #                 "Batch: {: <5} Train: {: <20} Test: {: <20}".format(
    #                     num + 1, batch_loss, test_loss)
    #             )
    #     train_loss += batch_loss.detach().cpu()
    #     n += 1

if save_model:
    if nn_type == 'Baseline_NN':
        torch.save(ala2_net, "{}/Baseline_NN/baseline_nn_model.pt".format(directory))
    if nn_type == 'GRU_LinearLayer':
        torch.save(ala2_net, "{}/GRU_LinearLayer/gru_linearlayer_model.pt".format(directory))
    if nn_type == 'Regular_RNN':
        torch.save(ala2_net, "{}/Regular_RNN/regular_rnn_model.pt".format(directory))

fig = plt.figure()
plt.plot(np.arange(0, len(epochal_train_losses), 1),
         epochal_train_losses, label='Training Loss')
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.xticks(np.arange(1, num_epochs))
plt.ylabel("Loss")
plt.savefig(nn_type + '\\' + nn_type + ' Loss')
plt.close()
# plt.show()

n_sims = 1000
n_timesteps = 1000
save_interval = 1
test_coords = np.array([])
test_frames = 0
for i in range(n_files):
    if random.randint(0, 100) > test_split * 100:
        test_frames += traj_lens[i]
        test_coords = np.append(test_coords, coords[i:i+traj_lens[i]])

test_coords = test_coords.reshape((int(len(test_coords) / (3 * 10)), 10, 3))
print(test_coords)
print(test_frames)
initial_coords = np.concatenate([test_coords[0].reshape(-1, n_beads, 3)
                                 for i in np.arange(0, test_frames, test_frames // n_sims)],
                                axis=0)
initial_coords = torch.tensor(initial_coords, requires_grad=True)
print(initial_coords)
sim_embeddings = torch.tensor(embeddings[:initial_coords.shape[0]])
# print(ala2_net)
print("Produced {} initial coordinates.".format(len(initial_coords)))

ala2_net.eval()

print(np.shape(coords))
print(np.shape(sim_embeddings))
sim = Simulation(ala2_net, initial_coords, sim_embeddings, length=n_timesteps,
                 save_interval=save_interval, beta=stats.beta,
                 save_potential=True, device=device,
                 log_interval=save_interval, log_type='print')
traj = sim.simulate()
print(traj.shape)
if protein == 'ALA2':
    names = ['C', 'N', 'CA', 'C', 'N']
    resseq = [1, 2, 2, 2, 3]
    resmap = {1: 'ACE', 2: 'ALA', 3: 'NME'}
    bonds = 'standard'
if protein == 'CHIG':
    names = 10 * ['CA']
    resseq = [1, 1, 2, 3, 4, 5, 6, 7, 8, 1]
    resmap = {1: 'TYR', 2: 'ASP', 3: 'PRO', 4: 'GLU',
              5: 'THR', 6: 'GLY', 7: 'THR', 8: 'TRP'}
    bonds = 'chig'

ala2_cg = CGMolecule(names=names, resseq=resseq, resmap=resmap,
                     bonds=bonds)

ala2_traj = ala2_cg.make_trajectory(coords)
ala2_simulated_traj = ala2_cg.make_trajectory(np.concatenate(traj, axis=0))

ala2_traj.save_hdf5(nn_type + '\\' + nn_type + '_Actual_Traj.h5')
ala2_simulated_traj.save_hdf5(nn_type + '\\' + nn_type + '_Simulated_Traj.h5')


# u = mda.Universe(psf, pdb)
# protein = u.select_atoms('type CA')
# print(protein.residues)
#
# for res in protein.residues:
#     phi = res.phi_selection()
#     if phi is None:
#         names = None
#     else:
#         names = phi.names
#     print('{}: {} '.format(res.resname, names))

# _, phi = md.compute_phi(ala2_traj)
# _, psi = md.compute_psi(ala2_traj)
# # print(phi)
# # print(coords)
# # print(psi)
#
# _, sim_phi = md.compute_phi(ala2_simulated_traj)
# _, sim_psi = md.compute_psi(ala2_simulated_traj)
# # print(sim_phi)
#
# pot, _ = ala2_net.forward(torch.tensor(coords, requires_grad=True),
#                           torch.tensor(embeddings))
# pot = pot.detach().numpy()
# pot = pot - np.min(pot)
#
# sim_pot = np.concatenate(sim.simulated_potential, axis=0)
# sim_pot = sim_pot - np.min(sim_pot)
#
# fig, axes = plt.subplots(figsize=(8, 4))
#
#
# def plot_ramachandran(phi, psi, bins=60, cmap=plt.cm.magma):
#     edges = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
#     counts, _, _ = np.histogram2d(psi.reshape(-1),
#                                   phi.reshape(-1),
#                                   bins=bins,
#                                   range=edges)
#     populations = counts / np.sum(counts)
#
#     # compute energies for only non-zero entries
#     # 1/beta is approximately 0.6 kcal/mol at 300 K
#     energies = -0.6 * np.log(populations,
#                              out=np.zeros_like(populations),
#                              where=(populations > 0))
#
#     # make the lowest energy slightly above zero
#     energies = np.where(energies,
#                         energies - np.min(energies[np.nonzero(energies)]) + 1e-6,
#                         0)
#
#     # mask the zero values from the colormap
#     zvals_masked = np.ma.masked_where(energies == 0, energies)
#
#     img = plt.imshow(zvals_masked, interpolation='nearest', cmap=cmap)
#     plt.gca().invert_yaxis()
#
#     plt.xticks([-0.5, bins / 2, bins],
#                [r'$-\pi$', r'$0$', r'$\pi$'])
#
#     plt.yticks([-0.5, bins / 2, bins],
#                [r'$-\pi$', r'$0$', r'$\pi$'])
#
#     plt.xlabel(r'$\phi$', fontsize=16)
#     plt.ylabel(r'$\psi$', fontsize=16)
#
#     cb = plt.colorbar()
#     cb.ax.set_title(r'$\tilde{F}\left(\frac{kcal}{mol}\right)$')
#
#     return energies
#
#
# plt.subplot(1, 2, 1)
# aa_energies = plot_ramachandran(phi, psi)
# plt.title('Original all-atom trajectory')
#
# plt.subplot(1, 2, 2)
# cg_energies = plot_ramachandran(sim_phi, sim_psi)
# plt.title('Simulated CG trajectory')
#
# aa_energies = aa_energies.ravel()
# cg_energies = cg_energies.ravel()
#
# mod_aa_energies = []
# mod_cg_energies = []
# for i in range(len(aa_energies)):
#     if aa_energies[i] != 0 and cg_energies[i] != 0:
#         mod_aa_energies.append(aa_energies[i])
#         mod_cg_energies.append(cg_energies[i])
# print('Energy KL Divergence:')
# print(sum(rel_entr(mod_aa_energies, mod_cg_energies)))
#
# print('Energy MSE')
# print(np.sum(mean_squared_error(aa_energies, cg_energies)))

# plt.tight_layout()
# plt.show()


def plot_tica(mol_traj, is_aa):
    cg_tica_datas = []
    feat = coor.featurizer(ala2_cg.topology)
    allCA = feat.select('name CA')
    feat.add_distances(feat.pairs(allCA, excluded_neighbors=0))
    feat.describe()

    pyemma.config.show_progress_bars = False
    cgsource = coor.source(mol_traj, features=feat)
    cgdata = cgsource.get_output(stride=5)
    for i in range(len(cgdata)):
        cgdata[i] = cgdata[i] / 10.0
    tica_analysis = coor.tica(cgdata, lag=10, dim=4, kinetic_map=False, commute_map=False)
    CGschnet_TICA_AA = tica_analysis.transform(cgdata)
    cg_tica_datas.append(CGschnet_TICA_AA)
    cgyall_AA = np.concatenate(cg_tica_datas, axis=0)
    cgyall_AA = cgyall_AA.reshape(cgyall_AA.shape[0] * cgyall_AA.shape[1], cgyall_AA.shape[2])
    fig, ax, energies = pyemma.plots.plot_free_energy(cgyall_AA[:, 0], cgyall_AA[:, 1], kT=0.7, vmin=0.0,
                                                      vmax=7,
                                                      cmap=plt.cm.plasma)
    file_name = 'CG '
    if is_aa:
        file_name = 'AA '
    file_name += nn_type + ' TICA'
    plt.savefig(nn_type + '\\' + file_name)
    plt.close()
    print(energies)
    return energies.ravel()


print(np.concatenate(traj, axis=0))
print(np.concatenate(traj, axis=0).shape)
aa_energies = plot_tica(coords, True)
cg_energies = plot_tica(np.concatenate(traj, axis=0), False)

mod_aa_energies = []
mod_cg_energies = []
for i in range(len(aa_energies)):
    if not np.isinf(aa_energies[i]) and not np.isinf(cg_energies[i]):
        mod_aa_energies.append(aa_energies[i])
        mod_cg_energies.append(cg_energies[i])

rmsd = sum(md.rmsd(ala2_simulated_traj, ala2_traj))
kl = sum(rel_entr(mod_aa_energies, mod_cg_energies))
mse = np.sum(mean_squared_error(mod_aa_energies, mod_cg_energies))

results = open(nn_type + '\\' + nn_type + '_Results.txt', 'w+')
results.writelines(['Protein Trajectory RMSD: ' + str(rmsd), '\nEnergy KL Divergence: ' + str(kl), '\nEnergy MSE: ' + str(mse)])
results.close()

print('Protein Trajectory RMSD')
print(rmsd)

print('Energy KL Divergence:')
print(kl)

print('Energy MSE')
print(mse)

# plt.tight_layout()
# plt.show()