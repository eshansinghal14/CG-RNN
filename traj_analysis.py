from glob import glob

import numpy as np
import mdtraj as md
from scipy.special import rel_entr
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error

import pyemma
import pyemma.plots
import pyemma.coordinates as coor
import pickle as pkl

from cgnet.cgnet.molecule import CGMolecule
import matplotlib.pyplot as plt
from test import read_chig_coords, read_chig_forces
import h5py

from pyemma.plots.plots2d import get_histogram, _to_free_energy,plot_map

do_aa = True
nn_type = "Baseline_NN"  # Baseline NN
# nn_type = "Regular_RNN"  # Regular RNN
# nn_type = "GRU_LinearLayer"  # GRU+LinearLayer

protein = 'CHIG'
if protein == 'ALA2':
    names = ['C', 'N', 'CA', 'C', 'N']
    resseq = [1, 2, 2, 2, 3]
    resmap = {1: 'ACE', 2: 'ALA', 3: 'NME'}
    bonds = 'standard'
if protein == 'CHIG':
    cg_topo = md.load_psf('Extra/data/cg_chignolin.psf')
    pdb = 'data/aa_chignolin.pdb'
    names = 10 * ['CA']
    resseq = [1, 1, 2, 3, 4, 5, 6, 7, 8, 1]
    resmap = {1: 'TYR', 2: 'ASP', 3: 'PRO', 4: 'GLU',
              5: 'THR', 6: 'GLY', 7: 'THR', 8: 'TRP'}
    bonds = 'chig'

ala2_cg = CGMolecule(names=names, resseq=resseq, resmap=resmap,
                     bonds=bonds)
n_files = 3740  # 3740


def plot_tica(mol_traj, is_aa, n_frames):
    cg_tica_datas = []

    feat = coor.featurizer(ala2_cg.topology)
    print('featurizer constructed.')
    allCA = feat.select('name CA')
    feat.add_distances(feat.pairs(allCA, excluded_neighbors=0))
    len(feat.describe())
    feat.describe()
    print('CA distances identified')

    print("loading data from source...", end=None)
    pyemma.config.show_progress_bars = False
    cgsource = coor.source(mol_traj, features=feat)
    cgdata = cgsource.get_output(stride=5)
    for j in range(len(cgdata)):
        cgdata[j] = cgdata[j] / 10.0

    print('done')
    tica_obs_allatom_select = coor.tica(cgdata, lag=20, dim=4, kinetic_map=False, commute_map=False)
    CGschnet_TICA_AA = tica_obs_allatom_select.transform(cgdata)

    cgyall_AA = np.concatenate(CGschnet_TICA_AA)
    # cgyall_AA = cgyall_AA.reshape(cgyall_AA.shape[0] * cgyall_AA.shape[1], cgyall_AA.shape[2])
    print(cgyall_AA.shape)
    print(cgyall_AA[:, 1])
    fig, ax, energies = pyemma.plots.plot_free_energy(xall=cgyall_AA[:360000, 0], yall=cgyall_AA[:360000, 1], kT=0.7,
                                                      vmin=0.0,
                                                      vmax=5,
                                                      cmap=plt.cm.plasma)
    plt.xlabel('TIC 1', fontsize=16)
    plt.ylabel('TIC 2', fontsize=16)

    file_name = 'CG '
    if is_aa:
        file_name = 'AA '
    file_name += nn_type + ' TICA ' + str(i)
    plt.savefig(nn_type + '\\' + file_name)
    plt.close()
    print(energies)
    return energies.ravel()


sim_traj = h5py.File(nn_type + '\\' + nn_type + '_Simulated_Traj.h5', 'r')
actual_traj = h5py.File(nn_type + '\\' + nn_type + '_Actual_Traj.h5', 'r')
print(actual_traj.keys())

# for i in range(10000, 100000, 10000):
#     actual_coords = actual_traj.get('coordinates')
#     actual_coords = np.array(actual_coords)[:i]
#     print(actual_coords)
#     print(actual_coords.shape)
#
#     sim_coords = sim_traj.get('coordinates')
#     sim_coords = np.array(sim_coords)[:i]
#     # mod_sim = np.array([])
#     # for j in range(0, i*2, 1000):
#     #     mod_sim = np.append(mod_sim, sim_coords[i:i+1000])
#     sim_coords = sim_coords
#     print(sim_coords.shape)
#     print(sim_coords)
#
#     act_traj = md.core.trajectory.Trajectory(actual_coords / 10.0, cg_topo,
#                                              unitcell_angles=np.tile(np.array([90., 90., 90.]),
#                                                                      (actual_coords.shape[0], 1)),
#                                              unitcell_lengths=np.tile(np.array([3.9972, 3.9941003, 3.9935002]),
#                                                                       (actual_coords.shape[0], 1)))
#     act_traj.save_xtc('chignolin_data/npy2traj/Actual.xtc')
#     act_file = 'chignolin_data/npy2traj/Actual.xtc'
#
#     cg_traj = md.core.trajectory.Trajectory(sim_coords, cg_topo,
#                                             unitcell_angles=np.tile(np.array([90., 90., 90.]),
#                                                                     (sim_coords.shape[0], 1)),
#                                             unitcell_lengths=np.tile(np.array([3.9972, 3.9941003, 3.9935002]),
#                                                                      (sim_coords.shape[0], 1)))
#     print(cg_traj)
#     cg_traj.save_xtc('chignolin_data/npy2traj/' + nn_type + '.xtc')
#     sim_file = 'chignolin_data/npy2traj/' + nn_type + '.xtc'
#
#     cg_energies = plot_tica(sim_file, False, i)
#     aa_energies = plot_tica(act_file, True, i)

if do_aa:
    # aa_traj_coords_names = glob('coords_nowater/*.npy')
    # ca_indices = [8, 29, 50, 64, 76, 91, 105, 112, 126, 150]
    # for traj in aa_traj_coords_names:
    #     coord = np.load(traj)
    #     cg_coord = coord[:, ca_indices, :]
    #     cg_traj = md.core.trajectory.Trajectory(cg_coord / 10.0, cg_topo,
    #                                             unitcell_angles=np.tile(np.array([90., 90., 90.]),
    #                                                                     (cg_coord.shape[0], 1)),
    #                                             unitcell_lengths=np.tile(np.array([3.9972, 3.9941003, 3.9935002]),
    #                                                                      (cg_coord.shape[0], 1)))
    #     cg_traj.save_xtc('chignolin_data/npy2traj/' + traj[30:-4] + '.xtc')
    actual_coords = actual_traj.get('coordinates')
    actual_coords = np.array(actual_coords)
    act_traj = md.core.trajectory.Trajectory(actual_coords / 10.0, cg_topo,
                                             unitcell_angles=np.tile(np.array([90., 90., 90.]),
                                                                     (actual_coords.shape[0], 1)),
                                             unitcell_lengths=np.tile(np.array([3.9972, 3.9941003, 3.9935002]),
                                                                      (actual_coords.shape[0], 1)))
    act_traj.save_xtc('chignolin_data/Actual.xtc')
    traj_fns = 'chignolin_data/Actual.xtc'

    print('files globbed.')
    feat = coor.featurizer(ala2_cg.topology)
    print('featurizer constructed.')
    allCA = feat.select('name CA')
    feat.add_distances(feat.pairs(allCA, excluded_neighbors=0))
    len(feat.describe())
    feat.describe()
    print('CA distances identified')

    print("loading data from source...", end=None)
    pyemma.config.show_progress_bars = False
    cgsource = coor.source(traj_fns, features=feat)
    cgdata = cgsource.get_output(stride=5)
    # for i in range(len(cgdata)):
    #     cgdata[i] = cgdata[i]/10.0

    print('done')
    tica_obs_allatom_select = coor.tica(cgdata, lag=20, dim=4, kinetic_map=False, commute_map=False)
    cgTIC1_AA = tica_obs_allatom_select.transform(cgdata)

    cgyall_AA = np.concatenate(cgTIC1_AA)
    print(cgyall_AA.shape)

    figX, axX, aa_energies = pyemma.plots.plot_free_energy(cgyall_AA[:, 0], cgyall_AA[:, 1], kT=0.7, vmin=0.0, vmax=7,
                                                           cmap=plt.cm.plasma)
    aa_energies = aa_energies.ravel()
    plt.xlabel('TIC 1', fontsize=16)
    plt.ylabel('TIC 2', fontsize=16)
    # plt.xlim(-4, 4)
    # plt.ylim(-3, 7)
    plt.savefig('AA TICA.png')
    plt.close()

    pkl.dump(tica_obs_allatom_select, open("all_atomic_tica_select.pkl", "wb"))

sim_coords = sim_traj.get('coordinates')
sim_coords = np.array(sim_coords)[:500]

cg_traj = md.core.trajectory.Trajectory(sim_coords / 10.0, cg_topo,
                                        unitcell_angles=np.tile(np.array([90., 90., 90.]), (sim_coords.shape[0], 1)),
                                        unitcell_lengths=np.tile(np.array([3.9972, 3.9941003, 3.9935002]),
                                                                 (sim_coords.shape[0], 1)))
# cg_traj.save_xtc('chignolin_data/' + nn_type + '.xtc')
cg_traj.save_xtc('cg_chig.xtc')
tica_obs_allatom_select = pkl.load(open('all_atomic_tica_select.pkl', 'rb'))
# nn_data_dir = 'chignolin_data/' + nn_type + '/'
# traj_fns_nn = sorted(glob(nn_data_dir + "/*.xtc"))
traj_fns_nn = 'chignolin_data/' + nn_type + '.xtc'

print('files globbed.')
feat = coor.featurizer(ala2_cg.topology)
print('featurizer constructed.')
allCA = feat.select('name CA')
feat.add_distances(feat.pairs(allCA, excluded_neighbors=0))
len(feat.describe())
feat.describe()
print('CA distances identified')

print("loading data from source...", end=None)
pyemma.config.show_progress_bars = False
cgsource_nn = coor.source(traj_fns_nn, features=feat)
cgdata_nn = cgsource_nn.get_output(stride=25)

print('done')
cgTIC1_AA_nn = tica_obs_allatom_select.transform(cgdata_nn)

cgyall_AA_nn = np.concatenate(cgTIC1_AA_nn)
print(cgyall_AA_nn.shape)


print(cgyall_AA_nn[:100, 0])
print(cgyall_AA_nn.shape)
cgyall_AA_nn = cgyall_AA_nn[(cgyall_AA_nn[:, 0] > -2.5), :]
cgyall_AA_nn = cgyall_AA_nn[(cgyall_AA_nn[:, 0] < 2), :]
cgyall_AA_nn = cgyall_AA_nn[(cgyall_AA_nn[:, 1] > -3.2), :]
cgyall_AA_nn = cgyall_AA_nn[(cgyall_AA_nn[:, 1] < 4), :]

print(cgyall_AA_nn.shape)

x_aa, y_aa, z_aa = get_histogram(cgyall_AA_nn[:, 0], cgyall_AA_nn[:, 1])
f_aa = _to_free_energy(z_aa, minener_zero=True) * 0.7
fig, ax, misc = plot_map(x_aa, y_aa, f_aa, cmap=plt.cm.plasma, vmin=0.0, vmax=7)

print(x_aa)
print(y_aa)
print(f_aa)

figX, axX, cg_energies = pyemma.plots.plot_free_energy(cgyall_AA_nn[:, 0], cgyall_AA_nn[:, 1], kT=0.7, vmin=0.0, vmax=7,
                                                       cmap=plt.cm.plasma)
print(cg_energies.shape)
cg_energies = cg_energies.ravel()
plt.xlabel('TIC 1', fontsize=16)
plt.ylabel('TIC 2', fontsize=16)
# plt.xlim(-4, 4)
# plt.ylim(-3, 7)
plt.savefig(nn_type + ' TICA.png')
plt.close()

energies = open('Actual_Results.txt', 'w+')
energies.writelines(str(np.array(aa_energies)))
energies.close()
energies = open(nn_type + '_Results.txt', 'w+')
energies.writelines(str(np.array(cg_energies)))
energies.close()

mod_aa_energies = []
mod_cg_energies = []
mod_cg_energies = cg_energies
mod_aa_energies = aa_energies
mod_cg_energies[mod_cg_energies == float('inf')] = 0
mod_aa_energies[mod_aa_energies == float('inf')] = 0

# for i in range(len(cg_energies)):
#     # print(i)
#     if not aa_energies[i] == float('inf') and not cg_energies[i] == float('inf'):
#         mod_aa_energies.append(aa_energies[i])
#         mod_cg_energies.append(cg_energies[i])
#     else:
#         mod_aa_energies.append(0)
#         mod_cg_energies.append(0)

act_traj = md.load(traj_fns, top=cg_topo)
act_traj.save_hdf5('temp1_hdf.h5')
act_traj = h5py.File('temp1_hdf.h5')
ala2_traj = ala2_cg.make_trajectory(np.array(act_traj.get('coordinates')))

sim_traj = md.load(traj_fns_nn, top=cg_topo)
sim_traj.save_hdf5('temp2_hdf.h5')
sim_traj = h5py.File('temp2_hdf.h5')
ala2_simulated_traj = ala2_cg.make_trajectory(np.array(sim_traj.get('coordinates')))

# rmsd = []
# sim_traj = act_traj
# for i in range(0, 1001):
#     mini_traj = ala2_cg.make_trajectory(np.array(sim_traj.get('coordinates'))[i * 1000:i * 1000 + 1001])
#     ref_traj = ala2_cg.make_trajectory(np.array(sim_traj.get('coordinates'))[i * 1000:i * 1000 + 1])
#     rmsd.append(np.mean(md.rmsd(mini_traj, ref_traj)))
#     print(i)
# rmsd = np.mean(rmsd)
# rmsd = md.rmsd(ala2_simulated_traj, ala2_traj)
# rdf = md.compute_rdf(ala2_simulated_traj)
# kl = rel_entr(mod_aa_energies, mod_cg_energies)
# kl = kl[kl != float('inf')]
# kl = np.mean(kl)
js = distance.jensenshannon(mod_aa_energies, mod_cg_energies)
# js = js[js != float('inf')]
# js = np.mean(js)
mse = np.sum(mean_squared_error(mod_aa_energies, mod_cg_energies))

# plt.plot(rmsd)
# plt.savefig(nn_type + ' RMSD.png')
# plt.close()

results = open(nn_type + '\\' + nn_type + '_Results.txt', 'w+')
results.writelines(['\nEnergy JS Divergence: ' + str(js), '\nEnergy MSE: ' + str(mse)])
results.close()
#
# print('RMSD: ')
# print(rmsd)

# print('Energy KL Divergence:')
# print(kl)

print('Energy JS Divergence:')
print(js)

print('Energy MSE')
print(mse)

# plt.tight_layout()
# plt.show()
