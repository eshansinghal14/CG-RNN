import glob
import numpy as np
import mdtraj as md

# import numpy as np
#
# phi = np.arange(1, 10)
# psi = np.arange(1, 18, 2)
#
# print(phi)
# print(phi.reshape(len(phi), 1))


def read_chig(folder, n_files):
    indexes = [8, 29, 50, 64, 76, 91, 105, 112, 126, 150]
    bin_indexes = []
    for i in range(175):
        if i in indexes:
            bin_indexes.append(True)
        else:
            bin_indexes.append(False)
    output = []
    traj_lens = []
    counter = 0
    n_frames = 0
    cg_topo = md.load_psf('Extra/data/cg_chignolin.psf')

    for filename in glob.glob(folder + '/*.npy'):
        file = np.load(filename)
        n_frames += len(file)
        traj_lens.append(len(file))
        cg_coord = file[:, indexes, :]
        cg_traj = md.core.trajectory.Trajectory(cg_coord / 10.0, cg_topo,
                                                unitcell_angles=np.tile(np.array([90., 90., 90.]),
                                                                        (cg_coord.shape[0], 1)),
                                                unitcell_lengths=np.tile(np.array([3.9972, 3.9941003, 3.9935002]),
                                                                         (cg_coord.shape[0], 1)))
#        cg_traj.save_xtc('chignolin_data/npy2traj/' + filename[59:-4] + '.xtc')

        # file_arr = np.append(output, file)
        # file = file.reshape((int(len(file_arr) / (3 * 10)), 10, 3))
        # print(file_arr)
        frame_arr = np.compress(bin_indexes, file, axis=1)
        # print(frame_arr.shape)
        # for i in file:
        #     frame_arr = np.append(frame_arr, np.compress(bin_indexes, i, axis=0))
        # frame_arr = frame_arr.reshape((int(len(frame_arr) / (3 * 10)), 10, 3))
        output = np.append(output, frame_arr)
        # print(frame_arr)
        # print(len(frame_arr))
        print(n_frames)
        print(counter)
        counter += 1
        if n_files != 0:
            if counter == n_files:
                output = output.reshape((int(len(output) / (3 * 10)), 10, 3))
                print(output.shape)
                return output, n_frames, traj_lens
        else:
            if counter % 50 == 0:
                np.save('chig_coords' + str(int(counter / 200)) + '.npy', output)
                output = []

    # np.save('chig_coords' + str(int(counter / 200) + 1) + '.npy', output)