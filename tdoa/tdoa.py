import numpy as np

from tdoa.metrics import compute_error
from tdoa.correlation import compute_correlations


def compute_tdoas(simulation_results, simulation_fs, mode="gcc-phat"):
    correlations = compute_correlations(
        simulation_results, simulation_fs, mode=mode)
    tdoas = {
        key: value[0]
        for key, value in correlations.items()
    }

    return tdoas


def get_top_candidate_points(
        mic_1, mic_2, room_dims, tdoa, norm="l2", step=0.1, n_candidates=200):
    candidates = []

    for x in np.arange(0, room_dims[0], step):
        for y in np.arange(0, room_dims[1], step):
            error = compute_error((x, y), mic_1, mic_2, tdoa, norm=norm)
            candidates.append((x, y, error))

    candidates = np.array(candidates)
    sorted_candidates = candidates[candidates[:, 2].argsort()]
    top_candidates = sorted_candidates[0:n_candidates]

    return top_candidates


def tdoa_sum_error_grid(room, microphone_distances, norm="l2", grid_step=0.1):
    room_dims = room.dims[0:2]
    mics = [
        mic.loc[0:2] for mic in
        room.microphones.mic_array
    ]

    grids = []
    for mic_ixs, distance in microphone_distances.items():
        mic_1 = mics[mic_ixs[0]]
        mic_2 = mics[mic_ixs[1]]

        grid = _tdoa_error_grid(mic_1, mic_2, room_dims, distance, 
                                                    norm=norm, grid_step=grid_step)
        grids.append(grid)
    
    return np.sum(grids, 0)


def _tdoa_error_grid(mic_1, mic_2, room_dims, target_tdoa, 
                                        norm="l2", grid_step=0.1):

    num_x_points = int(room_dims[0]/grid_step)
    num_y_points = int(room_dims[1]/grid_step)

    x_points = np.arange(0, room_dims[0], grid_step)
    y_points = np.arange(0, room_dims[1], grid_step)

    grid = np.zeros((num_x_points, num_y_points))

    for ix, x in enumerate(x_points):
        for iy, y in enumerate(y_points):
            error = compute_error((x, y), mic_1, mic_2, target_tdoa, norm=norm)
            grid[ix, iy] = error

    return grid

