import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tdoa.tdoa import get_top_candidate_points, tdoa_sum_error_grid

from datasets.math_utils import compute_distance


def plot_correlations(correlations, output_path=None):
    for key, value in correlations.items():
        plt.plot(value[0], value[1], label=key)
    
    plt.legend()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_location_candidates(room, microphone_distances, output_path=None, grid_step=0.1):
    room_dims = room.dims
    mics = room.microphones.mic_array
    sources = room.sources.source_array

    sns.set_theme()

    error_grid = tdoa_sum_error_grid(room, microphone_distances, grid_step=grid_step)
    
    x_points = np.arange(0, room_dims[0], grid_step)
    y_points = np.arange(0, room_dims[1], grid_step)
    ax = sns.heatmap(error_grid) #, xticklabels=x_points, yticklabels=y_points)


    ax = draw_mics_and_sources(ax, 
                               room_dims,
                               mics,
                               sources,
                               x_max=len(x_points), y_max=len(y_points))

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_top_candidate_points(room, tdoas, output_path=None):
    ax = get_2d_room_plot_axis(room, plot_mics_and_sources=False)
    _plot_top_candidates(room, tdoas, ax)
    
    room_dims = room.dims
    mics = room.microphones.mic_array
    sources = room.sources.source_array
    ax = draw_mics_and_sources(ax, room_dims, mics, sources)

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def _plot_top_candidates(room, microphone_distances, ax):
    # Filter only first and second dimensions
    room_dims = room.dims[0:2]
    mics = [
        mic.loc[0:2] for mic in
        room.microphones.mic_array
    ]

    for mic_ixs, distance in microphone_distances.items():
        mic_1 = mics[mic_ixs[0]]
        mic_2 = mics[mic_ixs[1]]

        candidates = get_top_candidate_points(
            mic_1, mic_2, room_dims, distance
        )
        candidates_x = [candidate[0] for candidate in candidates]
        candidates_y = [candidate[1] for candidate in candidates]

        ax.scatter(candidates_x, candidates_y, label="candidates")
    
    return ax


def get_2d_room_plot_axis(room, plot_mics_and_sources=True,
                          plot_distances=True):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, room.dims[0])
    plt.ylim(0, room.dims[1])

    if plot_mics_and_sources:
        room_dims = room.dims
        mics = room.microphones.mic_array
        sources = room.sources.source_array
        ax = draw_mics_and_sources(ax, room_dims, mics, sources)
    
    if plot_distances:
        _plot_source_to_microphone_distances(room, plt.gca())

    return ax


def plot_mics_and_sources(room_dims, mics, sources):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, room_dims[0])
    plt.ylim(0, room_dims[1])

    draw_mics_and_sources(ax, room_dims, mics, sources)

    return ax


def draw_mics_and_sources(ax, room_dims, mics, sources, x_max=None, y_max=None):
    "Draw microphones and sources in an existing room"

    def normalize(dimension, max_value=None):
        if max_value is None:
            return dimension
        else:
            return max_value/dimension
    
    # mics_x = [mic[0]*normalize(room_dims[0], x_max) for mic in mics]
    # mics_y = [mic[1]*normalize(room_dims[1], y_max) for mic in mics]
    # sources_x = [source[0]*normalize(room_dims[0], x_max) for source in sources]
    # sources_y = [source[1]*normalize(room_dims[1], y_max) for source in sources]
    
    mics_x = [mic[0] for mic in mics]
    mics_y = [mic[1] for mic in mics]
    sources_x = [source[0] for source in sources]
    sources_y = [source[1] for source in sources]
    
    ax.scatter(mics_x, mics_y, marker="^", label="microphones")
    ax.scatter(sources_x, sources_y, marker="o", label="sources")
    ax.legend()
    ax.grid()

    return ax


def _plot_source_to_microphone_distances(room, ax):
    mics = [
        mic.loc[0:2] for mic in
        room.microphones.mic_array
    ]
    sources = [
        source.loc[0:2] for source in
        room.sources.source_array
    ]

    for source in sources:
        for mic in mics:
            distance = compute_distance(source, mic)
            ax.plot(
                [source[0], mic[0]],
                [source[1], mic[1]],
                "--", color="grey",
                label="distance={:.2f}m".format(distance)
            )

    ax.legend()

    return ax
