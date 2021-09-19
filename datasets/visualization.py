import matplotlib.pyplot as plt


def plot_microphone_signals(signals, config, share_axis=True):
    if share_axis:
        plt.plot(signals.T)
    else:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('Microphone signals')
        x = range(signals.shape[1]) # /config["sr"]
        ax1.plot(x, signals[0])
        ax2.plot(x, signals[1])

    print("Delays: {} Sampling Rates: {} Gains: {}".format(
        config["mic_delays"],
        config["mic_sampling_rates"],
        config["mic_gains"]
    ))

    return fig, (ax1, ax2)


# def plot_results(config, output_path):
#     margin = 0.02 # 2% margin at the end of the document
#     n_groups = len(config["x_labels"])
#     group_size = 1/n_groups
#     group_start = np.linspace(0, 1, n_groups + 1)[:-1] + margin
#     group_end = group_start + group_size - margin
#     group_center = (group_start + group_end)/2

#     n_bars_per_group = len(config["bars"])
#     bar_width = (group_size - 2*margin)/n_bars_per_group

#     fig, ax = plt.subplots()

#     bars = config["bars"].items()

#     rects = []

#     ax.bar(0, )
#     for i, bar in enumerate(bars):
#         rect = ax.bar(group_start + i*bar_width, bar[1], bar_width, label=bar[0])
#         rects.append(rect)
#     # Have to come up with a new way of plotting th
#     # rects1 = ax.bar(x - width/2, bars[0][1], width, label=bars[0][0])
#     # rects2 = ax.bar(x + width/2, config[1][1], width, label=bars[1][0])

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel(config["y_label"])
#     ax.set_title(config["title"])
#     ax.set_xticks(group_center)
#     ax.set_xticklabels(config["x_labels"])
#     ax.legend()

#     for rect in rects:
#         ax.bar_label(rect, padding=3)    
    
#     # ax.bar_label(rects1, padding=3)
#     # ax.bar_label(rects2, padding=3)
