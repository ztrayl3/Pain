import pandas as pd
import numpy as np
import pickle
import mne


def pretty_plot(EEG, title, ref, picks, ylim):
    epochs = EEG.copy()
    epochs.set_eeg_reference(ref_channels=ref)  # re-reference to average
    evokeds = dict(low=epochs['Stimulus/S  1'].crop(tmin=-0.2, tmax=1).average(),
                   med=epochs['Stimulus/S  2'].crop(tmin=-0.2, tmax=1).average(),
                   high=epochs['Stimulus/S  3'].crop(tmin=-0.2, tmax=1).average())

    mne.viz.plot_compare_evokeds(evokeds, picks=picks,  # choose only the channels we care about
                                 title=gender.capitalize() + " " + title,  # title the graph
                                 ylim=dict(eeg=ylim),  # scale y axis accordingly
                                 colors=colors[gender],  # use custom color scheme
                                 )[0].savefig("Figures/" + gender + title + ".png",
                                              pad_inches=0, dpi=200)  # and save


mne.set_log_level(verbose="Error")  # set all the mne verbose to warning
path = "Processed/"
condition = "Control"  # string, either Perception, EDA, Motor, or Control
data = dict(male=None,
            female=None)
colors = dict(male=dict(low="#2DE1FC",
                        med="#2274A5",
                        high="#090C9B",),
              female=dict(low="#FA7DEB",
                          med="#CE7DA5",
                          high="#563440",))

# ERP Figures
# Load our epochs, male and female
print("{} ERPs".format(condition))
for gender in data.keys():
    print("LOADING GENDER: {}".format(gender))
    s1 = open(path + "{0}_epochs_{1}.pkl".format(condition, gender), "rb")
    data[gender] = pickle.load(s1)
    s1.close()

    epochs = data[gender].copy()
    epochs.shift_time(-1, relative=True)
    epochs.filter(l_freq=1.0, h_freq=30.0, n_jobs=-1)  # 1-30Hz filter

    # N1
    pretty_plot(epochs, "N1", ["Fz"], ["C4"], [-4, 4])
    # N2/P2 combined
    pretty_plot(epochs, "N2 - P2", "average", ["Cz"], [-6, 6])
    # P2
    # pretty_plot(epochs, "P2", "average", ["Cz"], [-6, 6])

quit()

# Gamma Band Figures
# Load our epochs, male and female
print("{} Gamma".format(condition))
all_epochs = []
for gender in data.keys():
    print("LOADING GENDER: {}".format(gender))
    s1 = open(path + "{0}_epochs_{1}.pkl".format(condition, gender), "rb")
    data[gender] = pickle.load(s1)
    s1.close()

    epochs = data[gender].copy().shift_time(-1, relative=True)

    epochs.filter(l_freq=1.0, h_freq=None)  # high-pass filter at 1Hz

    # "moving time window with a length of 250 ms and a step size of 20 ms"
    size = 0.250
    step = 0.020
    length = epochs.last / 1000  # the latest (in seconds) timepoint
    start = epochs.first / 1000  # the earliest (in seconds) timepoint
    stop = start + size
    timeseries = []
    axis = []

    while stop < length:
        kwargs = dict(fmin=70, fmax=90,
                      tmin=start, tmax=stop,
                      picks=["Cz", "FCz", "C2"])
        psds, freqs = epochs.compute_psd(**kwargs).get_data(return_freqs=True)

        # Convert power to dB scale.
        psds = 10 * np.log10(psds)

        # average across epochs, channels, and frequency bands for a single max gamma amplitude value
        gamma = np.mean(np.average(np.average(psds, axis=0), axis=0))
        timeseries.append(gamma)

        # keep track of time index and slide window
        axis.append(np.round(start, 3))  # log a time point at the start of the window
        start = start + step
        stop = start + size

    # "pre-stimulus baseline of -1000 to 0ms" (so 0 to +1 in our case)
    start_index, stop_index = axis.index(-1), axis.index(0)
    baseline = np.mean(timeseries[start_index:stop_index])

    timeseries = np.array(timeseries) - baseline  # save baseline-corrected gamma power timseries
    pd.DataFrame({"Time": axis,
                  "Gamma": timeseries}).to_csv("Figures/{}GammaTS.csv".format(gender))  # to a csv file for use in R
