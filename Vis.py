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
conditions = ["Control", "Perception", "EDA", "Motor"]
stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
data = dict(male=None,
            female=None)
colors = dict(male=dict(low="#2DE1FC",
                        med="#2274A5",
                        high="#090C9B",),
              female=dict(low="#FA7DEB",
                          med="#CE7DA5",
                          high="#563440",))
ERP = False
FREQ = True
Band = "beta"

# ERP Figures
if ERP:
    for gender in data.keys():
        combined = []
        for condition in conditions:
            print(condition, gender)
            s1 = open(path + "{0}_epochs_{1}.pkl".format(condition, gender), "rb")
            data[gender] = pickle.load(s1)
            s1.close()

            epochs = data[gender].copy()
            epochs.shift_time(-1, relative=True)
            epochs.filter(l_freq=1.0, h_freq=30.0, n_jobs=-1)  # 1-30Hz filter
            combined.append(epochs)

        epochs = mne.concatenate_epochs(combined)  # mush all 4 conditions together

        # N1
        pretty_plot(epochs, "N1", ["Fz"], ["C4"], [-6, 6])
        # N2/P2 combined
        pretty_plot(epochs, "N2 - P2", "average", ["Cz"], [-6, 6])

# Band Power Figures
if FREQ:
    for gender in data.keys():
        combined = []
        for condition in conditions:
            print(condition, gender)
            s1 = open(path + "{0}_epochs_{1}.pkl".format(condition, gender), "rb")
            data[gender] = pickle.load(s1)
            s1.close()

            epochs = data[gender].copy().shift_time(-1, relative=True)
            epochs.filter(l_freq=1.0, h_freq=None)  # high-pass filter at 1Hz
            combined.append(epochs)

        all_epochs = mne.concatenate_epochs(combined)
        for stim in stims:
            epochs = all_epochs[stim]  # select one stimulus level
            # "moving time window with a length of 250 ms and a step size of 20 ms"
            size = 0.250
            step = 0.020
            length = epochs.last / 1000  # the latest (in seconds) timepoint
            start = epochs.first / 1000  # the earliest (in seconds) timepoint
            stop = start + size
            timeseries = []
            axis = []

            while stop < length:
                if Band == "gamma":
                    kwargs = dict(fmin=70, fmax=90,
                                  tmin=start, tmax=stop,
                                  picks=["Cz", "FCz", "C2"])
                elif Band == "alpha":
                    kwargs = dict(fmin=8, fmax=13,  # Alpha band frequencies
                                  tmin=start, tmax=stop,
                                  picks=["FCz", "Cz", "CPz", "C1", "C2", "CP1", "CP2", "FC1", "FC2"])
                elif Band == "beta":
                    kwargs = dict(fmin=14, fmax=30,  # Beta band frequencies
                                  tmin=start, tmax=stop,
                                  picks=["FCz", "Cz", "CPz", "C1", "C2", "CP1", "CP2", "FC1", "FC2"])
                psds, freqs = epochs.compute_psd(**kwargs).get_data(return_freqs=True)

                # average across epochs, channels, and frequency bands for a single max amplitude value
                power = np.mean(np.average(np.average(psds, axis=0), axis=0))
                timeseries.append(power)

                # keep track of time index and slide window
                axis.append(np.round(start, 3))  # log a time point at the start of the window
                start = start + step
                stop = start + size

            # "pre-stimulus baseline of -1000 to 0ms" (so 0 to +1 in our case)
            start_index, stop_index = axis.index(-1), axis.index(0)
            baseline = np.mean(timeseries[start_index:stop_index])

            timeseries = ((np.array(timeseries) - baseline) / baseline) * 100  # save timseries as % change from baseline
            if Band == "gamma":
                pd.DataFrame({"Time": axis,  # to a csv file for use in R
                              "Gamma": timeseries}).to_csv("Figures/{0}GammaTS_{1}.csv".format(gender, stim[-1:]))
            elif Band == "alpha":
                pd.DataFrame({"Time": axis,  # to a csv file for use in R
                              "Alpha": timeseries}).to_csv("Figures/{0}AlphaTS_{1}.csv".format(gender, stim[-1:]))
            elif Band == "beta":
                pd.DataFrame({"Time": axis,  # to a csv file for use in R
                              "Beta": timeseries}).to_csv("Figures/{0}BetaTS_{1}.csv".format(gender, stim[-1:]))
