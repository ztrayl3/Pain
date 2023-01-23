import numpy as np
import pickle
import mne
mne.set_log_level(verbose="ERROR")  # set all the mne verbose to warning
path = "Processed/"
condition = "Perception"  # string, either Perception, EDA, Motor, or Control

stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
data = dict(male=None,
            female=None)

fill = []
print("Processing {}".format(condition))

for gender in data.keys():
    print("Processing {} subjects".format(gender))
    s1 = open(path + "{0}_epochs_{1}.pkl".format(condition, gender), "rb")
    data[gender] = pickle.load(s1)
    s1.close()

    s2 = open(path + "{0}_labels_{1}.pkl".format(condition, gender), "rb")
    labels = pickle.load(s2)
    s2.close()

    for sub in set(labels):  # for each subject in labels...
        indexes = [i for i, x in enumerate(labels) if
                   x == sub]  # find the indices in data that contain sub's epochs
        selected = mne.concatenate_epochs([data[gender][i] for i in indexes])  # select epochs for this subject

        for level in stims:
            epochs = selected[level].copy()

            epochs.filter(l_freq=1.0, h_freq=None)  # high-pass filter at 1Hz

            # "moving time window with a length of 250 ms and a step size of 20 ms"
            size = 0.250
            step = 0.020
            length = epochs.last / 1000  # the latest (in seconds) timepoint
            start = 0
            stop = start + size
            timeseries = []
            axis = []

            while stop < length:
                kwargs = dict(fmin=70, fmax=90,  # High-Gamma band frequencies
                              tmin=start, tmax=stop,
                              picks=["Cz", "FCz", "C2"])
                psds, freqs = epochs.compute_psd(**kwargs).get_data(return_freqs=True)

                # average across epochs, channels, and frequency bands for a single max gamma amplitude value
                gamma = np.mean(np.average(np.average(psds, axis=0), axis=0))
                timeseries.append(gamma)

                # keep track of time index and slide window
                axis.append(np.round(start, 3))  # log a time point at the start of the window
                start = start + step
                stop = start + size

            # "pre-stimulus baseline of -1000 to 0ms" (so 0 to +1 in our case)
            start_index, stop_index = axis.index(0), axis.index(1)
            baseline = np.mean(timeseries[start_index:stop_index])

            # "...power estimates in a time window of 150-350 ms...", so we crop our timseries to this window
            start_index, stop_index = axis.index(1.14), axis.index(1.36)
            cropped = timeseries[start_index:stop_index]

            min_index = timeseries.index(min(cropped))  # take the min point in cropped, find its location in full timeseries
            min_time = int((axis[min_index] - 1.0) * 1e3)  # convert latency to ms and remove the +1s
            amplitude = min(cropped)

            raise EOFError