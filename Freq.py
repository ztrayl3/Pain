from tqdm import tqdm
import numpy as np
import pickle
import pandas
import mne
mne.set_log_level(verbose="ERROR")  # set all the mne verbose to warning

stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
data = dict(male=None,
            female=None)
subjects = [str(sub) for sub in range(1, 52)]
fill = np.zeros((len(stims), len(subjects)))
results = pandas.DataFrame(data=fill.T, index=subjects,  # make a 3D dataframe to store results easily
                           columns=stims)
# NOTE: values = results["Stimulus/S 1"][subject-1] as it is zero based indexing

# Load our epochs, male and female
for gender in tqdm(data.keys()):
    s1 = open("epochs_{}.pkl".format(gender), "rb")
    data[gender] = pickle.load(s1)
    s1.close()

    s2 = open("labels_{}.pkl".format(gender), "rb")
    labels = pickle.load(s2)
    s2.close()

    for sub in tqdm(set(labels)):  # for each subject in labels...
        indexes = [i for i, x in enumerate(labels) if x == sub]  # find the indices in data that contain sub's epochs
        selected = mne.concatenate_epochs([data[gender][i] for i in indexes])  # select epochs for this subject
        for level in tqdm(stims):
            epochs = selected.copy()
            epochs.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz

            # "moving time window with a length of 250 ms and a step size of 20 ms"
            size = 0.250
            step = 0.020
            length = epochs.last/1000  # the latest (in seconds) timepoint
            start = 0
            stop = start + size
            timeseries = []
            axis = []
            while stop < length:
                kwargs = dict(fmin=70, fmax=90, n_jobs=-1,
                              tmin=start, tmax=stop,
                              picks=["Cz", "FCz", "C2"])
                psds, freqs = epochs.compute_psd(**kwargs).get_data(return_freqs=True)

                # Convert power to dB scale.
                psds = 10 * np.log10(psds)

                # average across epochs, channels, and frequency bands for a single value
                gamma = np.mean(np.average(np.average(psds, axis=0), axis=0))
                timeseries.append(gamma)

                # keep track of time index and slide window
                axis.append(np.round(start + size/2, 2))  # log a time point at the 1/2 mark of the window
                start = start + step
                stop = start + size

            max_index = timeseries.index(max(timeseries))
            max_time = axis[max_index] - 1  # subtract 1s because of same timestamp bug from ERP
            results[level][sub - 1] = max_time  # append timeseries to results

results.to_csv("Stats/freq.csv")
