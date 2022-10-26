import numpy as np
import pickle
import pandas
import mne
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning

stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
data = dict(male=None,
            female=None)
subjects = [str(sub) for sub in range(1, 52)]
fill = np.zeros((len(stims), len(subjects)))
results = pandas.DataFrame(data=fill.T, index=subjects,  # make a 3D dataframe to store results easily
                           columns=stims)
# NOTE: values = results["Stimulus/S 1"][subject-1] as it is zero based indexing

# Load our epochs, male and female
for gender in data.keys():
    s1 = open("epochs_{}.pkl".format(gender), "rb")
    data[gender] = pickle.load(s1)
    s1.close()

    s2 = open("labels_{}.pkl".format(gender), "rb")
    labels = pickle.load(s2)
    s2.close()

    print("ANALYZING GENDER: {}".format(gender))

    for sub in set(labels):  # for each subject in labels...
        print("SUBJECT: {}".format(sub))
        indexes = [i for i, x in enumerate(labels) if x == sub]  # find the indices in data that contain sub's epochs
        selected = mne.concatenate_epochs([data[gender][i] for i in indexes])  # select epochs for this subject
        for level in stims:
            epochs = data[gender][level].copy()
            epochs.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz

            # "moving time window with a length of 250 ms and a step size of 20 ms"
            # At 250Hz (sampling frequency), 250ms = 63 samples and 20ms = 5 samples
            # With a step size of 5 samples, the number of points of overlap is 58 samples
            kwargs = dict(fmin=70, fmax=90, n_jobs=-1,
                          tmin=1.150, tmax=1.350,
                          n_fft=250, n_per_seg=51,
                          picks=["Cz", "FCz", "C2"])
            psds_welch_mean, freqs_mean = epochs.compute_psd('welch', average='mean', **kwargs).get_data(return_freqs=True)

            # Convert power to dB scale.
            psds_welch_mean = 10 * np.log10(psds_welch_mean)

            high_gamma = np.average(np.average(psds_welch_mean, axis=0), axis=0)  # average across epochs AND channels
            results[level][sub-1] = np.mean(high_gamma)  # should be 228 +/-4

results.to_csv("Stats/freq.csv")
