from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pickle
import mne
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning

stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
data = dict(male=None,
            female=None)
bands = ["alpha", "beta", "gamma", "high_gamma"]
labels = [gender + " " + level for gender in data.keys() for level in stims]
results = pandas.DataFrame(np.zeros((6, len(bands))), index=labels, columns=bands)  # to store the results cleanly

# Load our epochs, male and female
for gender in data.keys():
    source = open("epochs_{}.pkl".format(gender), "rb")
    data[gender] = pickle.load(source)  # load our epochs
    source.close()

    print("ANALYZING GENDER: {}".format(gender))

    for level in stims:
        epochs = data[gender][level].copy()
        epochs.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz

        # "moving time window with a length of 250 ms and a step size of 20 ms"
        # At 250Hz (sampling frequency), 250ms = 63 samples and 20ms = 5 samples
        # With a step size of 5 samples, the number of points of overlap is 58 samples
        kwargs = dict(fmin=1, fmax=100, n_jobs=-1,
                      tmin=1.150, tmax=1.350,
                      n_fft=63, n_overlap=58, n_per_seg=63,
                      picks=["Cz", "FCz", "C2"])
        psds_welch_mean, freqs_mean = epochs.compute_psd('welch', average='mean', **kwargs).get_data(return_freqs=True)

        # Convert power to dB scale.
        psds_welch_mean = 10 * np.log10(psds_welch_mean)

        alpha_range = np.arange(8.0, 12.5, 0.5)  # 8-12 Hz
        alpha = []
        beta_range = np.arange(12.5, 30.5, 0.5)  # 12-30Hz
        beta = []
        gamma_range = np.arange(30.5, 69.5, 0.5)  # 30-69Hz
        gamma = []
        high_gamma_range = np.arange(70, 90.5, 0.5)  # 70-90Hz
        high_gamma = []
        powers = np.average(psds_welch_mean, axis=0)  # this averages across all channels and all subjects
        for i in range(len(freqs_mean)):  # for each frequency...
            if i in alpha_range:
                alpha.append(powers[i])
            elif i in beta_range:
                beta.append(powers[i])
            elif i in gamma_range:
                gamma.append(powers[i])
            elif i in high_gamma_range:
                high_gamma.append(powers[i])

        label = gender + " " + level
        results["alpha"][label] = np.mean(alpha)
        results["beta"][label] = np.mean(beta)
        results["gamma"][label] = np.mean(gamma)
        results["high_gamma"][label] = np.mean(high_gamma)  # should be 228 +/-41

print(results)
