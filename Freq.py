from mne.time_frequency import psd_welch
import numpy as np
import pickle

data = dict(male=None,
            female=None)

# Load our epochs, male and female
for gender in data.keys():
    source = open("epochs_{}.pkl".format(gender), "rb")
    data[gender] = pickle.load(source)
    source.close()

for gender in data.keys():
    epochs = data[gender]

    epochs.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz
    epochs.notch_filter(50.0, n_jobs=-1)  # notch filter at 50Hz

    # Estimate PSDs based on average
    kwargs = dict(fmin=1, fmax=100, n_jobs=-1,
                  n_overlap=58, n_per_seg=63)
    # "moving time window with a length of 250 ms and a step size of 20 ms"
    # At 250Hz (sampling frequency), 250ms = 63 samples and 20ms = 5 samples
    # With a step size of 5 samples, the number of points of overlap is 58 samples
    psds_welch_mean, freqs_mean = psd_welch(epochs, average='mean', **kwargs, picks=['eeg'])
    psds_welch_mean, freqs_mean = epochs.compute_psd('welch', average='mean', picks=['eeg'],
                                                     tmin=.150, tmax=.350, **kwargs).get_data(return_freqs=True)

    # Convert power to dB scale.
    psds_welch_mean = 10 * np.log10(psds_welch_mean)

    # We will only plot the PSD for a single sensor in the first epoch.
    ch_name = 'MEG 0122'
    ch_idx = epochs.info['ch_names'].index(ch_name)
    epo_idx = 0

    _, ax = plt.subplots()
    ax.plot(freqs_mean, psds_welch_mean[epo_idx, ch_idx, :], color='k',
            ls='-', label='mean of segments')
    ax.plot(freqs_median, psds_welch_median[epo_idx, ch_idx, :], color='k',
            ls='--', label='median of segments')

    ax.set(title=f'Welch PSD ({ch_name}, Epoch {epo_idx})',
           xlabel='Frequency (Hz)', ylabel='Power Spectral Density (dB)')
    ax.legend(loc='upper right')