import pickle
import mne


def print_peak_measures(ch, tmin, tmax, lat, amp):
    print(f'Channel: {ch}')
    print(f'Time Window: {tmin * 1e3:.3f} - {tmax * 1e3:.3f} ms')
    print(f'Peak Latency: {lat * 1e3:.3f} ms')
    print(f'Peak Amplitude: {amp * 1e6:.3f} µV')
    print()


def get_latency_amplitude(good_tmin, good_tmax, dat, ref, positive=False, title=None):
    if positive:  # if we're looking at a positive potential
        mode = 'pos'
    else:  # or a negative one
        mode = 'neg'

    erp = dat.copy()
    erp.pick([ref])  # focus on one electrode
    # low = erp["Stimulus/S  1"].average()
    # med = erp["Stimulus/S  2"].average()
    # high = erp["Stimulus/S  3"].average()
    stim = erp.average()

    _, lat = stim.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode=mode)  # gather negative peak latency

    # Extract mean amplitude in µV over time
    stim.crop(tmin=good_tmin, tmax=good_tmax)
    mean_amp = stim.data.mean(axis=1)

    # Report results
    print(title)
    print_peak_measures(ref, good_tmin, good_tmax, lat, mean_amp[0])


# Load our database of subjects
source = open("data.pkl", "rb")
epochs = pickle.load(source)
source.close()

##################
# ERP components #
##################

epochs.filter(l_freq=1.0, h_freq=30.0, n_jobs=-1)  # 1-30Hz filter
epochs.set_eeg_reference(ref_channels=["Fz"])  # re-reference to Fz

# Get peak amplitude and latency of N1 (164 +/-6ms and -4uV amplitude, ideally) at electrode C4
get_latency_amplitude(0.150, 0.180, epochs, "C4", title="N1")

# Get peak amplitude and latency of N2 (164 +/-6ms and -4uV amplitude, ideally) at electrode CZ
get_latency_amplitude(0.180, 0.210, epochs, "Cz", title="N2")

# Get peak amplitude and latency of P2 (164 +/-6ms and -4uV amplitude, ideally) at electrode Cz
get_latency_amplitude(0.290, 0.320, epochs, "Cz", positive=True, title="P2")
