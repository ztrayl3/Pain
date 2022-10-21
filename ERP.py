import numpy as np
import pickle
import pandas
import mne
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning


def get_latency_amplitude(good_tmin, good_tmax, dat, ref, positive=False):
    if positive:  # if we're looking at a positive potential
        mode = 'pos'
    else:  # or a negative one
        mode = 'neg'

    #  NOTE: FOR UNKNOWN REASONS, ERP TIMESTAMPS NEED +1s ADDED ON (ex: 150ms -> 1150ms)
    good_tmax = good_tmax + 1.0
    good_tmin = good_tmin + 1.0

    erp = dat.copy()
    erp.pick([ref])  # focus on one electrode
    stim = erp.average()

    try:
        _, lat = stim.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode=mode)  # gather peak latency
    except ValueError:
        return 0, 0

    # Extract mean amplitude in µV over time
    stim.crop(tmin=good_tmin, tmax=good_tmax)
    mean_amp = stim.data.mean(axis=1)

    lat = int((lat - 1.0) * 1e3)  # convert latency to ms and remove the +1s
    amp = mean_amp[0] * 1e6  # grab our mean amplitude in µV
    return lat, amp


stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
data = dict(male=None,
            female=None)
components = ["N1_Lat", "N1_Amp",
              "N2_Lat", "N2_Amp",
              "P2_Lat", "P2_Amp"]
subjects = [str(sub) for sub in range(1, 52)]
fill = np.zeros((len(stims)*len(components), len(subjects)))
results = pandas.DataFrame(data=fill.T, index=subjects,  # make a 3D dataframe to store results easily
                      columns=pandas.MultiIndex.from_tuples(zip(np.repeat(stims, 6), components*3)))
# NOTE: values = results[("Stimulus/S 1", "N1_Lat")][subject-1] as it is zero based indexing

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
        for level in stims:  # for each stimulus level experienced
            ##################
            # ERP components #
            ##################
            print("STIMULUS: {}".format(level))
            epochs = selected.copy()[level]
            epochs.filter(l_freq=1.0, h_freq=30.0, n_jobs=-1)  # 1-30Hz filter
            epochs.set_eeg_reference(ref_channels=["Fz"])  # re-reference to Fz

            # Get peak amplitude and latency of N1 (164 +/-6ms and -4uV amplitude, ideally) at electrode C4
            latency, amplitude = get_latency_amplitude(0.150, 0.180, epochs, "C4")
            results[(level, "N1_Lat")][sub - 1] = latency
            results[(level, "N1_Amp")][sub - 1] = amplitude

            epochs.set_eeg_reference(ref_channels="average")  # re-reference to average

            # Get peak amplitude and latency of N2 (194 +/-7ms and -4uV amplitude, ideally) at electrode CZ
            latency, amplitude = get_latency_amplitude(0.180, 0.210, epochs, "Cz")
            results[(level, "N2_Lat")][sub-1] = latency
            results[(level, "N2_Amp")][sub-1] = amplitude

            # Get peak amplitude and latency of P2 (306 +/-7ms and -4uV amplitude, ideally) at electrode Cz
            latency, amplitude = get_latency_amplitude(0.290, 0.320, epochs, "Cz", positive=True)
            results[(level, "P2_Lat")][sub-1] = latency
            results[(level, "P2_Amp")][sub-1] = amplitude

results.to_csv("Stats/erp.csv")
