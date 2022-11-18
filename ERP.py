import numpy as np
import pickle
import pandas
import mne
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning


def get_latency_amplitude(good_tmin, good_tmax, dat, ref=None, mode="abs"):
    #  NOTE: FOR UNKNOWN REASONS, ERP TIMESTAMPS NEED +1s ADDED ON (ex: 150ms -> 1150ms)
    good_tmax = good_tmax + 1.0
    good_tmin = good_tmin + 1.0

    erp = dat.copy()
    if ref:  # if a reference electrode is provided
        erp.pick([ref])  # focus on the one electrode
    stim = erp.average()

    try:
        _, lat = stim.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode=mode)  # gather peak latency
    except ValueError:
        return "NA", "NA"

    # Extract mean amplitude in µV over time
    stim.crop(tmin=good_tmin, tmax=good_tmax)
    mean_amp = stim.data.mean(axis=1)

    lat = int((lat - 1.0) * 1e3)  # convert latency to ms and remove the +1s
    amp = mean_amp[0] * 1e6  # grab our mean amplitude in µV
    return lat, amp

condition = "Perception"  # string, either Perception, EDA, Motor, or Control
data = dict(male=None,
            female=None)
stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
components = ["N1_Lat", "N1_Amp",
              "N2_Lat", "N2_Amp",
              "P2_Lat", "P2_Amp"]
sex = ["female", "male", "female", "male", "male", "male", "female", "female", "male", "female", "female", "female",
       "female", "male", "male", "female", "female", "male", "male", "female", "male", "male", "female", "female",
       "male", "female", "male", "female", "female", "female", "female", "female", "male", "male", "female", "male",
       "female", "male", "male", "male", "male", "male", "male", "male", "male", "female", "female", "male",
       "female", "female", "male"]
header = ["ID", "Sex", "Stimulus", "Component", "Value"]
subjects = [str(sub) for sub in range(1, 52)]
fill = []

# Load our epochs, male and female
print("ANALYZING CONDITION: {}".format(condition))
for gender in data.keys():
    s1 = open("{0}_epochs_{1}.pkl".format(condition, gender), "rb")
    data[gender] = pickle.load(s1)
    s1.close()

    s2 = open("{0}_labels_{1}.pkl".format(condition, gender), "rb")
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
            latency, amplitude = get_latency_amplitude(0.150, 0.180, epochs, "C4", mode="neg")
            fill.append([sub, sex[sub - 1], level[-1], "N1_Lat", latency])
            fill.append([sub, sex[sub - 1], level[-1], "N1_Amp", amplitude])

            epochs.set_eeg_reference(ref_channels="average")  # re-reference to average

            # Get peak amplitude and latency of a baseline period
            _, amplitude = get_latency_amplitude(-1, 0, epochs)
            fill.append([sub, sex[sub - 1], level[-1], "Baseline_Amp", amplitude])

            # Get peak amplitude and latency of N2 (194 +/-7ms and -4uV amplitude, ideally) at electrode CZ
            latency, amplitude = get_latency_amplitude(0.180, 0.210, epochs, "Cz", mode="neg")
            fill.append([sub, sex[sub - 1], level[-1], "N2_Lat", latency])
            fill.append([sub, sex[sub - 1], level[-1], "N2_Amp", amplitude])

            # Get peak amplitude and latency of P2 (306 +/-7ms and -4uV amplitude, ideally) at electrode Cz
            latency, amplitude = get_latency_amplitude(0.290, 0.320, epochs, "Cz", mode="pos")
            fill.append([sub, sex[sub - 1], level[-1], "P2_Lat", latency])
            fill.append([sub, sex[sub - 1], level[-1], "P2_Amp", amplitude])

results = pandas.DataFrame(data=fill, columns=header)
results.to_csv("Stats/{}_erp.csv".format(condition))
