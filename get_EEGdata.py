import numpy as np
import pickle
import mne


def print_peak_measures(ch, tmin, tmax, lat, amp):
    print(f'Channel: {ch}')
    print(f'Time Window: {tmin * 1e3:.3f} - {tmax * 1e3:.3f} ms')
    print(f'Peak Latency: {lat * 1e3:.3f} ms')
    print(f'Peak Amplitude: {amp * 1e6:.3f} µV')


# Load our database of subjects
source = open("data.pkl", "rb")
P = pickle.load(source)
source.close()

# denote all events and their IDs
mapping = {"Stimulus/S  5": 101, "Stimulus/S  6": 103, "Laser/L  1": 102, "New Segment/": 99999,
           "Stimulus/S  1": 1000, "Stimulus/S  2": 2000, "Stimulus/S  3": 3000}  # add the known stimulus labels
for i in range(0, 101):  # add the verbal pain ratings in a loop, as they can be anywhere from 0-100
    read = "Comment/" + str(i)
    write = i
    mapping[read] = write  # mimic the same format of {read this: change to this} for event annotations

# mark male vs female subjects
male = [2, 4, 5, 6, 9, 14, 15, 18, 19, 21, 22, 25, 27, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48, 51]
female = [1, 3, 7, 8, 10, 11, 12, 13, 16, 17, 20, 23, 24, 26, 28, 29, 30, 31, 32, 35, 37, 46, 47, 49, 50]

# begin processing the data!
all_epochs = []  # lists for holding epochs and labels
all_labels = []  # in case we need it later
gender = male  # what gender are we analyzing?
for subject in P.keys():  # for each subject
    if int(subject) in gender:  # if this subject is a member of our gender of interest...
        data = P[subject]  # load the subject
        events, event_dict = mne.events_from_annotations(data, event_id=mapping)  # extract their events
        data.load_data()

        #######################################
        # Pre-Processing and Artifact Removal #
        #######################################
        artifact_removal = data.copy()
        artifact_removal.filter(l_freq=1.0, h_freq=None)  # high-pass filter at 1Hz
        artifact_removal.notch_filter(50.0)  # notch filter at 50Hz

        # Identify EOG and ECG artifacts
        artifact_picks = mne.pick_types(artifact_removal.info, eog=True, ecg=True)
        # artifact_removal.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
        eog_evoked = mne.preprocessing.create_eog_epochs(artifact_removal)  # create EOG epochs
        ecg_evoked = mne.preprocessing.create_ecg_epochs(artifact_removal)  # create ECG epochs

        # ICA artifact removal and rejection of +/-100uV
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
        ica.fit(artifact_removal)  # fit the ICA with EEG and EOG information

        ica.exclude = []
        ecg_indices, ecg_scores = ica.find_bads_ecg(artifact_removal, method='correlation', threshold='auto')  # find ECG
        eog_indices, eog_scores = ica.find_bads_eog(artifact_removal)  # find EOG
        ica.exclude = ica.exclude + ecg_indices + eog_indices
        ica.plot_sources(data, block=True)  # last chance to visually inspect ICA
        ica.apply(data)  # apply ICA to data, removing the artifacts

        # Re-reference to average
        data.set_eeg_reference(ref_channels="average")

        # Epoch from -1500 to 3000ms. Should be 18 trials per stimulus intensity
        reject_criteria = dict(eeg=200e-6)  # 200 µV
        epochs = mne.Epochs(data, events, event_id=event_dict, tmin=-1.5, tmax=3.0,
                            reject=reject_criteria, preload=True)

        all_epochs.append(epochs)  # record all epochs to a list
        all_labels.append(subject)  # create identical list of subject IDs, for good measure


##################
# ERP components #
##################
epochs_combined = mne.concatenate_epochs(all_epochs)
epochs_combined.filter(l_freq=1.0, h_freq=30.0)  # 1-30Hz filter
epochs_combined.set_eeg_reference(ref_channels=["Fz"])  # re-reference to Fz

# Get peak amplitude and latency of N1 (164 +/-6ms and -4uV amplitude, ideally)
N1 = epochs_combined.copy()
good_tmin, good_tmax = 0.15, 0.18
N1.pick(["C4"])  # focus on electrode C4
low = N1["Stimulus/S  1"].average()
med = N1["Stimulus/S  2"].average()
high = N1["Stimulus/S  3"].average()
stim = N1[["Stimulus/S  1", "Stimulus/S  2", "Stimulus/S  3"]].average()

_, lat = stim.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax, mode='neg')  # gather negative peak latency

# Extract mean amplitude in µV over time
stim.crop(tmin=good_tmin, tmax=good_tmax)
mean_amp = stim.data.mean(axis=1)

# Report results
print_peak_measures("C4", good_tmin, good_tmax, lat, mean_amp[0])

