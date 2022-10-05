import numpy as np
import pickle
import mne
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning

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
sex = {}
sex["male"] = [2, 4, 5, 6, 9, 14, 15, 18, 19, 21, 22, 25, 27, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48, 51]
sex["female"] = [1, 3, 7, 8, 10, 11, 12, 13, 16, 17, 20, 23, 24, 26, 28, 29, 30, 31, 32, 35, 37, 46, 47, 49, 50]

# begin processing the data!
all_epochs = []  # lists for holding epochs and labels
all_labels = []  # in case we need it later
gender = "male"  # what gender are we analyzing?
for subject in P.keys():  # for each subject
    if int(subject) in sex[gender]:  # if this subject is a member of our gender of interest...
        data = P[subject]  # load the subject
        events, event_dict = mne.events_from_annotations(data, event_id=mapping)  # extract their events
        data.load_data()

        #######################################
        # Pre-Processing and Artifact Removal #
        #######################################
        artifact_removal = data.copy()
        artifact_removal.filter(l_freq=1.0, h_freq=None)  # high-pass filter at 1Hz
        artifact_removal.notch_filter(50.0)  # notch filter at 50Hz

        # ICA artifact removal and rejection of +/-100uV
        ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter="auto")
        ica.fit(artifact_removal)  # fit the ICA with EEG and EOG information

        ica.exclude = []
        ecg_indices, ecg_scores = ica.find_bads_ecg(artifact_removal, method='correlation', threshold='auto')  # find ECG
        eog_indices, eog_scores = ica.find_bads_eog(artifact_removal)  # find EOG
        ica.exclude = ica.exclude + ecg_indices + eog_indices
        ica.plot_sources(data, block=True, title=subject)  # last chance to visually inspect ICA
        ica.apply(data)  # apply ICA to data, removing the artifacts

        # Re-reference to average
        data.set_eeg_reference(ref_channels="average")

        # Epoch from -1500 to 3000ms. Should be 18 trials per stimulus intensity
        reject_criteria = dict(eeg=200e-6)  # 200 µV
        epochs = mne.Epochs(data, events, event_id=event_dict, tmin=-1.5, tmax=3.0,
                            reject=reject_criteria, preload=True)

        all_epochs.append(epochs[["Stimulus/S  1", "Stimulus/S  2", "Stimulus/S  3"]])  # record stim epochs to a list
        all_labels.append(subject)  # create identical list of subject IDs, for good measure


del data, artifact_removal, epochs  # try and clear up as much memory as we can...
epochs_combined = mne.concatenate_epochs(all_epochs)  # create a master epoch list of low/med/high stimuli

data = open("epochs_{}.pkl".format(gender), "wb")
pickle.dump(epochs_combined, data)  # save it
data.close()
