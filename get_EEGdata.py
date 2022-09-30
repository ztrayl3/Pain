import numpy as np
import pickle
import mne

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

# begin processing the data!
for subject in P.keys():
    data = P[subject]  # load each subject
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
    ica.plot_overlay(data, exclude=ica.exclude)  # display change made by ICA removal
    ica.apply(data)  # apply ICA to data, removing the artifacts
    del artifact_removal  # clear up memory

    # Re-reference to average
    data.set_eeg_reference(ref_channels="average")

    # Epoch from -1500 to 3000ms. Should be 18 trials per stimulus intensity
    reject_criteria = dict(eeg=200e-6)  # 200 µV
    epochs = mne.Epochs(data, events, event_id=event_dict, tmin=-1.5, tmax=3.0,
                        reject=reject_criteria, preload=True)

    len(epochs["Stimulus/S  1"])  # length should be at least 18 for all 3 of these
    len(epochs["Stimulus/S  2"])
    len(epochs["Stimulus/S  3"])

    ##################
    # ERP components #
    ##################
    erp = epochs.copy()
    erp.filter(l_freq=1.0, h_freq=30.0)  # 1-30Hz filter
    erp.set_eeg_reference(ref_channels=["Fz"])  # re-reference to Fz

    # Get peak amplitude and latency of N1
    good_tmin, good_tmax = 0.15, 0.18
    erp.pick(["Cz"])  # focus on electrode Cz
    low = erp["Stimulus/S  1"].average()
    med = erp["Stimulus/S  2"].average()
    high = erp["Stimulus/S  3"].average()

    ch, lat, amp = low.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax,
                                mode='pos', return_amplitude=True)

    break  # pause after 1 subject

