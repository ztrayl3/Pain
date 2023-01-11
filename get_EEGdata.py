import matplotlib
import pickle
import mne
mne.set_log_level(verbose="Warning")  # set all the mne verbose to warning
condition = "Control"  # string, either Perception, EDA, Motor, or Control
gender = "male"  # what gender are we analyzing?

# Load our database of subjects
source = open("{}_data.pkl".format(condition), "rb")
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
sex = dict(male=[2, 4, 5, 6, 9, 14, 15, 18, 19, 21, 22, 25, 27, 33, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45, 48, 51],
           female=[1, 3, 7, 8, 10, 11, 12, 13, 16, 17, 20, 23, 24, 26, 28, 29, 30, 31, 32, 35, 37, 46, 47, 49, 50])

# begin processing the data!
all_epochs = []  # lists for holding epochs and labels
all_labels = []  # in case we need it later
for subject in P.keys():  # for each subject
    if int(subject) in sex[gender]:  # if this subject is a member of our gender of interest...
        print(subject)
        data = P[subject]  # load the subject
        events, event_dict = mne.events_from_annotations(data, event_id=mapping)  # extract their events
        data.load_data()

        # Pre-Processing
        print("Filtering...")
        artifact_removal = data.copy()
        artifact_removal.filter(l_freq=1.0, h_freq=None, n_jobs=-1)  # high-pass filter at 1Hz
        artifact_removal.notch_filter(50.0, n_jobs=-1)  # notch filter at 50Hz

        # ICA artifact removal
        print("Fitting ICA...")
        ica = mne.preprocessing.ICA(n_components=0.95, random_state=97, max_iter="auto")
        ica.fit(artifact_removal)  # fit the ICA with EEG and EOG information

        # Visually inspect the data
        print("Visually inspecting components...")
        for i in range(ica.n_components_):  # look at each component
            ica.plot_properties(data, picks=[i], psd_args={"fmin": 1.0, "fmax": 60.0})
            matplotlib.pyplot.pause(.5)
            included = input("Include component? (y/n)")
            if included == 'n':
                ica.exclude.append(i)
            matplotlib.pyplot.close()
        ica.apply(data)  # apply ICA to data, removing the artifacts

        # Epoch from -1500 to 3000ms. Should be 18 trials per stimulus intensity
        data.set_eeg_reference(ref_channels="average")
        reject_criteria = dict(eeg=200e-6)  # 200 µV
        epochs = mne.Epochs(data, events, event_id=event_dict, tmin=-1.5, tmax=3.0,
                            reject=reject_criteria, preload=True, baseline=None)

        all_epochs.append(epochs[["Stimulus/S  1", "Stimulus/S  2", "Stimulus/S  3"]])  # record stim epochs to a list
        all_labels.append(subject)  # create identical list of subject IDs, for good measure


del data, artifact_removal, epochs  # try and clear up as much memory as we can...
epochs_combined = mne.concatenate_epochs(all_epochs)  # create a master epoch list of low/med/high stimuli

data = open("{0}_epochs_{1}.pkl".format(condition, gender), "wb")
pickle.dump(epochs_combined, data)  # save it
data.close()

l = []
if condition == "Motor" and gender == "male":
    sex[gender].remove(18)  # do not label anyone with subject 18 in the motor condition, since they aren't there
for i in range(len(sex[gender])):  # this should == len(all_epochs)
    l = l + [sex[gender][i]] * len(all_epochs[i])  # add label for all epochs (usually 58-60)
labels = open("{0}_labels_{1}.pkl".format(condition, gender), "wb")
pickle.dump(l, labels)
labels.close()
