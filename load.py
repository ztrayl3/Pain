import pickle
import mne
import os

# load standard 10-20 for use later
montage = mne.channels.make_standard_montage('standard_1020')

# Establish master data path
path = os.path.join("Data", "Pain Dataset")

# Load all Perception condition files
# Format: Exp_Mediation_Paradigm1_Perception_vpxx.vhdr

P = {}
for i in range(1, 52):
    if i < 10:  # pad numbers <10 with one zero
        num = str(0) + str(i)
    else:
        num = str(i)
    fname = os.path.join(path, "Exp_Mediation_Paradigm1_Perception_vp" + num + ".vhdr")
    P[num] = mne.io.read_raw_brainvision(fname)

    # 69 Channels total:
    # -65 scalp electrodes
    # -2 EOG electrodes (LE, RE)
    # -1 GSR electrode (GSR_MR_100_finger)
    # -1 ECG electrode (ECG)
    # Note: 3 unknown channels (Ne, Mat, Ext)
    # Referenced to FCz
    # Grounded at AFz
    # Sampled at 1000Hz
    # Filtered from 0.015Hz to 250Hz  *Note: MNE is reading these wrong, but not too important here

    new_types = []  # create a new channel types array
    for j in P[num].ch_names:
        if j == "LE" or j == "RE":  # mark left and right eye channels
            new_types.append("eog")
        elif "GSR" in j:  # mark GSR channel (won't be used)
            new_types.append("gsr")
        elif j == "ECG":  # mark our ECG channel
            new_types.append("ecg")
        elif j == "NE" or j == "Ma" or j == "Ext":  # mark misc channels, MA ?= mastoid, Ne ?= nasion, Ext ?= events
            new_types.append("misc")
        else:  # mark the rest as EEG channels from extended 10-20
            new_types.append("eeg")
    P[num].set_channel_types(dict(zip(P[num].ch_names, new_types)))  # apply new channel types to raw object
    P[num].set_montage(montage, on_missing="ignore")  # add standard 10-20 montage information for channel locations

data = open("data.pkl", "wb")
pickle.dump(P, data)
data.close()
