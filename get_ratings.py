import numpy as np
import pickle
import mne
condition = "Control"  # either Perception or Control

# Load our database of subjects
source = open("{}_data.pkl".format(condition), "rb")
P = pickle.load(source)
source.close()


# Event Mapping:
# 0-100 = verbal pain ratings
# 101 = start
# 102 = laser
# 103 = stop
# 1000 = low level stimulus
# 2000 = med level stimulus
# 3000 = high level stimulus
# 99999 = Brainvision New Segment

mapping = {"Stimulus/S  5": 101, "Stimulus/S  6": 103, "Laser/L  1": 102, "New Segment/": 99999,
           "Stimulus/S  1": 1000, "Stimulus/S  2": 2000, "Stimulus/S  3": 3000}  # add the known stimulus labels
for i in range(0, 101):  # add the verbal pain ratings in a loop, as they can be anywhere from 0-100
    read = "Comment/" + str(i)
    write = i
    mapping[read] = write  # mimic the same format of {read this: change to this} for event annotations

output = [["Subject", "Stimulus Level", "Verbal Pain Rating"]]  # to be used later for csv output
for subject in P.keys():  # for each subject...
    data = P[subject]  # load their file
    events, event_dict = mne.events_from_annotations(data, event_id=mapping)  # and extract their events
    squished = np.delete(events, 1, 1)  # remove the middle column of the np array (all zeros)

    for event in range(len(squished)):  # iterate through squished by index
        current = squished[event][1]
        if current == 102:  # if we have a laser event
            last = squished[event-1][1]  # grab the previous (stimulus intensity)
            next = squished[event+1][1]  # and the next event (pain rating)

            # rename all stimulus events for an easier CSV file
            if last == 1000:
                last = "low"
            elif last == 2000:
                last = "med"
            elif last == 3000:
                last = "high"

            output.append([subject, last, next])  # append all data to out output array

np.savetxt("Stats/{}_ratings.csv".format(condition), output, delimiter=",", fmt='%s')  # save output array to a csv file
