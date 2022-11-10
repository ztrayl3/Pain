import multiprocessing
from itertools import repeat
import numpy as np
import pickle
import pandas
import mne
mne.set_log_level(verbose="ERROR")  # set all the mne verbose to warning


def work(eeg, queue):
    epochs = eeg["data"]  # grab all the values from our task
    gender = eeg["gender"]
    level = eeg["stimulus"]
    sub = eeg["subject"]
    epochs.filter(l_freq=1.0, h_freq=None)  # high-pass filter at 1Hz

    # "moving time window with a length of 250 ms and a step size of 20 ms"
    size = 0.250
    step = 0.020
    length = epochs.last / 1000  # the latest (in seconds) timepoint
    start = 1  # start from 1s not 0s because of +1s timestamp bug
    stop = start + size
    timeseries = []
    axis = []
    while stop < length:
        kwargs = dict(fmin=70, fmax=90,
                      tmin=start, tmax=stop,
                      picks=["Cz", "FCz", "C2"])
        psds, freqs = epochs.compute_psd(**kwargs).get_data(return_freqs=True)

        # Convert power to dB scale.
        psds = 10 * np.log10(psds)

        # average across epochs, channels, and frequency bands for a single max gamma amplitude value
        gamma = np.mean(np.average(np.average(psds, axis=0), axis=0))
        timeseries.append(gamma)

        # keep track of time index and slide window
        axis.append(np.round(start, 2))  # log a time point at the start of the window
        start = start + step
        stop = start + size

    max_index = timeseries.index(max(timeseries))
    max_time = int((axis[max_index] - 1.0) * 1e3)  # convert latency to ms and remove the +1s
    results = dict(gender=gender,
                   subject=sub,
                   stimulus=level,
                   time=max_time,
                   gamma=max(timeseries))
    print("Completed Subject: {0}, Stimulus: {1}".format(sub, level))

    queue.put(results)


def main():
    stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
    data = dict(male=None,
                female=None)
    tasks = []
    tasks_that_are_done = multiprocessing.Manager().Queue()

    fill = []

    for gender in data.keys():
        print("Processing {} subjects".format(gender))
        s1 = open("epochs_{}.pkl".format(gender), "rb")
        data[gender] = pickle.load(s1)
        s1.close()

        s2 = open("labels_{}.pkl".format(gender), "rb")
        labels = pickle.load(s2)
        s2.close()

        for sub in set(labels):  # for each subject in labels...
            indexes = [i for i, x in enumerate(labels) if
                       x == sub]  # find the indices in data that contain sub's epochs
            selected = mne.concatenate_epochs([data[gender][i] for i in indexes])  # select epochs for this subject
            for level in stims:
                epochs = selected[level]
                t = dict(gender=gender,
                         subject=sub,
                         stimulus=level,
                         data=epochs)
                tasks.append(t)  # append each task that contains all necessary information for analysis

        number_of_processes = 16

        # creating processes
        print("Creating {} processes".format(number_of_processes))
        p = multiprocessing.Pool(number_of_processes)
        p.starmap(work, zip(tasks, repeat(tasks_that_are_done)))
        p.close()
        p.join()  # block until all processes are complete

        # Gather results from queue
        print("Gathering results from processes")
        while not tasks_that_are_done.empty():
            results = tasks_that_are_done.get()
            max_time = results["time"]  # grab all the values from our finished task
            level = results["stimulus"]
            sub = results["subject"]
            gender = results["gender"]
            gamma = results["gamma"]

            fill.append([sub, gender, level, "Gamma_Lat", max_time])
            fill.append([sub, gender, level, "Gamma_Amp", gamma])

    header = ["ID", "Sex", "Stimulus", "Component", "Value"]
    output = pandas.DataFrame(data=fill, columns=header)
    output.to_csv("Stats/freq.csv")  # save to csv
    return True


if __name__ == '__main__':
    main()
