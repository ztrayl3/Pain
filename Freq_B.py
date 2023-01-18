import multiprocessing
from itertools import repeat
import numpy as np
import pickle
import pandas
import mne
mne.set_log_level(verbose="ERROR")  # set all the mne verbose to warning
path = "Processed/"
condition = "Control"  # string, either Perception, EDA, Motor, or Control


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
    start = 0
    stop = start + size
    timeseries = []
    axis = []

    while stop < length:
        kwargs = dict(fmin=14, fmax=30,  # Beta band frequencies
                      tmin=start, tmax=stop,
                      picks=["FCz", "Cz", "CPz", "C1", "C2", "CP1", "CP2", "FC1", "FC2"])
        psds, freqs = epochs.compute_psd(**kwargs).get_data(return_freqs=True)

        # Convert power to dB scale.
        psds = 10 * np.log10(psds)

        # average across epochs, channels, and frequency bands for a single max beta amplitude value
        beta = np.mean(np.average(np.average(psds, axis=0), axis=0))
        timeseries.append(beta)

        # keep track of time index and slide window
        axis.append(np.round(start, 3))  # log a time point at the start of the window
        start = start + step
        stop = start + size

    # "pre-stimulus baseline of -1000 to 0ms" (so 0 to +1 in our case)
    start_index, stop_index = axis.index(0), axis.index(1)
    baseline = np.mean(timeseries[start_index:stop_index])

    # "observed at latencies between about 300 and 1000 ms", so we crop our timseries to this window
    start_index, stop_index = axis.index(1.14), axis.index(2.00)  # 140-1000ms
    cropped = timeseries[start_index:stop_index]

    max_index = timeseries.index(max(cropped))  # take the max point in cropped, find its location in full timeseries
    max_time = int((axis[max_index] - 1.0) * 1e3)  # convert latency to ms and remove the +1s
    amplitude = max(cropped)
    results = dict(gender=gender,
                   subject=sub,
                   stimulus=level,
                   time=max_time,
                   beta=amplitude,
                   baseline=baseline)
    print("Completed Subject: {0}, Stimulus: {1}".format(sub, level))

    queue.put(results)


def main():
    stims = ['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3']
    data = dict(male=None,
                female=None)
    tasks = []
    tasks_that_are_done = multiprocessing.Manager().Queue()

    fill = []
    print("Processing {}".format(condition))

    for gender in data.keys():
        print("Processing {} subjects".format(gender))
        s1 = open(path + "{0}_epochs_{1}.pkl".format(condition, gender), "rb")
        data[gender] = pickle.load(s1)
        s1.close()

        s2 = open(path + "{0}_labels_{1}.pkl".format(condition, gender), "rb")
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
            beta = results["beta"]
            baseline = results["baseline"]

            fill.append([sub, gender, level[-1], "Baseline_Amp", baseline])
            fill.append([sub, gender, level[-1], "Beta_Lat", max_time])
            fill.append([sub, gender, level[-1], "Beta_Amp", beta])

    header = ["ID", "Sex", "Stimulus", "Component", "Value"]
    output = pandas.DataFrame(data=fill, columns=header)
    output.to_csv("Stats/{}_freq_B.csv".format(condition))  # save to csv
    return True


if __name__ == '__main__':
    main()
