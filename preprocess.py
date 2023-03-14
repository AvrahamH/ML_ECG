import random
import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def iterate_over_files(dir_path):
    """
    A function to iterate over files in a directory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            yield file_path


def create_dict_from_header(path):
    """
    This function takes a path to the data directory and iterates over the files in the directory looking for the
    header files, then from rhe header files it extracts the Dx number and the number of the file and adds the
    Dx as the key and adds the file number as a value for the dictionary, returns this dictionary.
    """
    label_dic = {}
    for file_path in iterate_over_files(path):
        if ".hea" in file_path:
            start_word = "HR"
            end_word = ".hea"
            start_index = file_path.find(start_word) + len(start_word)
            end_index = file_path.find(end_word)
            num_of_file = file_path[start_index:end_index]
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                if '#Dx:' in line:
                    start_word = "#Dx: "
                    end_word = "\n"
                    end_index = file_path.find(end_word)
                    start_index = line.find(start_word) + len(start_word)
                    num_of_Dx = line[start_index:end_index].split(',')
            for num in num_of_Dx:
                if num not in label_dic.keys():
                    label_dic[num] = []
                    label_dic[num].append(num_of_file)
                else:
                    label_dic[num].append(num_of_file)
    return label_dic


def plot_histogram_of_abnormalities(dic, title):
    """
    This function takes a dictionary object and plots the keys and values as a histogram
    """
    # Extract keys and values
    abnormalities = list(dic.keys())
    amount_of_files = list(dic.values())
    plt.figure(figsize=(15, 7))
    # Create a histogram
    plt.bar(abnormalities, amount_of_files)
    plt.xticks(rotation=75, fontsize='xx-small')
    # Set the x and y axis labels
    plt.xlabel("Abnormalities")
    plt.ylabel("Amount of files")
    plt.title(title)
    # Show the plot and save the plot
    plt.savefig(f'{title}.png')
    plt.show()

classes = ['NSR', 'MI', 'LAD', 'abQRS', 'LVH', 'TAb', 'MIs']
abnormalities = ['426783006', '164865005', '39732003', '164951009', '164873001', '164934002','164861001']

def identify_top_seven_abnormalities(label_dic, phase, histogram_enable=None):
    """
    This function takes the dictionary of all the headers files and retunes a dictionary with only the 7 most common Dx,
    if enabled, this function can also print the histogram of the amount of files per abnormality
    """
    seven_most_common_files = {}
    seven_most_common = sorted([key for key in label_dic.keys()], key=lambda x: len(label_dic[x]), reverse=True)[:7]
    classes_by_key = {key: classes[i] for i, key in enumerate(seven_most_common)}
    if histogram_enable:
        histogram_by_abnormalities = {key: len(label_dic[key]) for key in label_dic.keys()}
        histogram_by_abnormalities_seven_most_common = {classes_by_key[key]: len(label_dic[key]) for key in
                                                        seven_most_common}
        plot_histogram_of_abnormalities(histogram_by_abnormalities, "Amount of files per abnormality")
        plot_histogram_of_abnormalities(histogram_by_abnormalities_seven_most_common,
                                        "Amount of files per abnormality - "
                                        "seven most common")
    for key in seven_most_common:
        seven_most_common_files[key] = label_dic[key]
    return seven_most_common_files


def split_data(path, max_count):
    if os.path.exists(f'{path}/train'):
        return
    else:
        os.mkdir(f'{path}/train')
        os.mkdir(f'{path}/validation')
        os.mkdir(f'{path}/test')

    label_dic = create_dict_from_header(path)
    seven_most_common_files = identify_top_seven_abnormalities(label_dic, histogram_enable=True)

    # iterate over the abnormalities and split into folders (least common disease first)
    for key in sorted(seven_most_common_files, key=lambda key: len(seven_most_common_files[key])):
        val = seven_most_common_files[key]
        for count, file in enumerate(val):
            if count > max_count:  # limiting the amount of files for each diagnosis
                break
            if not os.path.exists(f"{path}/HR{file}.mat"):
                continue
            if count <= 0.7 * len(val):
                os.replace(f"{path}/HR{file}.mat", f"{path}/train/HR{file}.mat")
                os.replace(f"{path}/HR{file}.hea", f"{path}/train/HR{file}.hea")
            elif count <= 0.85 * len(val):
                os.replace(f"{path}/HR{file}.mat", f"{path}/validation/HR{file}.mat")
                os.replace(f"{path}/HR{file}.hea", f"{path}/validation/HR{file}.hea")
            else:
                os.replace(f"{path}/HR{file}.mat", f"{path}/test/HR{file}.mat")
                os.replace(f"{path}/HR{file}.hea", f"{path}/test/HR{file}.hea")


def zero_padding(matrix, X=5000):
    """
    Pads a matrix with zeros for size X until it reaches a size of 5000 on 12.
    """
    return np.pad(matrix, ((0, X - matrix.shape[0]), (0, 0)), mode='constant', constant_values=0)


def shift(sig, interval=20):
    offset = np.random.choice(range(-interval, interval))
    sig = np.roll(sig, offset, axis=0)
    return sig


def load_files(path, phase, max_count=7500):
    """
    This functions takes the dictionary we extracted from the header files and splits the data to sets with labels
    we can also put boundary for the amount of files with the same label
    """
    label_dic = create_dict_from_header(path)   # keys=all abnormalities in path, vals=file names
    file_dict = {}                              # keys=only 7 abnormalities, vals=file names
    valid_file_dic = {}                         # keys=file names, vals=Dx for this file (from the 7)
    file_count = {}
    count = {}
    for key in abnormalities:
        file_dict[key] = label_dic[key]
        for file in file_dict[key]:
            file_count[file] = 1 if file not in file_count.keys() else file_count[file] + 1


    # while fine tuning we want to use only signals that aren't NSR
    if phase == 'fine_tune':
        max_count = max([len(file_dict[key]) for key in file_dict.keys()])
        threshold = int(max_count/3)
        # max_count = int(max_count/2)
        ft_files = {}


    for key, val in file_dict.items():
        count[key] = 0

        for file in val:
            if file in valid_file_dic.keys():
                valid_file_dic[file].append(key)
            else:
                valid_file_dic[file] = []
                valid_file_dic[file].append(key)
                count[key] += 1
            if count[key] == max_count:
                break

        # duplicate the data and augment it when fine tuning
        if phase == 'fine_tune':
            val = [i for i in val if file_count[i] == 1]    # duplicate only the files with a single diagnosis
            n_repeats = threshold // len(val) + 1 if len(val) else 1
            ft_files[key] = np.tile(val, n_repeats)[len(file_dict[key]):threshold]

    # extracting the data from the files with the wfdb library and putting it as a list with the labels of the files
    # if the file data isn't 5000 x 12 pad with zeros
    data = [zero_padding(wfdb.rdsamp(f"{path}//HR{key}")[0]) for key in valid_file_dic.keys()]
    labels = [valid_file_dic[key] for key in valid_file_dic.keys()]

    if phase == 'fine_tune':
        for files in ft_files.values():
            for file in files: 
                data.append(zero_padding(shift(wfdb.rdsamp(f"{path}//HR{file}")[0])))
                labels.append(valid_file_dic[file])

    return data, labels


def filt(samples):
    b, a = signal.butter(5,[0.5,100],'bandpass',fs=500)

    for i, sample in enumerate(samples):
        sample = signal.lfilter(b, a, sample)
        # normalize the signal to the range [-1, 1]
        # max_sample = max(np.max(np.abs(sample)), 1)
        # samples[i] = sample / max_sample
        samples[i] = sample

    return samples
