import random
import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np

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

    # Create a histogram
    plt.bar(abnormalities, amount_of_files)

    # Set the x and y axis labels
    plt.xlabel("Abnormalities")
    plt.ylabel("Amount_of_files")
    plt.title(title)
    # Show the plot
    plt.show()


def identify_top_seven_abnormalities(label_dic, histogram_enable=None):
    """
    This function takes the dictionary of all the headers files and retunes a dictionary with only the 7 most common Dx,
    if enabled, this function can also print the histogram of the amount of files per abnormality
    """
    seven_most_common_files = {}
    seven_most_common = sorted([key for key in label_dic.keys()], key=lambda x: len(label_dic[x]), reverse=True)[:7]
    if histogram_enable:
        histogram_by_abnormalities = {key: len(label_dic[key]) for key in label_dic.keys()}
        histogram_by_abnormalities_seven_most_common = {key: len(label_dic[key]) for key in seven_most_common}
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
    seven_most_common_files = identify_top_seven_abnormalities(label_dic)

    # iterate over the abnormalities and split into folders (least common disease first)
    for key in sorted(seven_most_common_files, key = lambda key: len(seven_most_common_files[key])):
        val = seven_most_common_files[key]
        for count, file in enumerate(val):
            if count > max_count:       # limiting the amount of files for each diagnosis
                break
            if not os.path.exists(f"{path}/HR{file}.mat"):
                continue
            if count <= 0.7*len(val):
                os.replace(f"{path}/HR{file}.mat", f"{path}/train/HR{file}.mat")
                os.replace(f"{path}/HR{file}.hea", f"{path}/train/HR{file}.hea")
            elif count <= 0.85*len(val):
                os.replace(f"{path}/HR{file}.mat", f"{path}/validation/HR{file}.mat")
                os.replace(f"{path}/HR{file}.hea", f"{path}/validation/HR{file}.hea")
            else:
                os.replace(f"{path}/HR{file}.mat", f"{path}/test/HR{file}.mat")
                os.replace(f"{path}/HR{file}.hea", f"{path}/test/HR{file}.hea")
    

def zero_padding(matrix, X = 5000):
    """
    Pads a matrix with zeros for size X until it reaches a size of 5000 on 12.

    Args:
    matrix (numpy.ndarray): Input matrix to pad with zeros.
    X (int): Size of padding. Default is 12.

    Returns:
    numpy.ndarray: The padded matrix.
    """
    return np.pad(matrix,((0,X - matrix.shape[0]),(0,0)),mode='constant',constant_values=0)


# def split_data_with_labels(path, boundary_for_amount_of_same_label_files):
#     """
#     This functions takes the dictionary we extracted from the header files and splits the data to sets with labels
#     we can also put boundary for the amount of files with the same label
#     """
#     label_dic = create_dict_from_header(path)
#     seven_most_common_files = identify_top_seven_abnormalities(label_dic)
#     valid_file_dic = {}
#     count = {}
#     for key, val in seven_most_common_files.items():
#         count[key] = 0
#         for file in val:
#             if count[key] >= boundary_for_amount_of_same_label_files:  # putting a boundary for the amount of files
#                 # with the same label (for better training data)
#                 continue
#             elif file in valid_file_dic.keys():
#                 valid_file_dic[file].append(key)
#             else:
#                 valid_file_dic[file] = []
#                 valid_file_dic[file].append(key)
#                 count[key] += 1
    
#     # extracting the data from the files with the wfdb library and putting it as a list with the labels of the files
#     # if the file data isn't 5000 x 12 pad with zeros
#     valid_data_and_labels = [[zero_padding(wfdb.rdsamp(f"{path}//HR{key}")[0]), valid_file_dic[key]] for key in
#                              valid_file_dic.keys()]

#     # we want to randomize the data before we split it to the sets of files, and we want to check we get a good amount
#     # of data with abnormalities and not just data with healthy heart rate labeling
#     random.shuffle(valid_data_and_labels)
#     nsr_count = sum(
#         [1 for x in valid_data_and_labels[:int(0.7 * len(valid_data_and_labels))] if
#          '426783006' in x[1] and (len(x[1]) == 1)])
#     while (nsr_count > (0.4 * count['426783006'])) or (nsr_count < (0.2 * count['426783006'])):
#         random.shuffle(valid_data_and_labels)
#         sum(
#             [1 for x in valid_data_and_labels[:int(0.7 * len(valid_data_and_labels))] if
#              '426783006' in x[1] and (len(x[1]) == 1)])

#     training_data = valid_data_and_labels[:int(0.7 * len(valid_data_and_labels))]
#     test_data = valid_data_and_labels[
#                 int(0.7 * len(valid_data_and_labels)):int(0.85 * len(valid_data_and_labels))]
#     validation_data = valid_data_and_labels[int(0.85 * len(valid_data_and_labels)):]
#     return training_data, test_data, validation_data


# example on how to run this code
# path = "//Users//avrahamhrinevitzky//Desktop//שנה ד //סמסטר א//הולכה חשמלית בתאים//ML_model_project/data"
# training_data, test_data, validation_data = split_data_with_labels(path, 7500)

