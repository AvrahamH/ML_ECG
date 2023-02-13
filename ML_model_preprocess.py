import random
import os


def iterate_files(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            yield file_path


def make_test_training_validation_file_sets_with_labels(dic):
    valid_file_dic = {}
    count = {}
    for key, val in dic.items():
        count[key] = 0
        for file in val:
            if file in valid_file_dic.keys() or count[key] >= 10000:  # putting a boundary for the amount of  files
                # with the same label
                continue
            else:
                valid_file_dic[file] = key
                count[key] += 1

    valid_file_names_and_labels = [("HR" + key + ".mat", valid_file_dic[key]) for key in valid_file_dic.keys()]
    random.shuffle(valid_file_names_and_labels)
    training_files = valid_file_names_and_labels[:int(0.7 * len(valid_file_names_and_labels))]
    test_files = valid_file_names_and_labels[
                 int(0.7 * len(valid_file_names_and_labels)):int(0.85 * len(valid_file_names_and_labels))]
    validation_files = valid_file_names_and_labels[int(0.85 * len(valid_file_names_and_labels)):]
    return training_files, test_files, validation_files


def distinguish_seven_most_common_abnormalities(path):
    label_dic = {}
    seven_most_common_files = {}
    for file_path in iterate_files(path):
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
    seven_most_common = sorted([key for key in label_dic.keys()], key=lambda x: len(label_dic[x]), reverse=True)[:7]
    for key in seven_most_common:
        seven_most_common_files[key] = label_dic[key]
    training_files, test_files, validation_files = make_test_training_validation_file_sets_with_labels(
        seven_most_common_files)

    return seven_most_common_files, training_files, test_files, validation_files


seven_most_common_files, training_files, test_files, validation_files = distinguish_seven_most_common_abnormalities(
    "//Users//avrahamhrinevitzky//Desktop//שנה ד //סמסטר א//הולכה חשמלית בתאים//ML_model_project/data")
