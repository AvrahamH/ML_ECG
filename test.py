import torch
import wfdb
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  ## NEED FOR CONFUSION MATRIX
import matplotlib.pyplot as plt
import numpy as np
from preprocess import split_data, load_files
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import argparse
import ML_model_2 as ml

device = torch.device('cpu')

classes = ['NSR', 'MI', 'LAD', 'abQRS', 'LVH', 'TAb', 'MIs']
#
#
def plot_confusion_matrix(test_label_array, predict_vec, num_epochs):
    f, axes = plt.subplots(2, 4, figsize=(25, 15))
    f.delaxes(axes[1, 3])
    axes = axes.ravel()
    for i in range(7):
        disp = ConfusionMatrixDisplay(confusion_matrix(test_label_array[:,i],
                                                        predict_vec[:,i]),
                                                        display_labels=[0, i], 
                                                        cmap=plt.cm.Blues,
                                                        normalize='all')
        disp.plot(ax=axes[i], values_format='.4g')
        disp.ax_.set_title(f'{classes[i]}')
        if i < 10:
            disp.ax_.set_xlabel('')
        if i % 5 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.10, hspace=0.1)
    f.colorbar(disp.im_, ax=axes)
    plt.savefig(f'confusion_matrix_{num_epochs}.png')
    plt.show()


def run_test(path):
    criterion = nn.BCEWithLogitsLoss()
    test_path = f"WFDB/test"
    test_dataset = ml.EcgDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False)
    test_label_array = np.array(test_loader.dataset.labels)
    model = ml.ECGModel()
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    test_out, test_accuracy,predict_vec = ml.evaluate(model, test_loader, criterion, device,mode='test')
    predict_vec = np.array(predict_vec)
    
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_out, test_accuracy))
    predict_vec = np.loadtxt('predict_vec.txt')
    test_label_array = np.loadtxt('test_label_array.txt')
    plot_confusion_matrix(test_label_array, predict_vec, 42)
    # np.savetxt('predict_vec.txt', predict_vec, fmt='%d')
    # np.savetxt('test_label_array.txt', test_label_array, fmt='%d')


if __name__ == '__main__':
    run_test('/Users/dortau/Documents/TAU/סמסטר ז׳/הולכה חשמלית/ecg_model_42_08_03_23-52.pt')
