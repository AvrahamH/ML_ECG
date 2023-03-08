import torch
import datetime
import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np
from preprocess import split_data, load_files
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import argparse
# import scipy.signal as signal
import matplotlib
# from torch.nn.parallel import DistributedDataParallel
# from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# Define the ECG dataset
class EcgDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        self.label_map = {'426783006': 0, '164865005': 1, '39732003': 2, '164951009': 3, '164873001': 4, '164934002': 5, '164861001': 6}
        self.load_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        sample = sample.transpose()

        # b, a = signal.butter(5,[0.5,40],'bandpass',fs=500)
        # sample = [signal.lfilter(b, a, sample[i]) for i in range(len(sample))]
        # the samples need to be permuted because the conv layer is taking the first input as the amount of channels
        return torch.from_numpy(np.array(sample)).float(), torch.tensor(label).float()

    def load_data(self):
        # Load the ECG signals and labels using the wfdb package
        self.samples, labels = load_files(self.data_dir)
        self.labels = [[0]*7 for i in labels]
        for i, sub_labels in enumerate(labels):
            for label in sub_labels:
                self.labels[i][self.label_map[label]] = 1


device = torch.device('cuda:0,1' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Define the model architecture
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(512)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(1024)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)
        self.conv6 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm1d(2048)
        self.maxpool6 = nn.MaxPool1d(kernel_size=4)
        self.conv7 = nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=5, padding=2)
        self.bn7 = nn.BatchNorm1d(4096)
        self.maxpool7 = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(36864, 1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.bn9 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = self.maxpool6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.functional.relu(x)
        x = self.maxpool7(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn8(x)
        x = self.dropout1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn9(x)
        x = self.dropout2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x_min = torch.min(x)
        x_max = torch.max(x)
        x = (x - x_min) / (x_max - x_min)
        return x

    
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    correct = 0
    running_loss = 0.0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()  # Round the outputs to 0 or 1
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy


def evaluate(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Round the outputs to 0 or 1
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy

classes = ['NSR', 'MI', 'LAD', 'abQRS', 'LVH', 'TAb', 'MIs']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str, default='WFDB', help='Directory for data dir')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


if __name__ == "__main__":
    path = "WFDB"
    split_data(path, 7500)

    train_path = f"{path}/train"
    val_path = f"{path}/validation"
    test_path = f"{path}/test"
    print("Number of files in each dataset:\ntrain={}, validation={}, test={}"\
          .format(len(os.listdir(train_path))//2,len(os.listdir(val_path))//2, len(os.listdir(test_path))//2))

    # Create PyTorch data loaders for the ECG data
    train_dataset = EcgDataset(train_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = EcgDataset(val_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    test_dataset = EcgDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the loss function and optimizer
    model = ECGModel()
    # model = DistributedDataParallel(model)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    num_epochs = 250
    train_losses = []
    val_losses = []
    # Train the model on the ECG data
    try:
        for epoch in range(num_epochs):
            train_out,train_accuracy= train(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_out)
            # print('Epoch %d training loss: %.3f' % (epoch + 1, train_out))
            # print('Epoch %d training accuracy: %.2f%%' % (epoch + 1, train_accuracy))

            # Evaluate the model on the validation set
            val_out, accuracy = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_out)
            scheduler.step()

            print('Epoch {}: training loss = {:.3f}, validation loss = {:.3f}, validation accuracy = {:.3f}'.format(epoch + 1, train_out, val_out, accuracy))
    except KeyboardInterrupt:
        print("Training stopped by keyboard interrupt")

    now = datetime.datetime.now().strftime('%d_%m_%H-%M')
    model_name = f'ecg_model_{epoch}_{now}.pt'
    torch.save(model.state_dict(), model_name)

    # Evaluate the model on the test dataset
    test_out, test_accuracy = evaluate(model, test_loader, criterion, device)
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_out, test_accuracy))

    # Plot and save the loss curve
    matplotlib.use('Agg')
    plt.plot(np.arange(epoch), train_losses, label='Training loss')
    plt.plot(np.arange(epoch), val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curve ({now})')
    plt.savefig(f'loss_plot_{epoch}_{now}.png')
    # plt.show()

    # disp = plot_confusion_matrix(cm, classes=classes, normalize=True,
    #                             title='Normalized confusion matrix')
    # disp.ax_.set_title('Normalized confusion matrix')
    # plt.show()
