import torch
import datetime
import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np
from preprocess import split_data, load_files, filt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import argparse
import logging
import scipy.signal as signal
import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# global parameters used in the run
classes = ['NSR', 'MI', 'LAD', 'abQRS', 'LVH', 'TAb', 'MIs']
now = datetime.datetime.now().strftime('%d_%m_%H-%M')
device = torch.device(
    'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


# Define the ECG dataset
class EcgDataset(Dataset):
    """
    Defines an ECG dataset class that inherits from the PyTorch Dataset class.
    It loads ECG data from a given directory and converts it to PyTorch tensors.
    The class provides methods to access individual samples and labels, and applies a bandpass filter to each sample to remove noise.
    The class also includes a label map to convert ECG diagnostic codes to numerical labels for classification.
    The fine_tune flag can be set to 1 to load only a subset of the data for fine-tuning.
    """

    def __init__(self, data_dir, fine_tune=0):
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        self.label_map = {'426783006': 0, '164865005': 1, '39732003': 2, '164951009': 3, '164873001': 4, '164934002': 5,
                          '164861001': 6}
        self.load_data(fine_tune)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        sample = sample.transpose()
        sample = filt(sample)

        return torch.from_numpy(np.array(sample)).float(), torch.tensor(label).float()

    def load_data(self, fine_tune):
        """
        Load the ECG signals and labels using the wfdb package
        """
        self.samples, labels = load_files(self.data_dir, fine_tune)
        self.labels = [[0] * 7 for i in labels]
        for i, sub_labels in enumerate(labels):
            for label in sub_labels:
                self.labels[i][self.label_map[label]] = 1


# Define the model architecture
class ECGModel(nn.Module):
    """
    Defines the architecture of an ECG classification model using Convolutional Neural Networks (CNNs).
    The model takes a 1D ECG signal with 12 leads as input and outputs a probability distribution over 7 classes.
    """

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Applies the forward pass through the convolutional layers, batch normalization layers,
        and fully connected layers of the neural network, using the input tensor x. Returns the output tensor.
        """
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
        return x


def train(model, train_loader, criterion, optimizer, device):
    """
    Trains the given model using the provided data loader, loss criterion, and optimizer.
    """
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


def evaluate(model, val_loader, criterion, device, test=0, output=''):
    """
    Evaluates the model on a validation set using the given criterion.
    """
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        total_labels = np.empty((0, 7))
        total_predict = np.empty((0, 7))
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            outputs = model.sigmoid(outputs)
            predicted = (outputs > 0.5).float()  # Round the outputs to 0 or 1
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            total_labels = np.vstack((total_labels, labels.cpu().numpy()))
            total_predict = np.vstack((total_predict, predicted.cpu().numpy()))
        accuracy = 100 * correct / total
        # printing and saving the confusion matrix of the run
        if test:
            cm = confusion_matrix(
                total_labels.argmax(axis=1),
                total_predict.argmax(axis=1),
                normalize='true')
            cm = cm.round(decimals=2)
            logger.info("Confusion matrix:")
            tab = '\t'
            logger.info(f"{tab * 20}Predicted labels")
            classes = ['NSR', 'MI', 'LAD', 'abQRS', 'LVH', 'TAb', 'MIs']
            str = "True labels"
            logger.info(
                f"ֿ{str:<21}{classes[0]:<20}{classes[1]:<20}{classes[2]:<20}{classes[3]:<20}{classes[4]:<20}{classes[5]:<20}{classes[6]:<20}")
            for i, label in enumerate(classes):
                logger.info(
                    f"{label:<20}{cm[i][0]:<20}{cm[i][1]:<20}{cm[i][2]:<20}{cm[i][3]:<20}{cm[i][4]:<20}{cm[i][5]:<20}{cm[i][6]:<20}")
            disp = ConfusionMatrixDisplay(confusion_matrix(
                total_labels.argmax(axis=1),
                total_predict.argmax(axis=1),
                normalize='true'),
                display_labels=classes
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.savefig(f'{output}cm_{now}.png')

    return running_loss / len(val_loader), accuracy


def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    start_run_time = datetime.datetime.now().strftime('%d_%m_%H-%M')
    fh = logging.FileHandler(f'{output}/run_log_{start_run_time}.log', 'w')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args():
    """
    Create an ArgumentParser object to handle command line arguments and adds command line arguments that the function expects
    """
    parser = argparse.ArgumentParser(
    description='''Analyze and diagnose an ECG signal using CNN.\nUsage examples:
    train - 'python3 ML_model.py --phase train --epochs 50'
    fine tune on another dataset - 'python3 ML_model.py --phase fine_tune --model-path ecg_model.pt --epochs 20'
    test the model and plot confusion matrix - 'python3 ML_model.py --phase test --model-path ecg_model.pt' ''',formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--input', type=str, default='WFDB', help='Directory for data dir')
    parser.add_argument('-o', '--output', type=str, default='dump', help='Directory for output dir')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train/test/fine_tune')
    parser.add_argument('--epochs', type=int, default=50, help='Training/fine tuning number of epochs')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


if __name__ == "__main__":
    # read the args and make the output folder
    args = parse_args()
    path, output_path, phase, num_of_epochs, model_path = args.input, args.output, args.phase, args.epochs, args.model_path
    fine_tune = args.phase == "fine_tune"
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    output = f'{args.output}/dump_{now}/'
    os.mkdir(output)

    # make the logger and save all the path for the training
    logger = create_logger()

    # defines the model loss function and optimizer
    model = ECGModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    num_epochs = args.epochs
    train_losses = []
    val_losses = []

    # used for training on a remote station
    matplotlib.use('Agg')
    if fine_tune or args.phase == 'test':
        if args.phase == 'test':
            device = torch.device('cpu')
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device, non_blocking=True)
    

    # Train the model on the ECG data
    if not args.phase == "test":
        train_path, val_path, test_path = f"{path}/train", f"{path}/validation", f"{path}/test"
        logger.info(
            f'input - {path},  output - {output_path}, phase - {phase}, num of epochs - {num_of_epochs}, model - {model_path}')
        print("Number of files in each dataset:\ntrain={}, validation={}, test={}" \
              .format(len(os.listdir(train_path)) // 2, len(os.listdir(val_path)) // 2,
                      len(os.listdir(test_path)) // 2))
        logger.info("Number of files in each dataset:\ntrain={}, validation={}, test={}" \
                    .format(len(os.listdir(train_path)) // 2, len(os.listdir(val_path)) // 2,
                            len(os.listdir(test_path)) // 2))
        split_data(path, 7500)

        # Create PyTorch data loaders for the ECG data
        train_dataset = EcgDataset(train_path, fine_tune)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = EcgDataset(val_path)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        try:
            for epoch in range(num_epochs):
                train_out, train_accuracy = train(model, train_loader, criterion, optimizer, device)
                train_losses.append(train_out)

                # Evaluate the model on the validation set
                val_out, accuracy = evaluate(model, val_loader, criterion, device)
                val_losses.append(val_out)
                scheduler.step()

                print('Epoch {}: training loss = {:.3f}, validation loss = {:.3f}, validation accuracy = {:.3f}'.format(
                    epoch + 1, train_out, val_out, accuracy))
                logger.info(
                    'Epoch {}: training loss = {:.3f}, validation loss = {:.3f}, validation accuracy = {:.3f}'.format(
                        epoch + 1, train_out, val_out, accuracy))
        except KeyboardInterrupt:
            print("Training stopped by keyboard interrupt")
            logger.info("Training stopped by keyboard interrupt")
        now = datetime.datetime.now().strftime('%d_%m_%H-%M')
        model_name = f'{output}/ecg_model_{len(train_losses)}_{now}_{args.phase}.pt'
        logger.info(f"Run End \nsaved model name - {model_name} ")
        torch.save(model.state_dict(), model_name)

        # Plot and save the loss curve
        plt.plot(np.arange(len(train_losses)), train_losses, label='Training loss')
        plt.plot(np.arange(len(val_losses)), val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve ({now})')
        plt.savefig(f'{output}loss_plot_{len(train_losses)}_{now}_{args.phase}.png')

    # Evaluate the model on the test dataset
    # logger.info(f'input - {path},  output - {output_path}, phase - {phase}, model - {model_path}')
    test_path = path if path != 'WFDB' else f"{path}/train"
    test_dataset = EcgDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Number of files in test:\nTest files = {}".format((
            len(os.listdir(test_path)) // 2)))
    logger.info("Number of files in test: {}".format((
            len(os.listdir(test_path)) // 2)))

    test_out, test_accuracy = evaluate(model, test_loader, criterion, device, test=1, output=output)
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_out, test_accuracy))
    logger.info('\nTest Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_out, test_accuracy))
    logger.info("End of the run")