import torch
import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np
from preprocess import split_data, load_files
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

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
        # the samples need to be permuted because the conv layer is taking the first input as the amount of channels
        return torch.from_numpy(sample).float().permute(1,0), torch.tensor(label).float()

    def load_data(self):
        # Load the ECG signals and labels using the wfdb package
        self.samples, labels = load_files(self.data_dir)
        self.labels = [[0]*7 for i in labels]
        for i, sub_labels in enumerate(labels):
            for label in sub_labels:
                self.labels[i][self.label_map[label]] = 1


device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Define the model architecture
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, padding=2)
        self.maxpool5 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)
        self.sigmoid = nn.Sigmoid()
        self.round = nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.maxpool5(x)
        x = self.flatten(x)
        # THE CODE FALLS HERE WITH THIS ERROR 'mps.matmul' op contracting dimensions differ 159744 & 1024 NEED TO FIX IT
        x = self.fc1(x)
        x = self.dropout1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.round(x)
        return x
    
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    running_loss = 0.0
    train_losses = []
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        running_loss += loss.item()
    return running_loss / len(train_loader), train_losses


def evaluate(model, val_loader, criterion, device):
    val_losses = []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Round the outputs to 0 or 1
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy, val_losses


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

    # Define the loss function and optimizer
    model = ECGModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 200
    # Train the model on the ECG data
    for epoch in range(num_epochs):
        train_out, train_losses = train(model, train_loader, criterion, optimizer, device)
        print('Epoch %d training loss: %.3f' % (epoch + 1, train_out))

        # Evaluate the model on the validation set
        val_out, accuracy, val_losses = evaluate(model, val_loader, criterion, device)
        print('Epoch %d validation accuracy: %.2f%%' % (epoch + 1, accuracy))

    test_dataset = EcgDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    plt.plot(np.arange(num_epochs), train_losses, label='Training loss')
    plt.plot(np.arange(num_epochs), val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

    # Evaluate the model on the test dataset
    test_out, test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
    print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_out, test_accuracy))

    torch.save(model.state_dict(), 'ecg_model.pt')  # save the trained model