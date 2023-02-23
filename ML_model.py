import torch
from ML_model_preprocess import make_test_training_validation_data_sets_with_labels
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



# Define the model architecture
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5)
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
        x = self.fc1(x)
        x = self.dropout1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.round(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "//Users//avrahamhrinevitzky//Desktop//שנה ד //סמסטר א//הולכה חשמלית בתאים//ML_model_project/data"
training_data, test_data, validation_data = make_test_training_validation_data_sets_with_labels(
    path, 7500)
x_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]
x_val = [x[0] for x in validation_data]
y_val = [x[1] for x in validation_data]
x_test = [x[0] for x in test_data]
y_test = [x[1] for x in test_data]
# Convert the ECG data and labels to PyTorch tensors
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_val = torch.Tensor(x_val)
y_val = torch.Tensor(y_val)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

# Create PyTorch data loaders for the ECG data
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the loss function and optimizer
model = ECGModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
# Train the model on the ECG data
for epoch in range(200):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch %d training loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    # Evaluate the model on the validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()  # Round the outputs to 0 or 1
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Epoch %d validation accuracy: %.2f%%' % (epoch + 1, accuracy))