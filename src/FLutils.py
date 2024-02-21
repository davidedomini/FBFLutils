import torch
import copy
from torch import nn
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F


class CNNMnist(nn.Module):

    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class EmailDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        features = torch.tensor(row.drop('class').values, dtype=torch.float32)
        label = torch.tensor(row['class'], dtype=torch.long)
        return features, label

    def __len__(self):
        return len(self.dataframe)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def update_weights(model, epochs, data_loader, device):
    criterion = nn.NLLLoss().to(device)
    model.train()
    epoch_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for _ in range(epochs):
        batch_loss = []
        for batch_index, (images, labels) in enumerate(data_loader):
            images, label = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        mean_epoch_loss = sum(batch_loss) / len(batch_loss)
        epoch_loss.append(mean_epoch_loss)
    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def evaluate(model, data_loader, device):
    criterion = nn.NLLLoss().to(device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    for batch_index, (images, labels) in enumerate(data_loader):
        images, label = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def load_data_division(path, idx):
    with open(path, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)
    return np.array(data[f'{idx}'])


# split indexes for train, validation, and test (80, 20)
def train_data_loader(dataset, idxs):
    idxs_train = idxs[:int(0.8 * len(idxs))]
    trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                             batch_size=10, shuffle=True)
    return trainloader


def val_data_loader(dataset, idxs):
    idxs_val = idxs[int(0.8 * len(idxs)):]
    validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                             batch_size=int(len(idxs_val) / 10), shuffle=False)
    return validloader


def test_data_loader(dataset, idxs):
    idxs_test = idxs[int(0.9 * len(idxs)):]
    testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                            batch_size=int(len(idxs_test) / 10), shuffle=False)
    return testloader


#def spamemail_train_val_loader(id):
#    df = pd.read_csv(f'data/federated_train/spambase_user{id}.csv')
#    dataset = EmailDataset(dataframe=df)
#    train_set, val_set = random_split(dataset, [450, 50])
#    td = DataLoader(train_set, batch_size=25, shuffle=True)
#    vd = DataLoader(val_set, batch_size=25, shuffle=False)
#    return td, vd



def mnist_cnn_factory():
    model = CNNMnist()
    model.load_state_dict(torch.load('networks/initialmodel_mnist'))
    return model


#def spamemail_mlp_factory():
#    model = MLP(57, 128, 2)
#    model.load_state_dict(torch.load('networks/initialmodel_spambase'))
#    return model
