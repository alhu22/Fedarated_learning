"""Fed task module adapted for CIFAR-10."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
import numpy as np

# ----------------------------
# Model Definition
# ----------------------------
class Net(nn.Module):
    """CNN for CIFAR-10."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ----------------------------
# Data loading and transforms
# ----------------------------

def get_transforms():
    """Return standard transforms for CIFAR-10."""
    transforms = Compose([
        Resize((32, 32)),  # just in case
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    def apply_transforms(batch):
        # Convert list of PIL images to tensor of shape [batch_size, 3, 32, 32]
        batch["img"] = torch.stack([transforms(img) for img in batch["img"]])
        return batch

    return apply_transforms


fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR-10 data for federated learning."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        # partitioner = PathologicalPartitioner(num_partitions=num_partitions, partition_by="label", num_classes_per_partition=10)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    if partition_id in [0, 1]:  # malicious clients
        def randomize_labels(example):
            example["label"] = torch.randint(0, 10, ()).item()
            return example

        # Apply permanently to dataset
        partition_train_test["train"] = partition_train_test["train"].map(randomize_labels)
    
    
    partition_train_test = partition_train_test.with_transform(get_transforms())

    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# ----------------------------
# Training & Evaluation
# ----------------------------
def train(net, trainloader, epochs, lr, device):
    """Train the model."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate model and compute metrics."""
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        roc = roc_auc_score(np.eye(len(set(y_true)))[y_true], y_prob, multi_class="ovr")

    except ValueError:
        roc = None

    return avg_loss, accuracy, kappa, f1, roc


# ----------------------------
# Model weights helpers
# ----------------------------
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)