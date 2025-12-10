"""flower: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    
    return batch


def load_data(partition_id: int, num_partitions: int, attack: int = 0, Iid: str = ""):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if Iid == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = PathologicalPartitioner(num_partitions=num_partitions, partition_by="label", num_classes_per_partition=10)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if attack:
        if partition_id in list(range(attack)):  # malicious clients
            def randomize_labels(example):
                example["label"] = torch.randint(0, 10, ()).item()
                return example

            # Apply permanently to dataset
            partition_train_test["train"] = partition_train_test["train"].map(randomize_labels)

    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=0.001,
                weight_decay=1e-4
            )
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
    """Validate the model on the test set and compute additional metrics."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    all_probs = []

    loss, correct = 0.0, 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)                     # logits
            probs = F.softmax(outputs, dim=1)         # probabilities
            preds = torch.argmax(outputs, dim=1)

            # Collect metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Loss & accuracy
            loss += criterion(outputs, labels).item()
            correct += (preds == labels).sum().item()

    # Aggregate metrics
    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)

    # Convert to NumPy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # F1 (macro)
    f1 = f1_score(y_true, y_pred, average="macro")

    # ROC-AUC (supports multi-class)
    try:
        roc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        roc = None  # e.g., only one class present → ROC undefined

    return avg_loss, accuracy, kappa, f1, roc
