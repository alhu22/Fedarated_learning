"""Fed: A Flower / PyTorch client app."""

import json
from random import random

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ConfigRecord, Context

from Fed.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train local model starting from the server-provided parameters."""
        # Apply parameters to local model
        set_weights(self.net, parameters)

        # Train
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
        )

        # Store persistent train loss history
        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            fit_metrics["train_loss_hist"].append(train_loss)

        # Example of sending complex metrics (serialized as JSON)
        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "my_metric": complex_metric_str},
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model using local validation set."""
        # Apply weights from server
        set_weights(self.net, parameters)

        # Run test (returns loss, accuracy, kappa, f1, roc)
        loss, accuracy, kappa, f1, roc = test(self.net, self.valloader, self.device)

        metrics = {
            "accuracy": accuracy,
            "kappa": kappa,
            "f1": f1,
            "roc_auc": roc,
        }

        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    """Return a FlowerClient instance based on node config."""
    net = Net()
    partition_id = context.node_config["partition-id"]
        
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    print("Trainloader length (num batches):", len(trainloader))
       
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
