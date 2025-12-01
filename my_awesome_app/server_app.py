"""my-awesome-app: A Flower / PyTorch app."""

import json
from typing import List, Tuple

from datasets import load_dataset
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from my_awesome_app.my_strategy import CustomFedAvg
from my_awesome_app.task import Net, get_transforms, get_weights, set_weights, test


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using centralized test set."""
        # Instantiate model and load global parameters
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)

        # Run test function (returns loss, accuracy, kappa, f1, roc)
        loss, accuracy, kappa, f1, roc = test(net, testloader, device)

        # Return metrics dict
        metrics = {
            "cen_accuracy": accuracy,
            "kappa": kappa,
            "f1": f1,
            "roc_auc": roc
        }
        return loss, metrics

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from a fit round by deserializing JSON sent by clients."""
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    return {"max_b": max(b_values)}


def on_fit_config(server_round: int) -> Metrics:
    """Adjust learning rate based on the current round."""
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}


def server_fn(context: Context):
    """Create the components for a ServerApp."""
    # Read from run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load centralized test set
    testset = load_dataset("uoft-cs/cifar10")["test"]
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    # Define Flower strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        proximal_mu=0.1,
        fraction_evaluate=1.0,  # All clients evaluated
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
    )

    

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
