"""flower: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from flower.task import Net, test, load_data
import json
# Create ServerApp
app = ServerApp()

import torch
from flwr.common import Parameters
import numpy as np

def bulyan_aggregate(client_updates, f=1):
    """
    client_updates: list of state_dicts from clients
    f: maximum number of Byzantine clients (malicious clients)
    """
    # Step 1: Flatten each client's state dict into a single vector
    updates_vec = []
    shapes = []
    for state in client_updates:
        vec = torch.cat([p.flatten() for p in state.values()])
        updates_vec.append(vec)
        shapes.append([p.shape for p in state.values()])

    updates_vec = torch.stack(updates_vec)  # shape: [num_clients, num_params]

    n = len(client_updates)
    m = n - 2*f  # number of updates to select (Krum selection)
    
    # Step 2: Compute pairwise distances
    distances = torch.cdist(updates_vec, updates_vec, p=2)

    # Step 3: Compute scores (sum of distances to closest n-f-2 points)
    scores = []
    for i in range(n):
        dists = distances[i]
        dists_sorted, _ = torch.sort(dists)
        score = dists_sorted[:n-f-2].sum()
        scores.append(score.item())

    # Step 4: Select n - 2f updates with lowest scores
    selected_indices = np.argsort(scores)[:m]
    selected_updates = updates_vec[selected_indices]

    # Step 5: Trimmed mean for each parameter
    # Remove f largest and f smallest values along each dimension
    trimmed = []
    selected_updates_np = selected_updates.numpy()
    for i in range(selected_updates_np.shape[1]):
        sorted_vals = np.sort(selected_updates_np[:, i])
        trimmed_vals = sorted_vals[f: -f]  # remove extremes
        trimmed.append(np.mean(trimmed_vals))
    bulyan_vec = torch.tensor(trimmed, dtype=torch.float32)

    # Step 6: Reshape back to state_dict
    bulyan_state = {}
    pointer = 0
    for i, shape in enumerate(shapes[0]):
        numel = np.prod(shape)
        param_vec = bulyan_vec[pointer:pointer+numel]
        bulyan_state[list(client_updates[0].keys())[i]] = param_vec.reshape(shape)
        pointer += numel

    return bulyan_state

class FedAvgBulyan(FedAvg):
    """FedAvg with Bulyan robust aggregation."""

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Extract state_dicts from client results
        client_updates = [res.parameters for res in results]

        # Convert Flower Parameters → PyTorch state_dict
        client_state_dicts = [parameters_to_state_dict(p, model_template=Net()) for p in client_updates]

        MAX_NORM = 7.0

        # --- STEP: Clip update norms ---
        for i, state in enumerate(client_state_dicts):
            # Flatten all params into a vector
            update_vec = torch.cat([p.flatten() for p in state.values()])
            norm = torch.norm(update_vec)
            if norm > MAX_NORM:
                factor = MAX_NORM / norm
                for key in state:
                    state[key] = state[key] * factor
            client_state_dicts[i] = state  # overwrite with clipped state

        # Apply Bulyan aggregation
        bulyan_state = bulyan_aggregate(client_state_dicts, f=2)

        # Convert back to Flower Parameters
        aggregated_parameters = state_dict_to_parameters(bulyan_state)

        return aggregated_parameters, {}


def state_dict_to_parameters(state_dict: dict) -> Parameters:
    """
    Convert PyTorch state_dict to Flower Parameters.

    Args:
        state_dict: PyTorch model's state_dict.

    Returns:
        Flower Parameters object.
    """
    # Flatten each tensor into a NumPy array
    tensors = [param.detach().cpu().numpy() for param in state_dict.values()]
    return Parameters(tensors=tensors)

def parameters_to_state_dict(parameters: Parameters, model_template: torch.nn.Module = None) -> dict:
    """
    Convert Flower Parameters to a PyTorch state_dict.

    Args:
        parameters: Flower Parameters object (list of numpy arrays)
        model_template: Optional PyTorch model used to get parameter names and shapes

    Returns:
        state_dict: PyTorch state_dict compatible with model_template
    """
    arrays = [torch.tensor(t) for t in parameters.tensors]  # Convert NumPy → Torch tensor

    if model_template is None:
        raise ValueError("model_template must be provided to map arrays to state_dict keys")

    # Extract keys and shapes from template model
    keys = list(model_template.state_dict().keys())
    state_dict = {}
    pointer = 0

    for key in keys:
        shape = model_template.state_dict()[key].shape
        numel = torch.prod(torch.tensor(shape)).item()
        state_dict[key] = arrays[pointer].reshape(shape).clone()
        pointer += 1

    return state_dict

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, testloader = load_data(partition_id=0, num_partitions=10)
    

    # Define server-side evaluation function
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> dict:
        """Evaluate global model on centralized test set."""
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy, kappa, f1, roc = test(net, testloader, device)
        
        roc_value = roc if roc is not None else 0.0
        print(f"Round {server_round:2d} | Loss: {loss:.4f} | Acc: {accuracy:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f} | ROC: {roc_value:.4f}")
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "kappa": kappa,
            "f1": f1,
            "roc": roc_value
        }
    # Initialize FedAvg strategy
    strategies = ["fedavg", "fedprox"]
    attacks = [0, 1, 2]
    partitions = ["iid", "non_iid"]

    # for strategy_name in strategies:
    #     for attack in attacks:
    #         for partitioning in partitions:
                
    #             print(f"Running: strategy={strategy_name}, attack={attack}, partition={partitioning}")
                
    #             # Update strategy
    #             if strategy_name == "fedavg":
    #                 strategy = FedAvg(fraction_train=fraction_train)
    #             else:
    #                 strategy = FedProx(fraction_train=fraction_train, proximal_mu=0.01)
                
    #             # Update global model / arrays if needed
    #             global_model = Net()
    #             arrays = ArrayRecord(global_model.state_dict())
                
    #             # You can pass additional config to clients
    #             train_config = ConfigRecord({"lr": lr, "attack": attack, "iid": partitioning == "iid"})
                
    #             # Start federated training
    #             result = strategy.start(
    #                 grid=grid,
    #                 initial_arrays=arrays,
    #                 train_config=train_config,
    #                 num_rounds=num_rounds,
    #                 evaluate_fn=evaluate_fn
    #             )
                
    #             # Save results to unique file
    #             output_file = f"results_{strategy_name}_{partitioning}_attack{attack}.json"
    #             with open(output_file, "w") as f:
    #                 json.dump({k: dict(v) for k, v in result.evaluate_metrics_serverapp.items()}, f, indent=4)
    
    train_config = ConfigRecord({"lr": lr, "attack": 2, "iid": "iid"})

    strategy = FedAvgBulyan(fraction_train=fraction_train)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn
    )
    with open("results/defence2.json", "w") as f:
        json.dump({k: dict(v) for k, v in result.evaluate_metrics_serverapp.items()}, f, indent=4)
    
