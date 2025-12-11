
import torch
from flwr.common import Parameters
import numpy as np
from flwr.serverapp.strategy import FedAvg

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

class FedAvgDefence(FedAvg):
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