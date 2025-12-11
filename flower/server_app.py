"""flower: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from flower.task import Net, test, load_data              
from flower.defence import FedAvgDefence
import json
# Create ServerApp
app = ServerApp()


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

    strategy = FedAvgDefence(fraction_train=fraction_train)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn
    )
    with open("results/defence2.json", "w") as f:
        json.dump({k: dict(v) for k, v in result.evaluate_metrics_serverapp.items()}, f, indent=4)
    
