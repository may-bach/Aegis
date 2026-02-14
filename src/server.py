import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    lr = 0.01 * (0.9 ** (server_round - 1)) 
    return {"lr": lr}

def server_fn(context: fl.common.Context):
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
    )

    # B. Define Config
    config = ServerConfig(num_rounds=10)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)