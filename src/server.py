import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,        # train on 100% of connected clients
        fraction_evaluate=1.0,   # test on 100% of connected clients
        min_fit_clients=2,       # wait for at least 2 clients to train
        min_evaluate_clients=2,  # wait for at least 2 clients to test
        min_available_clients=2, # wait for at least 2 clients to start
    )

    print("Server starting... waiting for 2 clients to connect.")
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )