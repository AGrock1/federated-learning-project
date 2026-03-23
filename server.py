import flwr as fl
from typing import List, Tuple, Optional, Dict, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, Scalar
from utils import save_global_model
import numpy as np
import argparse
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


RESULTS_DIR = "FL_Project_Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def aggregate_eval_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Weighted average for key metrics across clients."""
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}

    def wavg(key: str) -> float:
        return sum(m[key] * n for n, m in metrics) / total_examples

    return {
        "acc": wavg("acc"),
        "f1": wavg("f1"),
        "recall": wavg("recall"),
        "auc": wavg("auc"),
    }

def save_metrics(metrics: Dict[str, Scalar], round_num: int, model_name: str):
    """Save metrics to a JSON file."""
    path = os.path.join(RESULTS_DIR, f"metrics_{model_name.replace('/', '_')}_round_{round_num}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"
[Server] Metrics saved to {path}")

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_metrics:
            save_metrics(aggregated_metrics, server_round, self.model_name)

        if aggregated_parameters is not None and server_round == 3:
            print("
[Server] Final round completed. Saving global model...")
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            save_path = os.path.join(RESULTS_DIR, f"final_{self.model_name.replace('/', '_')}_model")
            save_global_model(self.model_name, weights, save_path)

        return aggregated_parameters, aggregated_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Name of the model to train")
    args = parser.parse_args()

    strategy = SaveModelStrategy(
        model_name=args.model_name,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,  # <-- This is the key change
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
