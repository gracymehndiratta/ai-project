"""Custom Flower strategy that logs federated round metrics."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class FedAvgWithLogging(fl.server.strategy.FedAvg):
    """
    FedAvg with additional logging of per-round metrics and a callback hook.
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[..., Tuple[float, Dict[str, Scalar]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        log_callback: Optional[Callable[[int, Dict[str, Scalar]], None]] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.log_callback = log_callback
        self.round_history: List[Dict[str, Scalar]] = []
        self.latest_parameters: Optional[Parameters] = initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and log metrics."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

        if aggregated_metrics is None:
            aggregated_metrics = {}

        # Compute average epsilon across clients if available
        epsilons = [res.metrics.get("epsilon", float("inf")) for _, res in results if "epsilon" in res.metrics]
        if epsilons:
            aggregated_metrics["avg_epsilon"] = float(np.mean(epsilons))
            aggregated_metrics["max_epsilon"] = float(np.max(epsilons))

        self.round_history.append({"round": server_round, **aggregated_metrics})
        if self.log_callback:
            self.log_callback(server_round, aggregated_metrics)

        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate (weighted by examples) metrics from multiple clients."""
    if not metrics:
        return {}

    keys = metrics[0][1].keys()
    aggregated: Dict[str, Scalar] = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    for key in keys:
        if isinstance(metrics[0][1][key], (int, float)):
            aggregated[key] = sum(m[key] * num_examples for num_examples, m in metrics) / total_examples

    return aggregated
