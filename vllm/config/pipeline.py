# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from bisect import bisect_left
from itertools import accumulate
from typing import Self

from pydantic import Field, field_validator, model_validator

from vllm.config.utils import config


def _normalize(values: list[float]) -> list[float]:
    scale = max(values)
    return [value / scale for value in values]


def _weighted_layer_counts(num_layers: int, weights: list[float]) -> list[int]:
    """Apportion layers proportionally, with at least one per stage."""
    counts = [0] * len(weights)
    pending = list(range(len(weights)))
    remaining = num_layers
    while pending:
        scaled = _normalize([weights[index] for index in pending])
        total = math.fsum(scaled)
        quotas = [remaining * (weight / total) for weight in scaled]
        small = [index for index, quota in zip(pending, quotas) if quota < 1]
        if small:
            for index in small:
                counts[index] = 1
            remaining -= len(small)
            pending = [index for index in pending if counts[index] == 0]
            continue
        for index, quota in zip(pending, quotas):
            counts[index] = math.floor(quota)
        remainder_order = sorted(
            zip(pending, quotas),
            key=lambda item: (-(item[1] - math.floor(item[1])), item[0]),
        )
        for index, _ in remainder_order[: remaining - sum(counts[i] for i in pending)]:
            counts[index] += 1
        break
    return counts


def _cost_layer_counts(costs: list[float], weights: list[float]) -> list[int]:
    """Choose contiguous boundaries nearest to cumulative cost targets."""
    cumulative_cost = list(accumulate(_normalize(costs), initial=0.0))
    cumulative_weight = list(accumulate(_normalize(weights), initial=0.0))
    boundaries = [0]
    for stage in range(len(weights) - 1):
        target = (
            cumulative_cost[-1] * cumulative_weight[stage + 1] / cumulative_weight[-1]
        )
        lower = boundaries[-1] + 1
        upper = len(costs) - (len(weights) - stage - 1)
        candidate = bisect_left(cumulative_cost, target, lower, upper + 1)
        candidates = {max(lower, candidate - 1), min(upper, candidate)}
        boundaries.append(
            min(
                candidates,
                key=lambda index: (abs(cumulative_cost[index] - target), index),
            )
        )
    boundaries.append(len(costs))
    return [end - start for start, end in zip(boundaries, boundaries[1:])]


@config
class PipelinePartitionConfig:
    """Plan contiguous transformer-layer ranges in pipeline rank order.

    Larger compute weights receive proportionally more work. Without layer
    costs, constrained largest-remainder apportionment assigns layer counts,
    breaking equal remainders in favor of lower ranks. With layer costs,
    boundaries approximate cumulative weighted cost targets; this heuristic
    does not promise the optimal maximum stage latency.

    Setting either endpoint count fixes both endpoints (the unspecified one
    defaults to zero). Only explicit endpoints may have zero layers; every
    other stage must own at least one transformer layer.
    """

    head_layers: int | None = Field(default=None, ge=0)
    """Fixed transformer-layer count on the first stage, including zero."""
    tail_layers: int | None = Field(default=None, ge=0)
    """Fixed transformer-layer count on the last stage, including zero."""
    stage_compute_weights: list[float] | None = None
    """Positive relative compute capacities, one per pipeline stage.

    Weights of fixed endpoint stages do not affect the remaining allocation.
    """
    layer_costs: list[float] | None = None
    """Positive relative cost of each transformer layer in model order."""
    layer_type_costs: dict[str, float] | None = None
    """Profiled relative costs keyed by the model's HF layer_types values.

    Expanded in model layer order. Mutually exclusive with layer_costs.
    """
    stage_node_ips: list[str] | None = None
    """Optional placement addresses, one per stage; addresses may repeat."""

    @field_validator("stage_compute_weights", "layer_costs")
    @classmethod
    def _validate_positive_values(
        cls, values: list[float] | None
    ) -> list[float] | None:
        if values is not None and (
            not values
            or any(not math.isfinite(value) or value <= 0 for value in values)
        ):
            raise ValueError(
                "Weights and layer costs must be nonempty, finite and positive"
            )
        return values

    @field_validator("layer_type_costs")
    @classmethod
    def _validate_type_costs(
        cls, values: dict[str, float] | None
    ) -> dict[str, float] | None:
        if values is not None and (
            not values
            or any(
                not name.strip() or not math.isfinite(cost) or cost <= 0
                for name, cost in values.items()
            )
        ):
            raise ValueError("Layer type costs must be nonempty, finite and positive")
        return values

    @model_validator(mode="after")
    def _validate_cost_source(self) -> Self:
        self._check_cost_source()
        return self

    def _check_cost_source(self) -> None:
        if self.layer_costs is not None and self.layer_type_costs is not None:
            raise ValueError("Use layer_costs or layer_type_costs, not both")

    @field_validator("stage_node_ips")
    @classmethod
    def _validate_node_ips(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and (
            not values or any(not value.strip() for value in values)
        ):
            raise ValueError("stage_node_ips must contain nonempty addresses")
        return values

    def resolve(
        self,
        num_hidden_layers: int,
        pp_size: int,
        *,
        layer_types: list[str] | None = None,
    ) -> list[int]:
        """Return per-stage counts covering all layers exactly once.

        Fixed endpoints require at least three stages. Insufficient layers for
        nonempty unfixed stages are rejected instead of creating empty stages.
        Node addresses constrain placement only, not layer allocation.
        """
        if (
            isinstance(num_hidden_layers, bool)
            or not isinstance(num_hidden_layers, int)
            or num_hidden_layers < 1
        ):
            raise ValueError("num_hidden_layers must be a positive integer")
        if isinstance(pp_size, bool) or not isinstance(pp_size, int) or pp_size < 1:
            raise ValueError("pp_size must be a positive integer")
        for name, values in (
            ("stage_compute_weights", self.stage_compute_weights),
            ("stage_node_ips", self.stage_node_ips),
        ):
            if values is not None and len(values) != pp_size:
                raise ValueError(f"{name} must have {pp_size} entries")
        if self.layer_costs is not None and len(self.layer_costs) != num_hidden_layers:
            raise ValueError(f"layer_costs must have {num_hidden_layers} entries")
        self._check_cost_source()
        costs = self.layer_costs
        if self.layer_type_costs is not None:
            if layer_types is None or len(layer_types) != num_hidden_layers:
                raise ValueError(
                    "layer_type_costs requires model layer_types with one entry "
                    "per transformer layer"
                )
            missing = set(layer_types) - self.layer_type_costs.keys()
            unused = self.layer_type_costs.keys() - set(layer_types)
            if missing or unused:
                raise ValueError(
                    "layer_type_costs must match model layer_types: "
                    f"missing={sorted(missing)}, unused={sorted(unused)}"
                )
            costs = [self.layer_type_costs[layer_type] for layer_type in layer_types]

        weights = self.stage_compute_weights or [1.0] * pp_size
        fixed_endpoints = self.head_layers is not None or self.tail_layers is not None
        head, tail = self.head_layers or 0, self.tail_layers or 0
        if fixed_endpoints:
            if pp_size < 3:
                raise ValueError("Fixed head/tail layers require pp_size >= 3")
            if head + tail >= num_hidden_layers:
                raise ValueError(
                    "head_layers + tail_layers must be less than num_hidden_layers"
                )
            weights = weights[1:-1]
        remaining = num_hidden_layers - head - tail
        if remaining < len(weights):
            raise ValueError("Each unfixed pipeline stage must have at least one layer")
        if costs is None:
            counts = _weighted_layer_counts(remaining, weights)
        else:
            counts = _cost_layer_counts(costs[head : num_hidden_layers - tail], weights)
        return [head, *counts, tail] if fixed_endpoints else counts


@config
class PipelineTransportConfig:
    """Bound asynchronous pipeline transfers and their retained buffers."""

    enabled: bool = False
    """Enable the bounded asynchronous pipeline transport."""
    max_inflight_sends: int = Field(default=2, ge=1)
    """Maximum pending sends before waiting for a completion."""
    max_inflight_bytes: int = Field(default=268435456, ge=1)
    """Maximum retained send-buffer bytes (256 MiB by default)."""
    async_receive: bool = True
    """Post asynchronous receives when the transport is enabled."""
