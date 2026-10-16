# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import accumulate

import pytest

from vllm.config.pipeline import PipelinePartitionConfig, PipelineTransportConfig

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("layers", "weights", "expected"),
    [
        (40, [1, 8, 1], [4, 32, 4]),
        (32, [1, 6, 1], [4, 24, 4]),
        (10, [1, 1, 1], [4, 3, 3]),
        (8, [1, 3], [2, 6]),
        (3, [100, 1, 1], [1, 1, 1]),
        (5, [1, 100, 1], [1, 3, 1]),
        (8, [1e308, 1e308], [4, 4]),
        (8, [5e-324, 1e308], [1, 7]),
        (10, [1e308, 1e-308, 5e-324], [8, 1, 1]),
        (8, [5e-324, 5e-324], [4, 4]),
    ],
)
def test_weighted_counts_respect_capacity_and_nonempty_stages(
    layers, weights, expected
):
    partition = PipelinePartitionConfig(stage_compute_weights=weights)

    assert partition.resolve(layers, len(weights)) == expected


@pytest.mark.parametrize(
    ("kwargs", "layers", "stages", "expected"),
    [
        ({}, 7, 1, [7]),
        ({}, 10, 3, [4, 3, 3]),
        ({"head_layers": 2, "tail_layers": 3}, 13, 4, [2, 4, 4, 3]),
        ({"head_layers": 0, "tail_layers": 0}, 12, 3, [0, 12, 0]),
        ({"head_layers": 0, "tail_layers": 0}, 1, 3, [0, 1, 0]),
        ({"head_layers": 2}, 12, 3, [2, 10, 0]),
        ({"tail_layers": 2}, 12, 3, [0, 10, 2]),
        (
            {
                "head_layers": 1,
                "tail_layers": 1,
                "stage_compute_weights": [100, 1, 3, 100],
            },
            10,
            4,
            [1, 2, 6, 1],
        ),
    ],
)
def test_endpoint_constraints_and_default_allocation(kwargs, layers, stages, expected):
    assert PipelinePartitionConfig(**kwargs).resolve(layers, stages) == expected


@pytest.mark.parametrize(
    ("costs", "weights", "expected"),
    [
        ([8, 1, 1, 1, 1, 1, 1, 1], [1, 1], [1, 7]),
        ([1] * 12, [1, 2, 1], [3, 6, 3]),
        ([1] * 5, [1, 1], [2, 3]),
        ([100, 1, 1], [1, 1, 1], [1, 1, 1]),
        ([1e308] * 4, [1e308, 1e308], [2, 2]),
        ([5e-324, 1e308, 5e-324], [1, 1], [1, 2]),
        ([5e-324, 1e308, 5e-324], [5e-324, 1e308], [1, 2]),
        ([1e308, 5e-324, 5e-324], [1e308, 5e-324], [1, 2]),
        ([5e-324] * 3, [5e-324] * 3, [1, 1, 1]),
    ],
)
def test_layer_costs_choose_contiguous_nonempty_ranges(costs, weights, expected):
    partition = PipelinePartitionConfig(
        layer_costs=costs, stage_compute_weights=weights
    )
    counts = partition.resolve(len(costs), len(weights))

    assert counts == expected
    assert partition.resolve(len(costs), len(weights)) == counts
    boundaries = list(accumulate(counts, initial=0))
    covered = [
        layer
        for start, end in zip(boundaries, boundaries[1:])
        for layer in range(start, end)
    ]
    assert covered == list(range(len(costs)))


def test_layer_cost_planning_excludes_fixed_endpoint_costs():
    partition = PipelinePartitionConfig(
        head_layers=2,
        tail_layers=1,
        layer_costs=[100, 100, 8, 1, 1, 1, 1, 1, 1, 1, 100],
    )

    assert partition.resolve(11, 4) == [2, 1, 7, 1]


def test_repeated_placement_addresses_do_not_change_plan_or_input():
    addresses = ["192.0.2.1", "192.0.2.2", "192.0.2.1"]
    partition = PipelinePartitionConfig(
        head_layers=0, tail_layers=0, stage_node_ips=addresses
    )

    assert partition.resolve(12, 3) == partition.resolve(12, 3) == [0, 12, 0]
    assert partition.stage_node_ips == addresses


@pytest.mark.parametrize("field", ["stage_compute_weights", "layer_costs"])
@pytest.mark.parametrize(
    "values", [[], [0], [-1], [float("inf")], [float("-inf")], [float("nan")]]
)
def test_nonpositive_or_nonfinite_costs_and_weights_are_rejected(field, values):
    with pytest.raises(ValueError, match="finite and positive"):
        PipelinePartitionConfig(**{field: values})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"head_layers": -1},
        {"tail_layers": -1},
        {"stage_node_ips": []},
        {"stage_node_ips": [""]},
        {"stage_node_ips": ["  "]},
    ],
)
def test_invalid_endpoint_counts_and_empty_addresses_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PipelinePartitionConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "layers", "stages", "message"),
    [
        ({"head_layers": 0}, 8, 2, "pp_size >= 3"),
        ({"tail_layers": 0}, 8, 1, "pp_size >= 3"),
        ({"head_layers": 4, "tail_layers": 4}, 8, 3, "must be less"),
        ({"head_layers": 9}, 8, 3, "must be less"),
        ({"head_layers": 1, "tail_layers": 1}, 3, 4, "at least one layer"),
        ({}, 2, 3, "at least one layer"),
        ({"stage_compute_weights": [1, 2]}, 8, 3, "stage_compute_weights must have 3"),
        ({"stage_node_ips": ["192.0.2.1"]}, 8, 3, "stage_node_ips must have 3"),
        ({"layer_costs": [1, 2]}, 8, 3, "layer_costs must have 8"),
        ({}, 0, 3, "num_hidden_layers must be a positive integer"),
        ({}, 8, 0, "pp_size must be a positive integer"),
        ({}, 8, 1.5, "pp_size must be a positive integer"),
        ({}, True, 1, "num_hidden_layers must be a positive integer"),
    ],
)
def test_invalid_topology_is_rejected_before_planning(kwargs, layers, stages, message):
    with pytest.raises(ValueError, match=message):
        PipelinePartitionConfig(**kwargs).resolve(layers, stages)


@pytest.mark.parametrize("field", ["max_inflight_sends", "max_inflight_bytes"])
@pytest.mark.parametrize("value", [0, -1])
def test_transport_requires_positive_queue_limits(field, value):
    with pytest.raises(ValueError):
        PipelineTransportConfig(**{field: value})
