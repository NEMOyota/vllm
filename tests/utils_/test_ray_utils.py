# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import vllm.v1.executor.ray_utils as ray_utils
from vllm.v1.executor.ray_utils import (
    _create_pipeline_placement_group_specs,
    get_bundles_for_pipeline_stages,
    get_bundles_sorted_by_node,
    get_pipeline_stage_node_ips,
    initialize_ray_cluster,
)

NODE_A = "node_a"
NODE_B = "node_b"
NODE_C = "node_c"

IP_A = "10.0.0.1"
IP_B = "10.0.0.2"
IP_C = "10.0.0.3"

NODE_ID_TO_IP = {NODE_A: IP_A, NODE_B: IP_B, NODE_C: IP_C}

MOCK_RAY_NODES = [
    {"NodeID": NODE_A, "NodeManagerAddress": IP_A, "Alive": True},
    {"NodeID": NODE_B, "NodeManagerAddress": IP_B, "Alive": True},
    {"NodeID": NODE_C, "NodeManagerAddress": IP_C, "Alive": True},
]


pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "bundles_to_node_id,bundle_specs,expected",
    [
        pytest.param(
            {0: NODE_C, 1: NODE_A, 2: NODE_B, 3: NODE_C, 4: NODE_A, 5: NODE_B},
            [{"GPU": 1}] * 6,
            [
                (1, NODE_A, IP_A),
                (4, NODE_A, IP_A),
                (2, NODE_B, IP_B),
                (5, NODE_B, IP_B),
                (0, NODE_C, IP_C),
                (3, NODE_C, IP_C),
            ],
        ),
        pytest.param(
            {0: NODE_B, 1: NODE_B, 2: NODE_A, 3: NODE_A},
            [{"GPU": 1}] * 4,
            [
                (2, NODE_A, IP_A),
                (3, NODE_A, IP_A),
                (0, NODE_B, IP_B),
                (1, NODE_B, IP_B),
            ],
        ),
        pytest.param(
            {0: NODE_C, 1: NODE_B, 2: NODE_C, 3: NODE_B},
            [{"GPU": 1}] * 4,
            [
                (1, NODE_B, IP_B),
                (3, NODE_B, IP_B),
                (0, NODE_C, IP_C),
                (2, NODE_C, IP_C),
            ],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_A, 2: NODE_A},
            [{"GPU": 1}] * 3,
            [(0, NODE_A, IP_A), (1, NODE_A, IP_A), (2, NODE_A, IP_A)],
        ),
        pytest.param(
            {},
            [],
            [],
        ),
        pytest.param(
            {0: NODE_A, 1: NODE_B, 2: NODE_A},
            [{"CPU": 1}, {"GPU": 1}, {"GPU": 1}],
            [(2, NODE_A, IP_A), (1, NODE_B, IP_B)],
        ),
    ],
)
def test_get_bundles_sorted_by_node(bundles_to_node_id, bundle_specs, expected):
    mock_pg = MagicMock()
    mock_pg.bundle_specs = bundle_specs

    mock_ctx = MagicMock()
    mock_ctx.get_node_id.return_value = NODE_A

    with (
        patch(
            "vllm.v1.executor.ray_utils.placement_group_table",
            return_value={"bundles_to_node_id": bundles_to_node_id},
        ),
        patch("vllm.v1.executor.ray_utils.ray") as mock_ray,
        patch("vllm.v1.executor.ray_utils.current_platform") as mock_platform,
    ):
        mock_ray.get_runtime_context.return_value = mock_ctx
        mock_ray.nodes.return_value = MOCK_RAY_NODES
        mock_platform.ray_device_key = "GPU"

        result = get_bundles_sorted_by_node(mock_pg)

    assert result == expected


@pytest.fixture
def pipeline_cluster(monkeypatch):
    monkeypatch.delenv("VLLM_RAY_BUNDLE_INDICES", raising=False)
    monkeypatch.setenv("VLLM_RAY_PER_WORKER_GPUS", "1")
    ray = MagicMock()
    ray.is_initialized.return_value = True
    ray.nodes.return_value = MOCK_RAY_NODES
    ray.cluster_resources.return_value = {"GPU": 6}
    ray.get_runtime_context.return_value.get_node_id.return_value = NODE_C
    ray.util.get_current_placement_group.return_value = None
    monkeypatch.setattr(ray_utils, "ray", ray)
    platform = MagicMock(ray_device_key="GPU")
    platform.is_cuda.return_value = False
    platform.is_rocm.return_value = False
    platform.is_xpu.return_value = False
    monkeypatch.setattr(ray_utils, "current_platform", platform)
    monkeypatch.setattr("vllm.platforms.current_platform", platform)
    available = MagicMock(return_value={NODE_A: {"GPU": 4}, NODE_B: {"GPU": 2}})
    monkeypatch.setattr(ray_utils, "available_resources_per_node", available)
    return ray, available


def test_pipeline_bundles_reserve_separate_head_and_tail_devices(pipeline_cluster):
    """The two edge stages each reserve their own TP-sized device set."""
    specs = _create_pipeline_placement_group_specs([IP_A, IP_B, IP_A], 2, "GPU")
    assert specs == [
        {"GPU": 1.0, f"node:{ip}": 0.001} for ip in [IP_A, IP_A, IP_B, IP_B, IP_A, IP_A]
    ]


@pytest.mark.parametrize("available_edge", [0, 2, 3])
def test_pipeline_bundles_reject_insufficient_combined_edge_capacity(
    pipeline_cluster, available_edge
):
    _, available = pipeline_cluster
    available.return_value[NODE_A]["GPU"] = available_edge
    with pytest.raises(ValueError, match="require 4 distinct GPUs"):
        _create_pipeline_placement_group_specs([IP_A, IP_B, IP_A], 2, "GPU")


def test_pipeline_bundles_reject_missing_or_dead_node(pipeline_cluster):
    ray, _ = pipeline_cluster
    ray.nodes.return_value = [
        MOCK_RAY_NODES[0],
        {**MOCK_RAY_NODES[1], "Alive": False},
    ]
    with pytest.raises(ValueError, match="not alive.*10.0.0.2"):
        _create_pipeline_placement_group_specs([IP_A, IP_B, IP_A], 2, "GPU")


def test_pipeline_existing_bundles_follow_stage_order_without_reuse(
    pipeline_cluster, monkeypatch
):
    pg = SimpleNamespace(bundle_specs=[{"GPU": 1}] * 7)
    pg.bundle_specs[1] = {"CPU": 1}
    monkeypatch.setattr(
        ray_utils,
        "placement_group_table",
        lambda _: {
            "bundles_to_node_id": {
                0: NODE_A,
                1: NODE_C,
                2: NODE_B,
                3: NODE_A,
                4: NODE_B,
                5: NODE_A,
                6: NODE_A,
            }
        },
    )
    assignments = get_bundles_for_pipeline_stages(pg, [IP_A, IP_B, IP_A], 2)
    assert assignments == [
        (0, NODE_A, IP_A),
        (3, NODE_A, IP_A),
        (2, NODE_B, IP_B),
        (4, NODE_B, IP_B),
        (5, NODE_A, IP_A),
        (6, NODE_A, IP_A),
    ]


def test_pipeline_existing_pg_rejects_capacity_on_wrong_hosts(
    pipeline_cluster, monkeypatch
):
    pg = SimpleNamespace(bundle_specs=[{"GPU": 1}] * 6)
    monkeypatch.setattr(
        ray_utils,
        "placement_group_table",
        lambda _: {
            "bundles_to_node_id": {
                0: NODE_A,
                1: NODE_A,
                2: NODE_B,
                3: NODE_B,
                4: NODE_C,
                5: NODE_C,
            }
        },
    )
    with pytest.raises(ValueError, match="pipeline stage 2"):
        get_bundles_for_pipeline_stages(pg, [IP_A, IP_B, IP_A], 2)


@pytest.mark.parametrize("existing_pg", [False, True])
def test_initialize_ray_cluster_uses_explicit_pipeline_hosts(
    pipeline_cluster, monkeypatch, existing_pg
):
    """Exercise real PG initialization with a CPU-only driver outside the layout."""
    ray, available = pipeline_cluster
    specs = [
        {"GPU": 1.0, f"node:{ip}": 0.001} for ip in [IP_A, IP_A, IP_B, IP_B, IP_A, IP_A]
    ]
    pg = SimpleNamespace(bundle_specs=specs)
    cfg = SimpleNamespace(
        pipeline_partition_config=SimpleNamespace(stage_node_ips=[IP_A, IP_B, IP_A]),
        tensor_parallel_size=2,
        world_size=6,
        placement_group=pg if existing_pg else None,
    )
    ray.util.placement_group.return_value = pg
    monkeypatch.setattr(ray_utils, "_wait_until_pg_ready", lambda _: None)
    monkeypatch.setattr(
        ray_utils,
        "placement_group_table",
        lambda _: {
            "bundles": specs,
            "bundles_to_node_id": {
                index: NODE_A if index in (0, 1, 4, 5) else NODE_B for index in range(6)
            },
        },
    )
    if existing_pg:
        available.return_value = {}

    initialize_ray_cluster(cfg)

    assert cfg.placement_group is pg
    if existing_pg:
        ray.util.placement_group.assert_not_called()
        available.assert_not_called()
    else:
        ray.util.placement_group.assert_called_once_with(specs, strategy="PACK")


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("VLLM_RAY_BUNDLE_INDICES", "0,1,2", "cannot be combined"),
        ("VLLM_RAY_PER_WORKER_GPUS", "0.5", "distinct devices"),
    ],
)
def test_pipeline_placement_rejects_conflicting_ray_options(
    pipeline_cluster, monkeypatch, name, value, message
):
    monkeypatch.setenv(name, value)
    cfg = SimpleNamespace(
        pipeline_partition_config=SimpleNamespace(stage_node_ips=[IP_A, IP_B, IP_A])
    )
    with pytest.raises(ValueError, match=message):
        get_pipeline_stage_node_ips(cfg)


@pytest.mark.parametrize(
    "driver_node,expected_local_ranks",
    [(NODE_A, [0, 1, 4, 5]), (NODE_B, [2, 3]), (NODE_C, [])],
)
def test_ray_v2_broadcast_maps_noncontiguous_local_pipeline_ranks(
    pipeline_cluster, monkeypatch, driver_node, expected_local_ranks
):
    """A returning edge stage must use SHM with its actual global rank."""
    from vllm.v1.executor import ray_executor_v2

    ray, _ = pipeline_cluster
    ray.get_runtime_context.return_value.get_node_id.return_value = driver_node
    monkeypatch.setattr(ray_executor_v2, "ray", ray)
    monkeypatch.setattr(ray_executor_v2, "initialize_ray_cluster", MagicMock())
    monkeypatch.setattr(
        ray_executor_v2,
        "get_bundles_for_pipeline_stages",
        lambda *_: [
            (rank, node, NODE_ID_TO_IP[node])
            for rank, node in enumerate(
                [NODE_A, NODE_A, NODE_B, NODE_B, NODE_A, NODE_A]
            )
        ],
    )
    executor = ray_executor_v2.RayExecutorV2.__new__(ray_executor_v2.RayExecutorV2)
    executor.parallel_config = SimpleNamespace(
        pipeline_partition_config=SimpleNamespace(stage_node_ips=[IP_A, IP_B, IP_A]),
        placement_group=MagicMock(),
    )
    executor.world_size = 6
    monkeypatch.setattr(executor, "_get_parallel_sizes", lambda: (2, 3, 1))

    class QueueCreated(Exception):
        pass

    queue_factory = MagicMock(side_effect=QueueCreated)
    monkeypatch.setattr(ray_executor_v2, "MessageQueue", queue_factory)
    with pytest.raises(QueueCreated):
        executor._init_executor()

    assert queue_factory.call_args.args == (6, len(expected_local_ranks))
    assert queue_factory.call_args.kwargs["local_reader_ranks"] == expected_local_ranks


@pytest.mark.parametrize("failure", [None, "wrong_node", "duplicate_ip"])
def test_legacy_ray_pipeline_placement_uses_node_ids_not_communication_ips(
    pipeline_cluster, monkeypatch, failure
):
    """Ray management addresses may differ from per-node VLLM_HOST_IP values."""
    from vllm.v1.executor import ray_executor

    ray, _ = pipeline_cluster
    monkeypatch.setattr(ray_executor, "ray", ray)
    monkeypatch.setattr(ray_executor, "current_platform", ray_utils.current_platform)
    monkeypatch.setattr(
        ray_utils.current_platform,
        "device_control_id_to_physical_device_id",
        MagicMock(side_effect=int),
    )
    monkeypatch.setattr(ray_executor, "get_env_vars_to_copy", lambda **_: set())
    monkeypatch.setattr(ray_executor, "PlacementGroupSchedulingStrategy", MagicMock())
    monkeypatch.setattr(
        ray_executor,
        "get_bundles_for_pipeline_stages",
        lambda *_: [(0, NODE_A, IP_A), (1, NODE_B, IP_B), (2, NODE_A, IP_A)],
    )
    nodes = [NODE_A, NODE_B, NODE_C if failure == "wrong_node" else NODE_A]
    communication_ips = ["192.168.1.1", "192.168.1.2", "192.168.1.1"]
    if failure == "duplicate_ip":
        communication_ips[1] = communication_ips[0]
    workers = [MagicMock() for _ in range(3)]
    for rank, worker in enumerate(workers):
        worker.get_node_ip.remote.return_value = communication_ips[rank]
        worker.get_node_and_physical_gpu_ids.remote.return_value = (
            nodes[rank],
            [1 if rank == 2 else 0],
        )
    workers[0].create_dist_init_method.remote.return_value = "tcp://192.168.1.1:12345"
    ray.remote.return_value.return_value.remote.side_effect = workers
    ray.get.side_effect = lambda result: result

    executor = ray_executor.RayDistributedExecutor.__new__(
        ray_executor.RayDistributedExecutor
    )
    executor.parallel_config = SimpleNamespace(
        pipeline_partition_config=SimpleNamespace(stage_node_ips=[IP_A, IP_B, IP_A]),
        ray_workers_use_nsight=False,
        world_size=3,
        tensor_parallel_size=1,
        pipeline_parallel_size=3,
    )
    executor.vllm_config = SimpleNamespace(parallel_config=executor.parallel_config)
    rpc = MagicMock()
    monkeypatch.setattr(executor, "collective_rpc", rpc)
    monkeypatch.setattr(executor, "_update_noset_device_env_vars", MagicMock())

    if failure is not None:
        message = (
            "do not match the pipeline stage placement"
            if failure == "wrong_node"
            else "Every node should have a unique IP address"
        )
        with pytest.raises(RuntimeError, match=message):
            executor._init_workers_ray(MagicMock())
        return

    executor._init_workers_ray(MagicMock())
    assert executor.workers == workers
    assert executor.pp_tp_workers == [[worker] for worker in workers]
    workers[0].create_dist_init_method.remote.assert_called_once_with(3)
    init_call = next(
        call for call in rpc.call_args_list if call.args == ("init_worker",)
    )
    assert [kwargs["local_rank"] for kwargs in init_call.kwargs["args"][0]] == [0, 0, 1]
