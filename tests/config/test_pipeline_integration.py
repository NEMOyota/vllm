# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
from transformers import LlamaConfig

from vllm.config import (
    ParallelConfig,
    PipelinePartitionConfig,
    PipelineTransportConfig,
    VllmConfig,
)
from vllm.distributed.utils import get_pp_indices
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def tiny_model(tmp_path):
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=12,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize(
    "partition,expected",
    [
        ({"head_layers": 2, "tail_layers": 3}, [(0, 2), (2, 9), (9, 12)]),
        ({"head_layers": 0, "tail_layers": 0}, [(0, 0), (0, 12), (12, 12)]),
        ({"stage_compute_weights": [1, 2, 1]}, [(0, 3), (3, 9), (9, 12)]),
        (
            {"stage_compute_weights": [1, 2, 1], "layer_costs": [1] * 12},
            [(0, 3), (3, 9), (9, 12)],
        ),
    ],
)
def test_cli_partition_reaches_model_layer_and_kv_counts(
    tiny_model, partition, expected
):
    """The serving CLI and model's KV sizing must resolve the same layer ranges."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(
        [
            "--model",
            tiny_model,
            "--skip-tokenizer-init",
            "--enforce-eager",
            "--distributed-executor-backend",
            "mp",
            "--pipeline-parallel-size",
            "3",
            "--pipeline-partition-config",
            json.dumps(partition),
            "--pipeline-transport-config",
            '{"max_inflight_sends": 3, "max_inflight_bytes": 4096, '
            '"async_receive": false}',
            "--max-model-len",
            "32",
            "--max-num-batched-tokens",
            "32",
            "--max-num-seqs",
            "4",
        ]
    )
    config = EngineArgs.from_cli_args(parsed).create_engine_config()
    assert config.parallel_config.pipeline_transport_config == PipelineTransportConfig(
        max_inflight_sends=3, max_inflight_bytes=4096, async_receive=False
    )
    for rank, bounds in enumerate(expected):
        config.parallel_config.rank = rank
        assert (
            config.model_config.get_layers_start_end_indices(config.parallel_config)
            == bounds
        )
        assert (
            config.model_config.get_num_layers_by_block_type(config.parallel_config)
            == bounds[1] - bounds[0]
        )


def test_conflicting_partition_sources_fail_before_worker_launch(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,8,2")
    with pytest.raises(ValueError, match="not both"):
        ParallelConfig(
            pipeline_parallel_size=3,
            distributed_executor_backend="mp",
            pipeline_partition_config={"head_layers": 2, "tail_layers": 2},
        )


def test_legacy_partition_rejects_negative_counts(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "-1,12,1")
    with pytest.raises(ValueError, match="non-negative"):
        get_pp_indices(12, 0, 3)


@pytest.mark.parametrize(
    "partition,expected",
    [
        ("2,7,3", [(0, 2), (2, 9), (9, 12)]),
        ("0,12,0", [(0, 0), (0, 12), (12, 12)]),
    ],
)
def test_legacy_partition_preserves_explicit_ranges(monkeypatch, partition, expected):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)

    assert [get_pp_indices(12, rank, 3) for rank in range(3)] == expected


def test_model_layer_ranges_follow_pp_pcp_tp_rank_order(tiny_model):
    model = EngineArgs(model=tiny_model, skip_tokenizer_init=True).create_model_config()
    parallel = ParallelConfig(
        pipeline_parallel_size=3,
        prefill_context_parallel_size=2,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        pipeline_partition_config={"head_layers": 2, "tail_layers": 3},
    )
    expected = [(0, 2)] * 4 + [(2, 9)] * 4 + [(9, 12)] * 4

    for rank, bounds in enumerate(expected):
        parallel.rank = rank
        assert model.get_layers_start_end_indices(parallel) == bounds
        assert model.get_num_layers_by_block_type(parallel) == bounds[1] - bounds[0]


def test_partition_changes_compilation_hash_but_rank_does_not():
    parallel = ParallelConfig(
        pipeline_parallel_size=3,
        distributed_executor_backend="mp",
        pipeline_partition_config={"head_layers": 2, "tail_layers": 2},
    )
    first = parallel.compute_hash()
    parallel.rank = 2
    assert parallel.compute_hash() == first
    assert parallel.pipeline_partition_config is not None
    parallel.pipeline_partition_config.head_layers = 3
    assert parallel.compute_hash() != first


@pytest.mark.parametrize("backend", ["uni", "external_launcher"])
def test_transport_rejects_executors_without_worker_integration(backend):
    with pytest.raises(ValueError, match="mp or Ray"):
        ParallelConfig(
            pipeline_parallel_size=3,
            distributed_executor_backend=backend,
            pipeline_transport_config={"enabled": True},
        )


def test_transport_rejects_legacy_ray_dag(monkeypatch):
    monkeypatch.setenv("VLLM_USE_RAY_V2_EXECUTOR_BACKEND", "0")
    with pytest.raises(ValueError, match="Ray V2"):
        ParallelConfig(
            pipeline_parallel_size=3,
            distributed_executor_backend="ray",
            pipeline_transport_config={"enabled": True},
        )


@pytest.mark.parametrize(
    "cuda_alike,runner_v2,match",
    [(False, True, "CUDA/ROCm"), (True, False, "V2 model runner")],
)
def test_transport_rejects_unsupported_platform_or_model_runner(
    tiny_model, monkeypatch, cuda_alike, runner_v2, match
):
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: cuda_alike)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if runner_v2 else "0")

    with pytest.raises(ValueError, match=match):
        EngineArgs(
            model=tiny_model,
            skip_tokenizer_init=True,
            enforce_eager=True,
            pipeline_parallel_size=3,
            distributed_executor_backend="mp",
            pipeline_transport_config={"enabled": True},
            max_model_len=32,
        ).create_engine_config()


def test_zero_endpoints_require_eager_execution(tiny_model):
    with pytest.raises(ValueError, match="enforce-eager"):
        EngineArgs(
            model=tiny_model,
            skip_tokenizer_init=True,
            pipeline_parallel_size=3,
            distributed_executor_backend="mp",
            pipeline_partition_config={"head_layers": 0, "tail_layers": 0},
            max_model_len=32,
        ).create_engine_config()


@pytest.mark.parametrize("mode", ["partition", "transport"])
def test_pipeline_rejects_speculative_decoding_before_draft_loading(tiny_model, mode):
    with pytest.raises(ValueError, match="speculative decoding"):
        EngineArgs(
            model=tiny_model,
            skip_tokenizer_init=True,
            enforce_eager=True,
            pipeline_parallel_size=3,
            distributed_executor_backend="mp",
            pipeline_partition_config=(
                {"head_layers": 2, "tail_layers": 3} if mode == "partition" else None
            ),
            pipeline_transport_config={"enabled": mode == "transport"},
            speculative_config={"method": "ngram", "num_speculative_tokens": 2},
            max_model_len=32,
        ).create_engine_config()


@pytest.mark.parametrize("mode", ["zero_endpoints", "transport"])
@pytest.mark.parametrize("offloading_backend", ["native", "lmcache"])
def test_pipeline_rejects_implicitly_created_kv_offload_connector(
    tiny_model, monkeypatch, mode, offloading_backend
):
    """Offloading creates its connector after the initial pipeline validation."""
    if mode == "transport":
        monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    with pytest.raises(ValueError, match="offloading"):
        EngineArgs(
            model=tiny_model,
            skip_tokenizer_init=True,
            enforce_eager=True,
            pipeline_parallel_size=3,
            distributed_executor_backend="mp",
            pipeline_partition_config=(
                {"head_layers": 0, "tail_layers": 0}
                if mode == "zero_endpoints"
                else None
            ),
            pipeline_transport_config={"enabled": mode == "transport"},
            kv_offloading_size=1,
            kv_offloading_backend=offloading_backend,
            max_model_len=32,
            max_num_batched_tokens=32,
            max_num_seqs=4,
        ).create_engine_config()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"architecture": "Qwen3MoeForCausalLM", "is_moe": True}, "native dense"),
        ({"is_hybrid": True}, "native dense"),
        ({"model_impl": "transformers"}, "native dense"),
        ({"multimodal_config": object()}, "native dense"),
    ],
)
def test_zero_endpoint_validation_rejects_unsupported_model_paths(overrides, match):
    model = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 12,
        architecture="LlamaForCausalLM",
        model_impl="vllm",
        runner_type="generate",
        is_moe=False,
        is_hybrid=False,
        multimodal_config=None,
        enforce_eager=True,
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    config = SimpleNamespace(
        model_config=model,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_partition_config=PipelinePartitionConfig(head_layers=0),
            pipeline_transport_config=PipelineTransportConfig(),
            pipeline_parallel_size=3,
        ),
    )
    with pytest.raises(ValueError, match=match):
        VllmConfig._resolve_pipeline_partition(config)
