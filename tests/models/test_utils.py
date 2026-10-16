# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import regex as re
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


class ModuleWithBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(x)


class ModuleWithNestedBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nested_mod = ModuleWithBatchNorm()

    def forward(self, x):
        return self.nested_mod(x)


@pytest.mark.cpu_test
def test_module_with_batchnorm_can_load():
    """Ensure the auto weight loader can load batchnorm stats."""
    mod = ModuleWithBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithBatchNorm()

    assert not torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert not torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_with_child_containing_batchnorm_can_autoload():
    """Ensure the auto weight loader can load nested modules batchnorm stats."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


VOCAB_SIZE = 16
HIDDEN_SIZE = 2


class ModuleWithTiedWeights(torch.nn.Module):
    """Mimics how models tie `lm_head` to the input embeddings."""

    def __init__(self, tie: bool):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
        if tie:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)


def make_embedding_weights(value: float) -> torch.Tensor:
    return torch.full((VOCAB_SIZE, HIDDEN_SIZE), value)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("tie", [True, False])
def test_module_skip_tied_weights(tie: bool):
    """Tied weights must be loaded once, under the first of their names."""
    mod = ModuleWithTiedWeights(tie)

    weights = [
        ("model.embed_tokens.weight", make_embedding_weights(1.0)),
        ("lm_head.weight", make_embedding_weights(2.0)),
    ]
    loaded = AutoWeightsLoader(mod).load_weights(iter(weights))

    if tie:
        assert loaded == {"model.embed_tokens.weight"}
        assert torch.all(mod.lm_head.weight[:VOCAB_SIZE] == 1.0)
    else:
        assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
        assert torch.all(mod.lm_head.weight[:VOCAB_SIZE] == 2.0)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_module_skip_tied_weights_without_canonical():
    """Skipping a tied weight must not leave the shared weight uninitialized."""
    mod = ModuleWithTiedWeights(tie=True)

    weights = [("lm_head.weight", make_embedding_weights(2.0))]
    with pytest.raises(ValueError, match="model.embed_tokens.weight"):
        AutoWeightsLoader(mod).load_weights(iter(weights))


class ModuleWithSharedParam(torch.nn.Module):
    """Mimics an MoE router shared between the MLP and its fused experts."""

    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Module()
        self.experts.gate = torch.nn.Linear(2, 2, bias=False)
        self.gate = self.experts.gate


@pytest.mark.cpu_test
def test_module_load_shared_params_that_are_not_tied_embeddings():
    """Only tied embeddings are skipped; other shared params must still load."""
    mod = ModuleWithSharedParam()

    weights = [("gate.weight", torch.Tensor([[1, 2], [3, 4]]))]
    loaded = AutoWeightsLoader(mod).load_weights(iter(weights))

    assert loaded == {"gate.weight"}
    assert torch.all(mod.gate.weight == torch.Tensor([[1, 2], [3, 4]]))


class raise_if_cuda_sync:
    def __enter__(self):
        self.previous_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.previous_debug_mode)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_merge_multimodal_embeddings_no_sync():
    inputs_embeds = torch.zeros(
        [5, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0"
    )
    multimodal_embeddings = [
        torch.ones([3, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0")
    ]
    is_multimodal = torch.tensor([True, False, True, True, False], device="cpu")
    with raise_if_cuda_sync():
        _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )


@pytest.mark.cpu_test
def test_get_rename_mapper_keeps_only_renames():
    """`None` means "do not load", which is meaningless to the consumers of
    this mapper (LoRA name parsing, quantization config layer lists), and
    applying it would silently shrink their lists."""
    mapper = WeightsMapper(
        orig_to_new_regex={re.compile(r"^drop_regex\."): None},
        orig_to_new_substr={"drop_substr": None, "keep_substr": "kept"},
        orig_to_new_stacked={".q_proj": (".qkv_proj", "q")},
        orig_to_new_prefix={"drop_prefix.": None, "keep_prefix.": "kept."},
        orig_to_new_suffix={".drop_suffix": None},
    )
    renames = mapper.get_rename_mapper()

    assert renames.orig_to_new_regex == {}
    assert renames.orig_to_new_substr == {"keep_substr": "kept"}
    assert renames.orig_to_new_stacked == {}
    assert renames.orig_to_new_prefix == {"keep_prefix.": "kept."}
    assert renames.orig_to_new_suffix == {}

    # Names the full mapper drops now survive unchanged.
    for name in ("drop_regex.w", "drop_substr.w", "drop_prefix.w", "w.drop_suffix"):
        assert mapper._map_name(name) is None
        assert renames._map_name(name) == name


def test_weights_mapper_stacks_one_weight_into_several_shards():
    """One checkpoint tensor can feed more than one shard of a stacked
    parameter, and a dropped name stays dropped rather than reappearing as a
    shard of the stacked one."""
    mapper = WeightsMapper(
        orig_to_new_substr={"layers.1.k_proj.": None},
        orig_to_new_stacked={".k_proj.": (".qkv_proj.", ["k", "v"])},
    )
    weight = torch.ones(2)
    weights = [(f"layers.{i}.k_proj.weight", weight) for i in (0, 1)]

    mapped = list(mapper.apply(weights))

    assert [(name, w.shard_id) for name, w in mapped] == [
        ("layers.0.qkv_proj.weight", "k"),
        ("layers.0.qkv_proj.weight", "v"),
    ]
    # Each shard needs its own tensor object, but they alias one allocation.
    assert mapped[0][1] is not mapped[1][1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr() == weight.data_ptr()


@pytest.mark.cpu_test
@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("partition", [(0, 4, 0), (1, 2, 1), (0, 3, 1), (1, 3, 0)])
@pytest.mark.parametrize("tied", [False, True])
def test_dense_pp_endpoints_match_unsplit_model(
    monkeypatch, tmp_path, family, partition, tied
):
    """Run real model projections/norms/MLPs with CPU SDPA in place of paged KV.

    This exercises prefill chunks and decode after checkpoint weight loading;
    only the device-specific attention cache kernel and process groups are replaced.
    """
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as F
    import transformers

    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import (
        CompilationConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    from vllm.sequence import IntermediateTensors

    config_cls, model_cls = {
        "llama": (transformers.LlamaConfig, LlamaForCausalLM),
        "qwen2": (transformers.Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (transformers.Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
        architectures=[model_cls.__name__],
    ).save_pretrained(tmp_path)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    cache: dict[Attention, tuple[torch.Tensor, torch.Tensor]] = {}

    def attention(layer, q, k, v, **kwargs):
        q = q.view(-1, layer.num_heads, layer.head_size).transpose(0, 1)
        k = k.view(-1, layer.num_kv_heads, layer.head_size).transpose(0, 1)
        v = v.view(-1, layer.num_kv_heads, layer.head_size).transpose(0, 1)
        previous = cache.get(layer)
        offset = 0 if previous is None else previous[0].shape[1]
        if previous is not None:
            k = torch.cat((previous[0], k), dim=1)
            v = torch.cat((previous[1], v), dim=1)
        cache[layer] = k, v
        mask = torch.arange(k.shape[1])[None, :] <= (
            offset + torch.arange(q.shape[1])[:, None]
        )
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return output.transpose(0, 1).reshape(-1, layer.num_heads * layer.head_size)

    monkeypatch.setattr(Attention, "forward", attention)

    def set_rank(rank, size):
        monkeypatch.setattr(
            parallel_state,
            "_PP",
            SimpleNamespace(
                rank_in_group=rank,
                world_size=size,
                is_first_rank=rank == 0,
                is_last_rank=rank == size - 1,
            ),
        )

    model_config = ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        dtype="float32",
        max_model_len=64,
    )

    def build_model():
        config = VllmConfig(
            model_config=model_config,
            compilation_config=CompilationConfig(custom_ops=["none"]),
        )
        with set_current_vllm_config(config):
            model = model_cls(vllm_config=config)
        return model, config

    torch.manual_seed(42)
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    set_rank(0, 1)
    reference, reference_config = build_model()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.1)
    checkpoint = [
        (name, p.detach().clone()) for name, p in reference.named_parameters()
    ]

    def finalize_weights(model):
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)

    finalize_weights(reference)

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, partition)))
    stages = []
    for rank in range(3):
        set_rank(rank, 3)
        stage, config = build_model()
        # Use the actual PP-aware loader, including tied embedding/head weights.
        stage.load_weights(iter(checkpoint))
        finalize_weights(stage)
        stages.append((stage, config))

    with torch.inference_mode():
        for tokens in (
            torch.tensor([1, 4, 8, 3, 2, 7]),
            torch.tensor([9, 2, 1, 6, 5, 4]),
        ):
            cache.clear()
            offset = 0
            for chunk_size in (3, 2, 1):
                input_ids = tokens[offset : offset + chunk_size]
                positions = torch.arange(offset, offset + chunk_size)
                set_rank(0, 1)
                with set_current_vllm_config(reference_config):
                    expected = reference(input_ids, positions)
                    expected_logits = reference.compute_logits(expected)

                intermediate = None
                for rank, (stage, config) in enumerate(stages):
                    set_rank(rank, 3)
                    with set_current_vllm_config(config):
                        result = stage(
                            input_ids if rank == 0 else None,
                            positions,
                            intermediate_tensors=intermediate,
                        )
                        if rank == 2:
                            logits = stage.compute_logits(result)
                        else:
                            assert isinstance(result, IntermediateTensors)
                            assert all(
                                isinstance(t, torch.Tensor)
                                for t in result.tensors.values()
                            )
                            intermediate = IntermediateTensors(
                                {key: value.clone() for key, value in result.items()}
                            )
                torch.testing.assert_close(
                    logits, expected_logits, atol=1e-6, rtol=1e-5
                )
                offset += chunk_size
