# Head-tail pipeline examples

These templates run a single pipeline-parallel model replica. A local node runs
the head and tail on separate GPUs, and a remote node runs the middle stage.
See the [feature guide](../../../docs/features/wan_head_tail_pipeline.md) for
partitioning rules, transport behavior, scope, and validation steps.

## Requirements

- A Ray cluster running the same vLLM revision and model on all nodes.
- At least two available GPUs on the local node and one on the remote node for
  the supplied TP=1, PP=3 configurations.
- The Ray V2 executor, GPU worker, and V2 model runner. The platform/model must
  support that runner. Zero-layer endpoints use native dense text models with
  eager execution; these examples use
  `Qwen/Qwen2.5-0.5B-Instruct`.

Follow the [Ray deployment guide](../../../docs/serving/parallelism_scaling.md#multi-node-deployment)
to start the cluster. Replace `10.0.0.10` and `10.0.0.20` in the YAML files with
its Ray `NodeManagerAddress` values. The repeated local address does not allow
sharing one GPU between two stages.

## Choose a partition

| File | First stage | Middle stage | Last stage |
| --- | --- | --- | --- |
| `explicit.yaml` | Embedding + 2 layers | 20 layers | 2 layers + output head |
| `weighted.yaml` | Embedding + 4 layers | 16 layers | 4 layers + output head |
| `encoding_only.yaml` | Embedding | All 24 layers | Final norm + output head |

The counts apply to the example model. Weighted allocation uses configured
relative capacities `[1, 4, 1]`; it does not measure the GPUs automatically.

From the repository root, launch one server:

```bash
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 VLLM_USE_V2_MODEL_RUNNER=1 \
    vllm serve --config examples/features/head_tail_pipeline/explicit.yaml
```

Select `weighted.yaml` or `encoding_only.yaml` in the same command for the other
layouts. Keep `VLLM_PP_LAYER_PARTITION` and `VLLM_RAY_BUNDLE_INDICES` unset and
`VLLM_RAY_PER_WORKER_GPUS` at its default value of `1`.

After startup, send a deterministic completion:

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"The capital of France is","temperature":0,"max_tokens":16}'
```

The templates enable chunked prefill with a 512-token scheduling budget and a
256 MiB per-worker retained-send budget. Tune those values for the real model
and workload. A send exceeding its byte budget is rejected. To compare the
ordinary PP path, set `pipeline-transport-config.enabled: false` in a copy of the
selected YAML and keep the other settings unchanged.

Local embedding still uses a GPU, and scheduler messages still carry token IDs
to workers. These examples neither implement a privacy boundary nor disaggregate
prefill and decode. Their WAN performance must be measured on the target link.
