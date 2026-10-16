# Head-tail pipeline placement across nodes

Head-tail pipeline placement runs one model replica as an ordered pipeline,
with the first and last stages on a local node and the middle stages on other
nodes. It adds explicit or weighted transformer-layer partitioning, stage-to-node
placement, and an optional bounded asynchronous pipeline transport.

This is an experimental, opt-in feature. Current validation covers CPU
configuration, reference-model, and transport-protocol tests. Multi-node
GPU/NCCL validation and WAN performance measurements are still pending.

The first hybrid acceptance target is the non-MoE Qwen3.5 language tower.
Here, hybrid means Gated DeltaNet (GDN) linear-attention layers mixed with full
attention, not a mixture of experts. Dense Llama, Qwen2, and Qwen3 configurations
remain available as simpler reference cases.

The objective is to use remote compute while controlling communication memory
and exposing opportunities to overlap communication with independent work.
Whether this helps a particular WAN deployment requires measurement: the
pipeline crosses its stage boundaries during both prefill and every decode step.
For enterprise burst capacity, long-input, short-output workloads are a useful
starting point: remote compute may amortize the communication cost. Interactive
long-output workloads repeatedly incur cross-node decode dependencies. Concurrent
batches can improve link utilization, while a single request still waits for its
own preceding stages.

## Architecture

For three pipeline stages with `tensor_parallel_size=1`:

```mermaid
flowchart LR
    subgraph Local[Local node A]
        H[GPU 0: embedding and head layers]
        T[GPU 1: tail layers, final norm and LM head]
    end
    subgraph Remote[Remote node B]
        M[GPU 0: middle layers]
    end
    H -->|Intermediate tensors| M
    M -->|Intermediate tensors| T
```

Each stage owns a contiguous layer range and its corresponding attention KV
cache or GDN recurrent/convolution state. The
existing model implementation still determines which intermediate tensors are
sent, such as hidden states and residuals. This is pipeline parallelism within
one replica; it does not create separate prefill and decode instances or transfer
their KV caches through a KV connector.

Repeating node A in the placement list places two independent stages on that
node. Each stage uses distinct devices, with the same tensor-parallel size across
the pipeline. For TP size `T`, node A needs at least `2*T` available devices and
node B needs at least `T`. A folded executor that reuses the same device for the
head and tail is not implemented.

### Partitioning

Pass a JSON object with `--pipeline-partition-config`, or use the equivalent
nested mapping in a `vllm serve --config` YAML file.

| Field | Meaning |
| --- | --- |
| `head_layers` | Fixed number of transformer layers on the first stage; zero is allowed. |
| `tail_layers` | Fixed number of transformer layers on the last stage; zero is allowed. |
| `stage_compute_weights` | Positive relative compute capacities, one per stage. Larger values receive more work. |
| `layer_costs` | Positive relative cost for every transformer layer, in model order. |
| `layer_type_costs` | Positive relative cost per HF layer type; expanded using the text model's `layer_types`. Mutually exclusive with `layer_costs`. |
| `stage_node_ips` | One Ray node address per pipeline stage; addresses may repeat. |

Setting either endpoint count fixes both endpoints; an omitted endpoint defaults
to zero. Fixed endpoints require at least three stages. Every middle stage must
receive at least one transformer layer, and the configured endpoints must leave
enough layers for those stages.

With no fixed endpoints, all stages receive at least one layer. Without
either cost option, weights divide layer counts proportionally using constrained
largest-remainder allocation. Equal remainders prefer lower pipeline ranks.
With costs, the planner chooses contiguous boundaries near cumulative weighted
cost targets. It is a partitioning heuristic, not an optimal latency solver.
Fixed endpoints ignore their weights when distributing the remaining layers.

For Qwen3.5, a mapping such as
`{"linear_attention": 1.0, "full_attention": 4.0}` expands to one cost per layer
using the local HF text configuration. These numbers only illustrate the API;
they are not measured GDN/full-attention performance ratios. Keys must exactly
match the layer types in the model; missing or unused keys are rejected. With
fixed zero endpoints and PP=3, all layers
belong to the sole middle stage, so cost weighting cannot change that boundary.

Weights and costs are user inputs, typically obtained from offline profiling.
There is no automatic hardware benchmarking, online weight adjustment, or
network-aware repartitioning. Profile the intended batch and context sizes, and
balance measured stage compute time rather than layer counts alone. Account
separately for embedding and LM-head work, per-stage KV memory, and link costs.

Hybrid partitions require the V2 model runner, including when the optional
transport is disabled, so stages with different local layer types use consistent
request ordering.

Do not combine this configuration with `VLLM_PP_LAYER_PARTITION`. Check the
`Resolved pipeline transformer layer counts` startup log before testing.

### Zero-transformer endpoints

With `head_layers: 0` and `tail_layers: 0`, stage 0 performs token embedding,
middle stages execute all transformer layers, and the last stage performs the
final normalization and LM head. The endpoints still require GPU memory and
compute; this is not a CPU-only frontend.

The zero-layer head embeds text token IDs, not images. Its intermediate-tensor
contract still sends hidden states and a full zero residual: the logical payload
is approximately `2 * T * H * dtype_bytes` for `T` scheduled tokens and hidden
size `H`, before metadata and backend overhead. No zero-residual wire compression
is implemented. Include both tensors when estimating the WAN traffic budget.

Zero-transformer endpoints support the native non-MoE Qwen3.5 language tower:
`Qwen3_5ForCausalLM`, or `Qwen3_5ForConditionalGeneration` with
`--language-model-only`. Native dense `LlamaForCausalLM`, `Qwen2ForCausalLM`, and
`Qwen3ForCausalLM` text generation are also supported. All require the V2 model
runner, `--enforce-eager`, and DP=PCP=DCP=1. Qwen3.5 MoE, image/video inputs, expert
parallelism, KV/encoder disaggregation, KV offloading, and microbatching/DBO are
outside the zero-endpoint scope. MTP/speculative decoding remains disabled.
Positive-layer partitions retain the model's ordinary pipeline-parallel support
requirements. Both new partition and transport configurations currently reject
speculative decoding.

For a hybrid partition, size both attention KV and GDN states on the stages that
own those layers. Empty endpoints own neither state type; an intermediate stage
may own only one type. A zero local Mamba-state allocation must not be treated as
evidence that the global model has no recurrent layers.
Prefix caching follows the native hybrid-cache rules: it requires chunked
prefill and aligned recurrent-state caching. Record the resolved cache mode and
keep cache-group identities consistent across stages when validating reuse.

Local embedding is not a privacy boundary. Native scheduler messages still
distribute token IDs, and intermediate representations may contain sensitive
information. Use the existing
[distributed deployment network requirements](../serving/parallelism_scaling.md#ray-cluster-setup-with-containers)
for the cluster's trusted connectivity.

### Bounded asynchronous transport

Enable the transport independently with:

```json
{
  "enabled": true,
  "max_inflight_sends": 2,
  "max_inflight_bytes": 268435456,
  "async_receive": true
}
```

The transport retains a snapshot of outgoing intermediate tensors until the
corresponding send completes. This avoids a later forward pass overwriting
buffers that communication still reads. Both pending-send count and retained
tensor bytes are bounded per worker transport; reaching a limit applies
backpressure. A single payload larger than the byte budget is rejected before
posting, so size this budget for the largest scheduled batch. These limits do
not bound total GPU memory, model state, receive buffers, or backend allocations.

With `async_receive: true`, receiving returns a ticket before tensor metadata has
arrived. The model runner can perform independent preparation before consuming
the incoming tensors. Consumption still waits for the matching transfer. The
transport preserves FIFO order and does not run a later ready batch ahead of an
earlier delayed batch.

This reuses native chunked-prefill and pipeline execution queues. It does not
increase engine scheduling depth, add a new request scheduler, or remove a
decode step's data dependencies. Overlap can cover some communication waiting
when independent work exists; RTT, bandwidth, and the transport backend's
retransmissions remain relevant. This layer does not implement packet repair.
Scheduler/control traffic and the model runner's sampled-token feedback retain
their existing paths; only pipeline intermediate-tensor transport is changed.

The transport currently requires the CUDA/ROCm GPU worker, the V2 model runner,
and the multiprocessing or Ray V2 executor. Set `VLLM_USE_V2_MODEL_RUNNER=1` and
use a platform/model combination that supports that runner. CPU execution,
KV/encoder disaggregation, KV offloading, and microbatching/DBO are excluded when
transport is enabled. Transport errors propagate to the worker/executor; this
feature does not provide session replay after failure. With this transport
disabled, positive-layer dense reference partitions can still use the V1 model
runner. Zero endpoints and hybrid partitions require an actually selected V2
model runner, independently of the transport setting.

## Configuration and launch

The runnable templates are in
[examples/features/head_tail_pipeline](../../examples/features/head_tail_pipeline/README.md).
Start with the Qwen3.5 language-only templates and replace
`/path/to/local/Qwen3.5` with the same local checkpoint path on every node. They
use TP=1, PP=3, eager execution, and native chunked prefill. The three
`Qwen/Qwen2.5-0.5B-Instruct` templates remain available for dense reference
checks. Templates demonstrate configuration, not a measured WAN benefit.

1. Prepare a Ray cluster using the same vLLM revision and model files on all
   nodes. Follow the [multi-node deployment guide](../serving/parallelism_scaling.md#multi-node-deployment).
   Verify the available GPUs and the exact `NodeManagerAddress` values with Ray.
2. Replace the example node addresses in the selected YAML with those addresses.
   Stage placement requires the Ray executor and one whole device per worker;
   leave `VLLM_RAY_PER_WORKER_GPUS=1` and do not set
   `VLLM_RAY_BUNDLE_INDICES` at the same time.
3. Launch one server against the running cluster:

    ```bash
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 VLLM_USE_V2_MODEL_RUNNER=1 \
        vllm serve --config examples/features/head_tail_pipeline/qwen3_5_hybrid.yaml
    ```

| Template | Partition |
| --- | --- |
| [`qwen3_5_hybrid.yaml`](../../examples/features/head_tail_pipeline/qwen3_5_hybrid.yaml) | Nonempty stages weighted by `[1, 4, 1]` and illustrative layer-type costs. Read resolved boundaries from the local Qwen3.5 configuration. |
| [`qwen3_5_encoding_only.yaml`](../../examples/features/head_tail_pipeline/qwen3_5_encoding_only.yaml) | `[0, L, 0]` for a Qwen3.5 language tower with `L` layers. |
| [`explicit.yaml`](../../examples/features/head_tail_pipeline/explicit.yaml) | Dense 24-layer reference: `[2, 20, 2]`. |
| [`weighted.yaml`](../../examples/features/head_tail_pipeline/weighted.yaml) | Dense 24-layer reference: `[4, 16, 4]`, weights `[1, 4, 1]`. |
| [`encoding_only.yaml`](../../examples/features/head_tail_pipeline/encoding_only.yaml) | Dense 24-layer reference: `[0, 24, 0]`. |

For a different model, verify its transformer-layer count and the resolved
partition. The YAML filename `encoding_only` refers to the first stage's token
embedding role; it does not enable multimodal encoder disaggregation.

The equivalent partition and transport CLI options are:

```bash
--pipeline-parallel-size 3 \
--pipeline-partition-config '{"head_layers":2,"tail_layers":2,"stage_node_ips":["10.0.0.10","10.0.0.20","10.0.0.10"]}' \
--pipeline-transport-config '{"enabled":true,"max_inflight_sends":2,"max_inflight_bytes":268435456,"async_receive":true}'
```

For a single-node smoke check, repeat that node's Ray address three times and
provide three distinct GPUs. To test partitioning without the new transport,
set `pipeline-transport-config.enabled: false` in a copy of the YAML.

## Correctness and performance validation

First run the CPU configuration, placement, and transport tests in a prepared
development environment. These validate planning and protocol behavior without
establishing GPU model correctness or WAN performance:

```bash
.venv/bin/python -m pytest \
    tests/config/test_pipeline.py \
    tests/config/test_pipeline_integration.py \
    tests/utils_/test_ray_utils.py \
    tests/distributed/test_pp_transport.py -q
```

On a CPU test environment, also run the model-reference and hybrid-state cases:

```bash
.venv/bin/python -m pytest \
    tests/models/test_utils.py::test_dense_pp_endpoints_match_unsplit_model \
    tests/v1/worker/test_gpu_model_runner.py::test_mamba_align_postprocess_uses_only_local_layers \
    tests/v1/worker/test_gpu_model_runner_v2.py::test_gather_batch_req_state_has_consistent_pp_order \
    tests/v1/worker/test_mamba_hybrid_model_state.py::test_hybrid_pp_stage_only_initializes_its_local_model_state -q
```

The model-reference cases use CPU attention/GDN reference kernels around native
model projections and state updates. They check partition equivalence, not the
production GPU recurrent kernels or inter-node execution.

Then validate on the target GPU cluster:

1. Compare deterministic generations with an unsplit reference using identical
   weights, dtype, prompts, sampling parameters, and context limits. Exercise
   short prompts, multiple prefill chunks, concurrent requests, cancellation,
   and requests near the context limit. Compare token outputs and investigate
   numerical differences with log probabilities or model evaluation.
2. Run the same partition with the new transport disabled and enabled. Check
   that no request receives stale intermediate data as buffers are reused.
   Exercise a batch near the configured byte budget and verify oversized
   batches fail clearly rather than hang.
3. Confirm stage-to-node placement and distinct GPU allocation. For zero-layer
   Qwen3.5 endpoints, verify that the middle stage owns both attention KV and GDN
   states, and that first/last stages perform their embedding/output roles.
   Also test a hybrid partition with a stage containing only GDN layers and one
   containing only full-attention layers. Include prefill followed by multiple
   decode steps; a single forward does not validate recurrent state updates.
4. Test link delay, constrained bandwidth, and transport failure on an isolated
   test network. Record the actual path and backend; host TCP shaping does not
   necessarily affect an RDMA path. A failed run must terminate visibly rather
   than silently reusing buffers or producing another request's output.

Use `vllm bench serve` for end-to-end measurements, for example:

```bash
vllm bench serve \
    --backend vllm \
    --model /path/to/local/Qwen3.5 \
    --served-model-name qwen3.5-head-tail \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 128 \
    --num-prompts 200 \
    --request-rate 2 \
    --max-concurrency 8 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-filename head-tail.json
```

Sweep prompt length, output length, offered load, chunk size, partition, and
send budgets while holding the model and hardware constant. Compare local
execution where feasible, ordinary PP, and each head-tail variant. Use separate
result filenames and separate cold-cache and warm-cache runs. If evaluating
goodput, supply workload-specific `--goodput ttft:<ms> tpot:<ms>` thresholds;
average TPOT does not replace tail ITL measurements.

Report TTFT, TPOT and tail ITL, successful requests per second, errors, device
memory, per-stage utilization, link throughput, and actual RTT. Capture a
timeline to establish whether communication overlaps useful compute. The
configuration weights and budgets are tuning inputs; no speedup, WAN distance,
or loss tolerance is promised by these examples.
