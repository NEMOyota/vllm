# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/Gloo coverage of PP transport ownership, progress, and failure."""

import gc
import multiprocessing as mp
import threading
import time
import weakref
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_transport import PipelineTransport, PipelineTransportError

# All process groups and transport owners live in the disposable child processes.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _GroupWrapper:
    def __init__(self, group):
        self.group = group

    def __getattr__(self, name):
        return getattr(self.group, name)


class _DeferredSend:
    """Hold a real Gloo send until the caller has reused its input buffer."""

    def __init__(self, group, payload, kwargs, release):
        self.group = group
        self.payload = payload
        self.kwargs = kwargs
        self.release = release
        self.done = threading.Event()

    def is_completed(self):
        return self.done.is_set()

    def wait(self):
        assert self.release.wait(10), "send was never released"
        for work in self.group.isend_tensor_dict(self.payload, **self.kwargs):
            work.wait()
        self.done.set()


class _DelayedSendGroup(_GroupWrapper):
    def __init__(self, group, release):
        super().__init__(group)
        self.release = release
        self.posts = 0

    def isend_tensor_dict(self, payload, **kwargs):
        self.posts += 1
        return [_DeferredSend(self.group, payload, kwargs, self.release)]


class _ObservedReceiveGroup(_GroupWrapper):
    def __init__(self, group):
        super().__init__(group)
        self.consumer_thread = threading.get_ident()
        self.postprocessed = False

    def irecv_tensor_dict(self, **kwargs):
        payload, works, postprocess = self.group.irecv_tensor_dict(**kwargs)

        def observe_completed_receive():
            assert threading.get_ident() == self.consumer_thread
            torch.testing.assert_close(payload["hidden"], torch.arange(16))
            self.postprocessed = True

        return payload, works, [*postprocess, observe_completed_receive]


def _receive_before_metadata(rank, group, signals):
    wrapper = _ObservedReceiveGroup(group)
    transport = PipelineTransport(wrapper, async_receive=True)
    try:
        if rank == 0:
            # The sender cannot produce metadata until receive() has returned.
            assert signals[0].wait(10), "receive() blocked on peer metadata"
            transport.send(
                {
                    "hidden": torch.arange(16),
                    "residual": None,
                    "empty": torch.empty((0, 3), dtype=torch.float32),
                    "label": "step zero",
                }
            )
            transport.drain()
        else:
            from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

            ticket = transport.receive()
            with pytest.raises(RuntimeError):
                transport.receive()
            with patch.object(ticket, "wait", wraps=ticket.wait) as wait:
                intermediate = AsyncIntermediateTensors({}, receive_ticket=ticket)
                wait.assert_not_called()
                signals[0].set()
                payload = intermediate.tensors
                assert intermediate.tensors is payload
                wait.assert_called_once_with()
            assert wrapper.postprocessed
            assert payload["residual"] is None
            assert payload["empty"].shape == (0, 3)
            assert payload["empty"].dtype == torch.float32
            assert payload["label"] == "step zero"
    finally:
        transport.close()


def _snapshot_and_backpressure(rank, group, signals, limit):
    release = threading.Event()
    wrapper = _DelayedSendGroup(group, release)
    payload_bytes = 256 * torch.float32.itemsize
    transport = PipelineTransport(
        wrapper,
        max_inflight_sends=1 if limit == "count" else 4,
        max_inflight_bytes=payload_bytes * (4 if limit == "count" else 1),
        async_receive=True,
    )
    try:
        if rank == 0:
            if limit == "bytes":
                with pytest.raises(ValueError):
                    transport.send({"hidden": torch.empty(257)})
                assert wrapper.posts == 0

            source = torch.arange(512, dtype=torch.float32).reshape(32, 16)
            transport.send({"hidden": source[:, ::2]})
            source.fill_(-1)
            assert transport.inflight_sends == 1
            assert transport.inflight_bytes == payload_bytes

            started = threading.Event()
            finished = threading.Event()
            errors = []

            def send_second():
                started.set()
                try:
                    transport.send({"hidden": torch.full((32, 8), 7.0)})
                except BaseException as error:
                    errors.append(error)
                finally:
                    finished.set()

            sender = threading.Thread(target=send_second, daemon=True)
            sender.start()
            assert started.wait(5)
            try:
                assert not finished.wait(0.2), "send exceeded its in-flight budget"
                assert wrapper.posts == 1
            finally:
                release.set()
                sender.join(timeout=10)
            assert not sender.is_alive()
            assert not errors, errors
            transport.drain()
            assert transport.inflight_sends == 0
            assert transport.inflight_bytes == 0
        else:
            first = transport.receive().wait()
            expected = torch.arange(512, dtype=torch.float32).reshape(32, 16)[:, ::2]
            torch.testing.assert_close(first["hidden"], expected)
            second = transport.receive().wait()
            torch.testing.assert_close(second["hidden"], torch.full((32, 8), 7.0))
    finally:
        release.set()
        transport.close()


def _peer_timeout(rank, group, signals):
    if rank == 0:
        # Remain alive without sending metadata: exercise the actual PG timeout.
        assert signals[0].wait(10), "receiver never reported the peer timeout"
        return

    failures = []
    failed = threading.Event()

    def on_fatal(error):
        failures.append(error)
        failed.set()

    transport = PipelineTransport(group, async_receive=True, on_fatal=on_fatal)
    ticket = transport.receive()
    try:
        # An idle consumer must still be notified when the background recv fails.
        assert failed.wait(10)
        with pytest.raises(PipelineTransportError):
            ticket.wait()
        with pytest.raises(PipelineTransportError):
            transport.check_error()
        with pytest.raises(PipelineTransportError):
            transport.send({})
        with pytest.raises(PipelineTransportError):
            transport.receive()
        with pytest.raises(PipelineTransportError):
            transport.close()
        assert len(failures) == 1
    finally:
        signals[0].set()


def _worker(rank, init_method, case, signals):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=2 if case == "timeout" else 15)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=2, timeout=timeout
    )
    group = None
    try:
        # Give both coordinator subgroups the same bounded timeout as WORLD.
        with (
            patch.dict("os.environ", {"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "0"}),
            patch(
                "vllm.distributed.utils.get_cpu_distributed_timeout_or_none",
                return_value=timeout,
            ),
            patch(
                "vllm.distributed.utils.get_distributed_timeout_or_none",
                return_value=timeout,
            ),
        ):
            group = GroupCoordinator(
                [[0, 1]],
                local_rank=rank,
                torch_distributed_backend="gloo",
                use_device_communicator=False,
                group_name="pp_transport_test",
            )
        if case == "metadata":
            _receive_before_metadata(rank, group, signals)
        elif case in ("count", "bytes"):
            _snapshot_and_backpressure(rank, group, signals, case)
        else:
            assert case == "timeout"
            _peer_timeout(rank, group, signals)
        # Keep the sender alive until the receiver has consumed its last work.
        dist.barrier()
    finally:
        if group is not None:
            group.destroy()
        dist.destroy_process_group()


def _run_two_ranks(tmp_path: Path, case: str):
    context = mp.get_context("spawn")
    signals = [context.Event()]
    init_method = (tmp_path / "gloo_init").as_uri()
    processes = [
        context.Process(target=_worker, args=(rank, init_method, case, signals))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 45
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                break
            if time.monotonic() >= deadline:
                pytest.fail(f"PP transport {case} workers exceeded 45 seconds")
            time.sleep(0.05)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=5)


def test_receive_returns_before_metadata_and_finishes_on_consumer(tmp_path):
    _run_two_ranks(tmp_path, "metadata")


@pytest.mark.parametrize("limit", ["count", "bytes"])
def test_send_preserves_snapshot_and_bounds_inflight_work(tmp_path, limit):
    _run_two_ranks(tmp_path, limit)


def test_peer_timeout_reaches_ticket_health_check_and_fatal_callback(tmp_path):
    _run_two_ranks(tmp_path, "timeout")


def test_failed_close_keeps_posted_buffers_when_completion_jobs_are_cancelled():
    class FailingWork:
        def wait(self):
            work_started.set()
            assert release_work.wait(10)
            raise OSError("peer disconnected during send")

    class PendingWork:
        def wait(self):
            unexpected_wait.set()

    release_work = threading.Event()
    work_started = threading.Event()
    release_callback = threading.Event()
    failed = threading.Event()
    unexpected_wait = threading.Event()
    snapshots = []

    def post(tensors, **kwargs):
        # The backend keeps an address, not a Python owner of the source tensor.
        snapshots.append(weakref.ref(tensors["hidden"]))
        return [FailingWork() if len(snapshots) == 1 else PendingWork()]

    def on_fatal(error):
        failed.set()
        # Hold the failing completion until close cancels the queued one.
        assert release_callback.wait(10)

    group = SimpleNamespace(
        world_size=2,
        device=torch.device("cpu"),
        use_cpu_custom_send_recv=False,
        isend_tensor_dict=post,
    )
    transport = PipelineTransport(group, on_fatal=on_fatal)
    try:
        transport.send({"hidden": torch.arange(4, dtype=torch.float32)})
        assert work_started.wait(5)
        transport.send({"hidden": torch.arange(4, dtype=torch.float32) + 10})
        release_work.set()
        assert failed.wait(5)
        with pytest.raises(PipelineTransportError):
            transport.close()
        gc.collect()
        assert not unexpected_wait.is_set()
        assert all(snapshot() is not None for snapshot in snapshots)
        torch.testing.assert_close(
            snapshots[1](), torch.arange(4, dtype=torch.float32) + 10
        )
        # Cancellation of a host completion job is not cancellation of P2P work.
        assert transport.inflight_sends == 2
        assert transport.inflight_bytes == 32
    finally:
        release_work.set()
        release_callback.set()
        with suppress(PipelineTransportError):
            transport.close()
