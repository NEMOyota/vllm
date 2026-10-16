# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, owned transfers for the native pipeline-parallel worker path."""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Condition, Lock
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from vllm.distributed.parallel_state import GroupCoordinator, Handle


class PipelineTransportError(RuntimeError):
    """A failed pipeline channel must be torn down before it is reused."""


class ReceiveTicket:
    """One FIFO receive, consumed on the model execution thread."""

    def __init__(self, transport: "PipelineTransport") -> None:
        self._transport = transport
        self._future: Future[Any] = Future()
        self._lock = Lock()
        self._result: dict[str, Any] | None = None
        self._buffers: list[torch.Tensor] = []

    def wait(self) -> dict[str, Any]:
        with self._lock:
            self._transport.check_error()
            if self._result is not None:
                return self._result
            try:
                tensors, handles, postprocess = self._future.result()
                for handle in handles:
                    handle.wait()
                # TP collectives must retain the execution thread's ordering.
                for fn in postprocess:
                    fn()
                # TP gather replaces slice views in `tensors`; its source
                # allocations also remain live on the consumer stream.
                for tensor in [*self._buffers, *tensors.values()]:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        tensor.record_stream(torch.cuda.current_stream(tensor.device))
            except Exception as exc:
                self._transport._fail(exc)
                raise PipelineTransportError("Pipeline receive failed") from exc
            self._result = tensors
            self._transport._consume_receive(self)
            return tensors


class PipelineTransport:
    """Snapshot sends and overlap receive metadata with model input preparation.

    Only the worker execution thread submits/consumes frames. Background owners
    receive metadata and retire sends; they never execute TP collectives. Failed
    transfers retain their buffers until worker teardown, not a presumed cancel.
    """

    def __init__(
        self,
        group: "GroupCoordinator",
        *,
        max_inflight_sends: int = 2,
        max_inflight_bytes: int = 268435456,
        async_receive: bool = True,
        on_fatal: "Callable[[BaseException], None] | None" = None,
    ) -> None:
        if max_inflight_sends < 1 or max_inflight_bytes < 1:
            raise ValueError("Pipeline transfer limits must be positive")
        if group.world_size <= 1:
            raise ValueError("Pipeline transport requires multiple PP ranks")
        if group.use_cpu_custom_send_recv:
            raise ValueError("Pipeline transport requires asynchronous tensor P2P")
        self._group = group
        self._device = torch.device(group.device)
        if self._device.type not in ("cpu", "cuda"):
            raise ValueError("Pipeline transport supports CPU/Gloo and CUDA P2P")
        self._max_sends = max_inflight_sends
        self._max_bytes = max_inflight_bytes
        self._async_receive = async_receive
        self._on_fatal = on_fatal
        self._condition = Condition()
        self._error: BaseException | None = None
        self._closed = False
        self._inflight_sends = 0
        self._inflight_bytes = 0
        self._receive_ticket: ReceiveTicket | None = None
        # Keep ownership independently of executor futures: failed shutdown can
        # cancel queued completion jobs while their P2P work is still in flight.
        self._send_buffers: dict[int, tuple[dict[str, Any], list[Handle]]] = {}
        self._sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pp-send")
        self._receiver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pp-recv")
        self._send_stream = None
        self._recv_stream = None
        self._completion_stream = None
        if self._device.type == "cuda":
            with torch.accelerator.device_index(self._device.index):
                self._send_stream = torch.cuda.Stream(self._device)
                self._recv_stream = torch.cuda.Stream(self._device)
                self._completion_stream = torch.cuda.Stream(self._device)

    @property
    def inflight_sends(self) -> int:
        with self._condition:
            return self._inflight_sends

    @property
    def inflight_bytes(self) -> int:
        with self._condition:
            return self._inflight_bytes

    def check_error(self) -> None:
        with self._condition:
            if self._error is not None:
                raise PipelineTransportError(
                    "Pipeline transport failed"
                ) from self._error

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            first = self._error is None
            if first:
                self._error = error
            self._condition.notify_all()
        if first and self._on_fatal is not None:
            self._on_fatal(error)

    @contextmanager
    def _stream_context(self, stream: Any) -> "Iterator[None]":
        if self._device.type == "cuda":
            with (
                torch.accelerator.device_index(self._device.index),
                torch.cuda.stream(stream),
            ):
                yield
        else:
            yield

    def send(
        self,
        tensors: dict[str, Any],
        *,
        all_gather_group: "GroupCoordinator | None" = None,
    ) -> None:
        self.check_error()
        size = sum(
            t.numel() * t.element_size()
            for t in tensors.values()
            if isinstance(t, torch.Tensor)
        )
        if size > self._max_bytes:
            raise ValueError(
                f"Pipeline frame ({size} bytes) exceeds the in-flight byte limit "
                f"({self._max_bytes}); reduce the batch/chunk size or increase "
                "max_inflight_bytes"
            )
        with self._condition:
            self.check_error()
            if self._closed:
                raise RuntimeError("Pipeline transport is closed")
            while (
                self._inflight_sends >= self._max_sends
                or self._inflight_bytes + size > self._max_bytes
            ):
                self._condition.wait()
                self.check_error()
                if self._closed:
                    raise RuntimeError("Pipeline transport is closed")
            self._inflight_sends += 1
            self._inflight_bytes += size

        snapshot: dict[str, Any] = {}
        handles: list[Handle] = []
        try:
            # These copies precede the next forward on the producer stream.
            snapshot = {
                key: value.detach().clone(memory_format=torch.contiguous_format)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in tensors.items()
            }
            with self._condition:
                self._send_buffers[id(snapshot)] = (snapshot, handles)
            ready = None
            if self._send_stream is not None:
                ready = torch.cuda.current_stream(self._device).record_event()
            with self._stream_context(self._send_stream):
                if ready is not None:
                    assert self._send_stream is not None
                    self._send_stream.wait_event(ready)
                self.check_error()
                handles = self._group.isend_tensor_dict(
                    snapshot,
                    all_gather_group=all_gather_group,
                    retain_tensors=False,
                )
            with self._condition:
                self._send_buffers[id(snapshot)] = (snapshot, handles)
            self._sender.submit(self._finish_send, snapshot, handles, size)
        except Exception as exc:
            # Submission can fail after publishing some work. Do not refund
            # credit or release a potentially still-readable source allocation.
            with self._condition:
                self._send_buffers[id(snapshot)] = (snapshot, handles)
            self._fail(exc)
            self.check_error()

    def _finish_send(
        self, snapshot: dict[str, Any], handles: "list[Handle]", size: int
    ) -> None:
        try:
            with self._stream_context(self._completion_stream):
                for handle in handles:
                    handle.wait()
                # NCCL Work.wait establishes stream order, not host completion.
                # A separate stream avoids waiting for subsequently queued sends.
                if self._completion_stream is not None:
                    self._completion_stream.record_event().synchronize()
        except Exception as exc:
            self._fail(exc)
            return
        with self._condition:
            del self._send_buffers[id(snapshot)]
            self._inflight_sends -= 1
            self._inflight_bytes -= size
            self._condition.notify_all()

    def receive(
        self, *, all_gather_group: "GroupCoordinator | None" = None
    ) -> ReceiveTicket:
        with self._condition:
            self.check_error()
            if self._closed:
                raise RuntimeError("Pipeline transport is closed")
            if self._receive_ticket is not None:
                raise RuntimeError("Consume the previous pipeline receive first")
            ticket = ReceiveTicket(self)
            self._receive_ticket = ticket
        if self._async_receive:
            self._receiver.submit(self._prepare_receive, ticket, all_gather_group)
        else:
            self._prepare_receive(ticket, all_gather_group)
        return ticket

    def _prepare_receive(
        self, ticket: ReceiveTicket, all_gather_group: "GroupCoordinator | None"
    ) -> None:
        try:
            with self._stream_context(self._recv_stream):
                result = self._group.irecv_tensor_dict(
                    all_gather_group=all_gather_group,
                    max_recv_bytes=self._max_bytes,
                    receive_buffer_refs=ticket._buffers,
                )
            ticket._future.set_result(result)
        except Exception as exc:
            ticket._future.set_exception(exc)
            self._fail(exc)

    def _consume_receive(self, ticket: ReceiveTicket) -> None:
        with self._condition:
            assert self._receive_ticket is ticket
            self._receive_ticket = None

    def drain(self) -> None:
        self.check_error()
        if self._receive_ticket is not None:
            self._receive_ticket.wait()
        with self._condition:
            while self._inflight_sends:
                self._condition.wait()
                self.check_error()
            self.check_error()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                self.check_error()
                return
            self._closed = True
            self._condition.notify_all()
        try:
            self.drain()
        except Exception:
            self._sender.shutdown(wait=False, cancel_futures=True)
            self._receiver.shutdown(wait=False, cancel_futures=True)
            raise
        self._sender.shutdown(wait=True)
        self._receiver.shutdown(wait=True)
