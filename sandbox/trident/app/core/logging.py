import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class StageLogger:
    def __init__(self, logger: logging.Logger, stage: str, request_id: str = ""):
        self._logger = logger
        self._stage = stage
        self._request_id = request_id or request_id_var.get()
        self._start: float = 0.0

    def start(self, context: dict[str, Any] | None = None) -> None:
        self._start = time.monotonic()
        extra = {"stage": self._stage, "event": "start", "request_id": self._request_id}
        if context:
            extra.update(context)
        self._logger.info("Stage %s: start", self._stage, extra=extra)

    def end(self, context: dict[str, Any] | None = None) -> None:
        elapsed = time.monotonic() - self._start
        extra = {
            "stage": self._stage,
            "event": "end",
            "latency_ms": round(elapsed * 1000, 1),
            "request_id": self._request_id,
        }
        if context:
            extra.update(context)
        self._logger.info("Stage %s: end (%.1fms)", self._stage, elapsed * 1000, extra=extra)

    def log(self, msg: str, context: dict[str, Any] | None = None) -> None:
        extra = {"stage": self._stage, "request_id": self._request_id}
        if context:
            extra.update(context)
        self._logger.info("Stage %s: %s", self._stage, msg, extra=extra)


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid
