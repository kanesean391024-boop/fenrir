import json
import os
import re
import threading
import time
from collections import deque

from typing import Dict, Optional, List, Tuple, Callable
import ctypes

import logging
import sys
import requests


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
GLOBAL_LOG_LEVEL = logging.INFO


logger = logging.getLogger(__name__)

# Registered callbacks to inspect JSON payloads sent to Ollama
JSON_WATCHERS: List[Callable[[Dict], None]] = []

_WORDLIKE_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)

from functools import lru_cache


@lru_cache(maxsize=256)
def _tokenize_cached(model: str, snippet: str) -> List[str]:
    return _WORDLIKE_RE.findall(snippet)


def tokenize_text(model: str, text: str) -> List[str]:
    """Very lightweight tokenization used only for budgeting/tracking.

    Returns a list so callers can ``len(...)`` it. This heuristic tokenizer is
    dependency-free and does not reflect model-specific tokenization. A small
    LRU cache avoids repeated tokenization of the same prefix.
    """

    if not text:
        return []
    snippet = text[:4096]
    try:
        tokens = _tokenize_cached(model, snippet)
        if len(text) > 4096:
            tokens += _WORDLIKE_RE.findall(text[4096:])
        return tokens
    except Exception:
        # Fallback: whitespace split so callers still get a length
        try:
            return (text or "").split()
        except Exception:
            return []


def add_json_watcher(func: Callable[[Dict], None]) -> None:
    """Register a callback for outgoing Ollama payloads."""
    JSON_WATCHERS.append(func)


def parse_log_level(level_name: str) -> int:
    """Return a logging level constant from a name."""
    logger.debug("Entering parse_log_level level_name=%s", level_name)
    result = getattr(logging, level_name.upper(), logging.INFO)
    logger.debug("Exiting parse_log_level")
    return result


def init_global_logging(level: int) -> None:
    """Initialize root logging and store the global level."""
    logger.debug("Entering init_global_logging level=%s", level)
    global GLOBAL_LOG_LEVEL
    GLOBAL_LOG_LEVEL = level
    os.makedirs("logs", exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "fenrir.log"), mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)
    logger.debug("Exiting init_global_logging")


def create_object_logger(class_name: str) -> logging.Logger:
    """Return a logger that propagates to the global handlers."""
    # Remove characters that could result in invalid names.
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "", class_name)

    logger = logging.getLogger(safe_name)
    logger.handlers.clear()
    logger.setLevel(GLOBAL_LOG_LEVEL)
    logger.propagate = True
    return logger


class TransientModelError(Exception):
    """Raised when the model returns a transient server error."""


class AITimeTracker:
    """Track recent successful generation times in AI Seconds."""

    def __init__(self, rolling_window: int = 10) -> None:
        logger.debug("Entering AITimeTracker.__init__ rolling_window=%s", rolling_window)
        self.data = deque(maxlen=rolling_window)
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized AITimeTracker")
        logger.debug("Exiting AITimeTracker.__init__")

    def record(self, wall_time: float, model_size_gb: float) -> None:
        self.logger.debug(
            "Entering AITimeTracker.record wall_time=%s model_size_gb=%s",
            wall_time,
            model_size_gb,
        )
        """Convert wall time to AI Seconds (normalized to 7GB) and store."""
        factor = (model_size_gb / 7) ** 1.2
        ai_seconds = wall_time / factor
        self.data.append(ai_seconds)
        self.logger.debug(
            "Recorded %.2fs wall time (%.2fGB) -> %.2f AI seconds",
            wall_time,
            model_size_gb,
            ai_seconds,
        )
        self.logger.debug("Exiting AITimeTracker.record")

    def average(self) -> float:
        self.logger.debug("Entering AITimeTracker.average")
        """Return the rolling average AI Seconds with a 300s baseline."""
        total = sum(self.data) + 300
        count = len(self.data) + 1
        avg = total / count
        self.logger.debug("Average AI seconds: %.2f", avg)
        self.logger.debug("Exiting AITimeTracker.average")
        return avg


class ThreadTimeTracker:
    """Manage per-thread time trackers and compute a global average."""

    def __init__(self, rolling_window: int = 10) -> None:
        logger.debug(
            "Entering ThreadTimeTracker.__init__ rolling_window=%s", rolling_window
        )
        self.trackers: Dict[int, AITimeTracker] = {}
        self.rolling_window = rolling_window
        self.timeout_extra = 0.0
        self.logger = create_object_logger(self.__class__.__name__)
        self.logger.info("Initialized ThreadTimeTracker")
        logger.debug("Exiting ThreadTimeTracker.__init__")

    def _tracker_for(self, tid: int) -> AITimeTracker:
        self.logger.debug("Entering ThreadTimeTracker._tracker_for tid=%s", tid)
        if tid not in self.trackers:
            self.trackers[tid] = AITimeTracker(self.rolling_window)
        tracker = self.trackers[tid]
        self.logger.debug("Exiting ThreadTimeTracker._tracker_for")
        return tracker

    def record(self, tid: int, wall_time: float, model_size_gb: float) -> None:
        self.logger.debug(
            "Entering ThreadTimeTracker.record tid=%s wall_time=%s model_size_gb=%s",
            tid,
            wall_time,
            model_size_gb,
        )
        self._tracker_for(tid).record(wall_time, model_size_gb)
        self.logger.debug("Exiting ThreadTimeTracker.record")

    def _total_and_count(self) -> Tuple[float, int]:
        """Return the aggregated total time and count across trackers."""
        total = 300 + self.timeout_extra
        count = 1
        for tracker in self.trackers.values():
            total += sum(tracker.data)
            count += len(tracker.data)
        return total, count

    def average(self) -> float:
        self.logger.debug("Entering ThreadTimeTracker.average")
        total, count = self._total_and_count()
        avg = total / count
        self.logger.debug("Global average AI seconds: %.2f", avg)
        self.logger.debug("Exiting ThreadTimeTracker.average")
        return avg

    def record_timeout(self) -> None:
        """Increase total time so the average grows by 5%."""
        self.logger.debug("Entering ThreadTimeTracker.record_timeout")
        total, _ = self._total_and_count()
        extra = total * 0.05
        self.timeout_extra += extra
        self.logger.debug(
            "Added %.2f seconds due to timeout. New extra total %.2f", extra, self.timeout_extra
        )
        self.logger.debug("Exiting ThreadTimeTracker.record_timeout")


WATCHDOG_TRACKER = ThreadTimeTracker()


def parse_model_size(model_id: str, max_retries: int = 3) -> float:
    """Return the disk size of the model in gigabytes.

    If the ``ollama list`` command fails, retry up to ``max_retries`` times
    before falling back to a default size of ``7.0`` gigabytes.
    """
    logger = logging.getLogger("ModelSize")
    logger.debug("Entering parse_model_size model_id=%s", model_id)
    import subprocess
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                ["ollama", "list", model_id],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.splitlines()
            for line in lines:
                if model_id in line:
                    match = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)", line, re.IGNORECASE)
                    if not match:
                        break
                    value = float(match.group(1))
                    unit = match.group(2).upper()
                    if unit == "KB":
                        return value / (1024 * 1024)
                    if unit == "MB":
                        return value / 1024
                    if unit == "GB":
                        return value
                    if unit == "TB":
                        return value * 1024
            logger.error(
                "Failed to parse model size from ollama output for %s", model_id
            )
        except FileNotFoundError:
            logger.error("ollama executable not found")
            break
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Error running ollama list for %s (attempt %d/%d): %s",
                model_id,
                attempt,
                max_retries,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error obtaining size for %s (attempt %d/%d): %s",
                model_id,
                attempt,
                max_retries,
                exc,
            )

        if attempt < max_retries:
            logger.info(
                "Retrying model size retrieval for %s (%d/%d)",
                model_id,
                attempt + 1,
                max_retries,
            )
            time.sleep(2)

    logger.error(
        "Failed to obtain model size for %s after %d attempts; defaulting to 7.0GB",
        model_id,
        max_retries,
    )
    logger.debug("Exiting parse_model_size")
    return 7.0
def parse_response(resp: requests.Response) -> str:
    """Parse a non-streaming Ollama response into text."""
    logger.debug("Entering parse_response")
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from Ollama") from exc
    logger.debug("Exiting parse_response")
    return data.get("response", "")


def strip_think_markup(text: str) -> str:
    """Return text with any <think>...</think> sections removed."""
    logger.debug("Entering strip_think_markup")
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    logger.debug("Exiting strip_think_markup")
    return result


def kill_thread(thread: threading.Thread) -> None:
    """Forcefully stop a thread by raising SystemExit within it."""
    logger.debug("Entering kill_thread thread=%s", thread)
    ident: Optional[int] = thread.ident
    if ident is None:
        logger.debug("Exiting kill_thread: no ident")
        return
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(ident), ctypes.py_object(SystemExit)
    )
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    logger.debug("Exiting kill_thread")


def generate_with_watchdog(
    payload: Dict,
    *,
    base_timeout: float | None = 900,
    agent_name: str | None = None,
) -> str:
    """Call Ollama. If base_timeout <= 0 or None, do not pass a requests timeout."""
    logger.debug(
        "Entering generate_with_watchdog base_timeout=%s agent_name=%s",
        base_timeout,
        agent_name,
    )

    model_id = payload.get("model", "unknown")
    wd_logger = create_object_logger(f"Watchdog-{model_id}")
    disabled = base_timeout is None or base_timeout <= 0
    if agent_name:
        wd_logger.info(
            "Starting generation for %s with watchdog=%s",
            agent_name,
            "disabled" if disabled else f"{base_timeout}s",
        )
    else:
        wd_logger.info(
            "Starting generation with watchdog=%s",
            "disabled" if disabled else f"{base_timeout}s",
        )

    start = time.time()
    timeout = None if disabled else float(base_timeout)
    wd_logger.debug("Timeout effective=%s", "disabled" if disabled else f"{timeout:.2f}s")
    try:
        if wd_logger.isEnabledFor(logging.DEBUG):
            wd_logger.debug("Payload to Ollama:\n%s", json.dumps(payload, indent=2))
        # Notify UI/watchers with the exact payload that is about to be sent
        payload_for_watchers = dict(payload)
        if agent_name:
            payload_for_watchers["__agent"] = agent_name
        for func in JSON_WATCHERS:
            try:
                func(payload_for_watchers)
            except Exception:
                pass
        kwargs = {"json": payload}
        if timeout is not None:
            kwargs["timeout"] = timeout
        resp = requests.post("http://localhost:11434/api/generate", **kwargs)
        if resp.status_code >= 500:
            raise TransientModelError(
                f"Ollama API error: {resp.status_code} {resp.text}"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama API error: {resp.status_code} {resp.text}"
            )
        text = parse_response(resp)
        wd_logger.info("Generation complete")
        logger.debug("Exiting generate_with_watchdog")
        return str(text)
    except requests.Timeout as exc:
        wd_logger.error("Timeout exceeded")
        raise exc
    except Exception as exc:  # noqa: BLE001
        wd_logger.error("Exception during generation: %s", exc)
        raise
