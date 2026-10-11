"""Home Assistant Cloud STT via the Assist pipeline websocket.

Streams normalized 16 kHz mono PCM to ``assist_pipeline/run`` (``start_stage`` =
``end_stage`` = ``stt``) on the configured Home Assistant instance, which
transcribes with its own STT engine (``stt.home_assistant_cloud`` on the
preferred pipeline). Facade-owned state (``_resolve_provider_key``,
``_load_stt_config``) is read lazily from ``tools.transcription_tools`` so this
module never creates an import cycle.

Wire framing (``websocket_api/http.py``): every binary frame is
``bytes([handler_id]) + payload`` where ``handler_id`` is the
``stt_binary_handler_id`` from the ``run-start`` event's ``runner_data``. The
stream ends by sending the bare handler byte (empty payload); a zero-length
frame would disconnect. The transcript arrives in the ``stt-end`` event at
``data.stt_output.text`` (older servers: ``data.stt_text``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tools.transcription_common import (
    DEFAULT_HA_CLOUD_PIPELINE, DEFAULT_HA_CLOUD_URL,
    _error_result, _get_stt_section, _log_prompt_unsupported, _ok_result)

# Log-record parity with the origin module.
logger = logging.getLogger("tools.transcription_tools")

HA_CLOUD_SAMPLE_RATE = 16000
HA_CLOUD_TOKEN_ENV = "HASS_TOKEN"
HA_CLOUD_URL_ENV = "HASS_URL"

_WS_CHUNK_BYTES = 8192
_WS_RUN_ID = 10
_WS_TIMEOUT_SECONDS = 90.0
_REACHABLE_TIMEOUT_SECONDS = 3.0


class HaCloudError(Exception):
    """A Home Assistant Cloud STT failure; always converted to the error envelope."""


# ---- Config / credentials -------------------------------------------------

def _ha_cloud_token() -> str:
    """Long-lived HA token via the shared secret resolver (secret scope > env > .env > pool)."""
    from tools.transcription_tools import _resolve_provider_key
    try:
        return _resolve_provider_key(HA_CLOUD_TOKEN_ENV, "ha_cloud")
    except Exception:  # pragma: no cover — resolver never raises, but stay total
        return ""


def _hass_url_secret() -> str:
    """Shared ``HASS_URL`` (same source as the Home Assistant tool), defaulting to LAN mDNS."""
    try:
        from agent.secret_scope import get_secret
        return (get_secret(HA_CLOUD_URL_ENV, DEFAULT_HA_CLOUD_URL)
                or DEFAULT_HA_CLOUD_URL).strip() or DEFAULT_HA_CLOUD_URL
    except Exception:  # multiplexed without a scope, or secret_scope unimportable
        return DEFAULT_HA_CLOUD_URL


def _ha_cloud_settings(stt_config: Dict[str, Any]) -> Tuple[str, str]:
    """``(ha_url, pipeline_id)`` from ``stt.ha_cloud``; ``HASS_URL`` fills a missing url."""
    cfg = _get_stt_section(stt_config, "ha_cloud")
    url = str(cfg.get("url") or "").strip() or _hass_url_secret()
    pipeline = str(cfg.get("pipeline") or "").strip() or DEFAULT_HA_CLOUD_PIPELINE
    return url.rstrip("/"), pipeline


def _ha_cloud_reachable(url: str, timeout: float = _REACHABLE_TIMEOUT_SECONDS) -> bool:
    """True when *url* answers any HTTP status (even 401 — that still proves HA is up)."""
    try:
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen
        try:
            with urlopen(Request(url.rstrip("/") + "/api/", method="GET"), timeout=timeout):
                return True
        except HTTPError:
            return True  # the server responded; auth state is irrelevant to reachability
        except (URLError, OSError, ValueError):
            return False
    except Exception:  # pragma: no cover — defensive; reachability must never raise
        return False


def _has_ha_cloud() -> bool:
    """Availability probe: a token is configured AND the HA instance answers."""
    if not _ha_cloud_token():
        return False
    try:
        from tools.transcription_tools import _load_stt_config
        url, _ = _ha_cloud_settings(_load_stt_config())
    except Exception:
        return False
    return _ha_cloud_reachable(url)


# ---- Audio normalization ----------------------------------------------------

def _normalize_to_16k_mono_pcm(file_path: str) -> bytes:
    """Decode *file_path* to 16 kHz mono signed-16-bit PCM bytes.

    ffmpeg handles every container phone uploads arrive in (wav/mp3/ogg/opus);
    without it, only an already-16k-mono-16-bit WAV can be passed through via
    the stdlib ``wave`` module. Raises :class:`HaCloudError` on failure.
    """
    from tools.transcription_audio import _find_ffmpeg_binary
    name = Path(file_path).name
    ffmpeg = _find_ffmpeg_binary()
    if ffmpeg:
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", file_path,
                 "-ac", "1", "-ar", str(HA_CLOUD_SAMPLE_RATE),
                 "-f", "s16le", "-acodec", "pcm_s16le", "-"],
                check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip() or str(exc)
            raise HaCloudError(f"ffmpeg could not decode {name} for HA Cloud STT: {detail}")
        except subprocess.TimeoutExpired:
            raise HaCloudError(f"ffmpeg decode timed out for {name}")
        if not proc.stdout:
            raise HaCloudError(f"ffmpeg produced no audio for {name}")
        return proc.stdout
    if Path(file_path).suffix.lower() != ".wav":
        raise HaCloudError(
            f"ffmpeg not found: HA Cloud STT needs ffmpeg to decode {name} "
            "(only 16 kHz mono 16-bit WAV works without it)")
    try:
        with wave.open(file_path, "rb") as wf:
            if (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) != (1, 2, HA_CLOUD_SAMPLE_RATE):
                raise HaCloudError(
                    f"{name} is not 16 kHz mono 16-bit audio and ffmpeg is not "
                    "installed to resample it")
            return wf.readframes(wf.getnframes())
    except HaCloudError:
        raise
    except Exception as exc:
        raise HaCloudError(f"could not read {name} as WAV: {exc}")


# ---- Wire framing (pure; unit-tested without a network) ---------------------

def _frame_audio_chunk(handler_id: int, chunk: bytes) -> bytes:
    """One binary WS frame: handler byte + PCM payload."""
    return bytes([handler_id]) + chunk


def _termination_frame(handler_id: int) -> bytes:
    """End-of-stream marker: the bare handler byte (empty payload)."""
    return bytes([handler_id])


def _extract_stt_transcript(event_data: Dict[str, Any]) -> str:
    """Transcript from an ``stt-end`` event: ``data.stt_output.text`` (``data.stt_text`` legacy)."""
    data = event_data or {}
    output = data.get("stt_output")
    if isinstance(output, dict) and isinstance(output.get("text"), str):
        return output["text"]
    legacy = data.get("stt_text")
    return legacy if isinstance(legacy, str) else ""


def _extract_handler_id(event_data: Dict[str, Any]) -> int:
    """``stt_binary_handler_id`` from a ``run-start`` event; raises when absent/invalid."""
    data = event_data or {}
    handler = (data.get("runner_data") or {}).get("stt_binary_handler_id")
    if not isinstance(handler, int) or not 0 <= handler <= 255:
        raise HaCloudError(
            "Home Assistant run-start carried no stt_binary_handler_id for the audio stream")
    return handler


def _ha_ws_url(http_url: str) -> str:
    """``http(s)://host:port`` -> ``ws(s)://host:port/api/websocket``; raises on garbage."""
    from urllib.parse import urlparse
    parts = urlparse(http_url)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme.lower())
    if not scheme or not parts.hostname:
        raise HaCloudError(f"invalid Home Assistant URL: {http_url!r}")
    return f"{scheme}://{parts.hostname}{f':{parts.port}' if parts.port else ''}/api/websocket"


# ---- Assist pipeline exchange -------------------------------------------------

async def _stt_exchange(
    pcm: bytes, ws_url: str, token: str, pipeline: str,
    *, msg_id: int = _WS_RUN_ID, timeout: float = _WS_TIMEOUT_SECONDS,
) -> str:
    """Run one STT-only assist pipeline and return the transcript (\"\" when none)."""
    try:
        import websockets
    except ImportError:
        raise HaCloudError("websockets package not installed (HA Cloud STT needs it)")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    def _time_left() -> float:
        return max(deadline - loop.time(), 0.1)

    try:
        async with websockets.connect(ws_url, max_size=4 * 1024 * 1024) as ws:
            await asyncio.wait_for(ws.recv(), _time_left())  # auth_required greeting
            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), _time_left()))
            if auth.get("type") != "auth_ok":
                raise HaCloudError(
                    f"Home Assistant auth failed ({auth.get('type')}: "
                    f"{auth.get('message') or 'check HASS_TOKEN'})")
            await ws.send(json.dumps({
                "id": msg_id, "type": "assist_pipeline/run",
                "start_stage": "stt", "end_stage": "stt",
                "input": {"sample_rate": HA_CLOUD_SAMPLE_RATE},
                "pipeline": pipeline,
            }))
            handler: Optional[int] = None
            sent = False
            while True:
                raw = await asyncio.wait_for(ws.recv(), _time_left())
                if isinstance(raw, bytes):
                    continue  # no binary traffic is expected server -> client here
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict) or msg.get("id") != msg_id:
                    continue
                if msg.get("type") == "result" and not msg.get("success", True):
                    raise HaCloudError(
                        f"Home Assistant pipeline error: {msg.get('error')}")
                event = msg.get("event") or {}
                etype = event.get("type")
                data = event.get("data") or {}
                if etype == "run-start" and handler is None:
                    handler = _extract_handler_id(data)
                elif etype == "stt-start" and not sent:
                    if handler is None:
                        raise HaCloudError(
                            "Home Assistant sent stt-start before run-start (no handler id)")
                    for i in range(0, len(pcm), _WS_CHUNK_BYTES):
                        await ws.send(_frame_audio_chunk(handler, pcm[i:i + _WS_CHUNK_BYTES]))
                    await ws.send(_termination_frame(handler))  # bare handler byte ends the stream
                    sent = True
                elif etype == "stt-end":
                    return _extract_stt_transcript(data)
                elif etype == "error":
                    raise HaCloudError(f"Home Assistant STT error: {data}")
                elif etype == "run-end":
                    raise HaCloudError("Home Assistant pipeline run ended with no transcript")
    except HaCloudError:
        raise
    except asyncio.TimeoutError:
        raise HaCloudError("Home Assistant STT timed out waiting for the pipeline")
    except Exception as exc:
        raise HaCloudError(f"Home Assistant STT websocket failed: {exc}")


def _run_ha_stt(pcm: bytes, url: str, token: str, pipeline: str) -> str:
    """Synchronous wrapper for :func:`_stt_exchange` (isolated loop when nested)."""
    ws_url = _ha_ws_url(url)
    try:
        return asyncio.run(_stt_exchange(pcm, ws_url, token, pipeline))
    except RuntimeError:
        # Already inside a running loop (embedded gateway thread): isolate ours in a worker.
        outcome: Dict[str, Any] = {}

        def _worker() -> None:
            try:
                outcome["result"] = asyncio.run(_stt_exchange(pcm, ws_url, token, pipeline))
            except Exception as exc:  # propagate verbatim to the caller below
                outcome["error"] = exc

        thread = threading.Thread(target=_worker, name="ha-cloud-stt", daemon=True)
        thread.start()
        thread.join(_WS_TIMEOUT_SECONDS + 30)
        if "result" in outcome:
            return outcome["result"]
        if "error" in outcome:
            raise outcome["error"]
        raise HaCloudError("Home Assistant STT worker timed out")


# ---- Provider entry point -----------------------------------------------------

def _transcribe_ha_cloud(
    file_path: str, model_name: str, *, language: Optional[str] = None, prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe via Home Assistant Cloud STT. ``model_name`` carries the resolved
    pipeline id (``stt.ha_cloud.pipeline``); every failure returns the error
    envelope so the caller can fall back to local — nothing is swallowed."""
    if prompt:
        _log_prompt_unsupported("STT provider 'ha_cloud'")
    from tools.transcription_tools import _load_stt_config
    stt_config = _load_stt_config()
    url, configured_pipeline = _ha_cloud_settings(stt_config if isinstance(stt_config, dict) else {})
    # The dispatcher resolves ``model_name`` from ``stt.ha_cloud.pipeline`` already;
    # a hook/caller override still wins here.
    pipeline = (model_name or "").strip() or configured_pipeline
    token = _ha_cloud_token()
    if not token:
        return _error_result(
            "HASS_TOKEN not set: HA Cloud STT needs a Home Assistant long-lived "
            "access token (set HASS_TOKEN via `hermes auth add` or the environment)",
            provider="ha_cloud")
    try:
        pcm = _normalize_to_16k_mono_pcm(file_path)
    except HaCloudError as exc:
        return _error_result(str(exc), provider="ha_cloud")
    except Exception as exc:  # noqa: BLE001 — normalization must never raise out
        logger.error("HA Cloud STT audio prep failed for %s: %s", file_path, exc, exc_info=True)
        return _error_result(f"HA Cloud STT audio prep failed: {exc}", provider="ha_cloud")
    try:
        transcript = _run_ha_stt(pcm, url, token, pipeline)
    except HaCloudError as exc:
        return _error_result(str(exc), provider="ha_cloud")
    except Exception as exc:  # noqa: BLE001 — transport failures are envelope errors
        logger.error("HA Cloud STT transcription failed for %s: %s", file_path, exc, exc_info=True)
        return _error_result(f"HA Cloud STT transcription failed: {exc}", provider="ha_cloud")
    if not transcript:
        return _error_result("HA Cloud STT returned an empty transcript",
                             provider="ha_cloud", no_speech=True)
    logger.info("Transcribed %s via HA Cloud STT (pipeline %s, %d chars)",
                Path(file_path).name, pipeline, len(transcript))
    return _ok_result(transcript, "ha_cloud")
