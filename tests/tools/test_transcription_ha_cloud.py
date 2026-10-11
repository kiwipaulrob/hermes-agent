"""Tests for the Home Assistant Cloud STT provider (tools/transcription_ha_cloud.py).

Covers the pure wire-framing helpers (handler-byte framing, bare-byte
termination, transcript/handler extraction, ws URL building), the
availability probe, the error-envelope contract (every failure returns an
error dict so callers can fall back to local), dispatcher registration, and
one full ``assist_pipeline/run`` exchange against a mocked websocket layer.
All network access is mocked.
"""

import asyncio
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from tools.transcription_ha_cloud import HaCloudError

if "faster_whisper" not in sys.modules:
    faster_whisper_stub = types.ModuleType("faster_whisper")
    faster_whisper_stub.WhisperModel = MagicMock(name="WhisperModel")
    from importlib.machinery import ModuleSpec
    faster_whisper_stub.__spec__ = ModuleSpec("faster_whisper", loader=None)
    sys.modules["faster_whisper"] = faster_whisper_stub

pytestmark = pytest.mark.usefixtures("disable_lazy_stt_install")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real credentials leak into tests."""
    monkeypatch.delenv("HASS_TOKEN", raising=False)
    monkeypatch.delenv("HASS_URL", raising=False)


# ============================================================================
# Wire framing (pure helpers)
# ============================================================================

class TestFraming:
    def test_audio_chunk_is_handler_byte_plus_payload(self):
        from tools.transcription_ha_cloud import _frame_audio_chunk
        assert _frame_audio_chunk(7, b"\x01\x02") == b"\x07\x01\x02"

    def test_termination_is_bare_handler_byte(self):
        """End-of-stream is the bare handler byte (a zero-length frame would disconnect)."""
        from tools.transcription_ha_cloud import _termination_frame
        frame = _termination_frame(3)
        assert frame == b"\x03"
        assert len(frame) == 1

    def test_handler_id_boundaries(self):
        from tools.transcription_ha_cloud import _frame_audio_chunk, _termination_frame
        assert _frame_audio_chunk(0, b"") == b"\x00"
        assert _termination_frame(255) == b"\xff"


class TestExtractTranscript:
    def test_current_stt_output_shape(self):
        from tools.transcription_ha_cloud import _extract_stt_transcript
        assert _extract_stt_transcript({"stt_output": {"text": "hello home"}}) == "hello home"

    def test_legacy_stt_text_shape(self):
        from tools.transcription_ha_cloud import _extract_stt_transcript
        assert _extract_stt_transcript({"stt_text": "legacy words"}) == "legacy words"

    def test_missing_transcript_is_empty(self):
        from tools.transcription_ha_cloud import _extract_stt_transcript
        assert _extract_stt_transcript({}) == ""
        assert _extract_stt_transcript({"stt_output": {}}) == ""
        assert _extract_stt_transcript({"stt_output": {"text": None}}) == ""


class TestExtractHandlerId:
    def test_run_start_handler_id(self):
        from tools.transcription_ha_cloud import _extract_handler_id
        data = {"runner_data": {"stt_binary_handler_id": 5}}
        assert _extract_handler_id(data) == 5

    def test_missing_handler_id_raises(self):
        from tools.transcription_ha_cloud import HaCloudError, _extract_handler_id
        with pytest.raises(HaCloudError):
            _extract_handler_id({})
        with pytest.raises(HaCloudError):
            _extract_handler_id({"runner_data": {}})


class TestWsUrl:
    def test_http_to_ws(self):
        from tools.transcription_ha_cloud import _ha_ws_url
        assert _ha_ws_url("http://192.168.1.10:8123") == "ws://192.168.1.10:8123/api/websocket"

    def test_https_to_wss(self):
        from tools.transcription_ha_cloud import _ha_ws_url
        assert _ha_ws_url("https://home.example.com") == "wss://home.example.com/api/websocket"

    def test_garbage_url_raises(self):
        from tools.transcription_ha_cloud import HaCloudError, _ha_ws_url
        with pytest.raises(HaCloudError):
            _ha_ws_url("not-a-url")
        with pytest.raises(HaCloudError):
            _ha_ws_url("ftp://host/x")


# ============================================================================
# Settings / availability probe
# ============================================================================

class TestSettings:
    def test_defaults(self):
        from tools.transcription_ha_cloud import (
            DEFAULT_HA_CLOUD_PIPELINE, _ha_cloud_settings)
        url, pipeline = _ha_cloud_settings({})
        assert url == "http://homeassistant.local:8123"
        assert pipeline == DEFAULT_HA_CLOUD_PIPELINE

    def test_config_overrides(self):
        from tools.transcription_ha_cloud import _ha_cloud_settings
        url, pipeline = _ha_cloud_settings(
            {"ha_cloud": {"url": "http://ha.lan:8123/", "pipeline": "deadbeef"}})
        assert url == "http://ha.lan:8123"
        assert pipeline == "deadbeef"


class TestHasHaCloud:
    def test_no_token_is_unavailable(self):
        from tools.transcription_ha_cloud import _has_ha_cloud
        with patch("tools.transcription_tools._resolve_provider_key", return_value=""):
            assert _has_ha_cloud() is False

    def test_unreachable_is_unavailable(self, monkeypatch):
        from tools.transcription_ha_cloud import _has_ha_cloud
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_ha_cloud._ha_cloud_reachable", return_value=False):
            assert _has_ha_cloud() is False

    def test_token_plus_reachable_is_available(self, monkeypatch):
        from tools.transcription_ha_cloud import _has_ha_cloud
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_ha_cloud._ha_cloud_reachable", return_value=True):
            assert _has_ha_cloud() is True


# ============================================================================
# Provider entry point: every failure is an error dict (callers fall back)
# ============================================================================

class TestTranscribeHaCloudEnvelope:
    def test_missing_token_is_error_dict(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 16)
        with patch("tools.transcription_tools._resolve_provider_key", return_value=""), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            from tools.transcription_tools import _transcribe_ha_cloud
            result = _transcribe_ha_cloud(str(audio), "")
        assert result["success"] is False
        assert result["provider"] == "ha_cloud"
        assert "HASS_TOKEN" in result["error"]

    def test_audio_prep_failure_is_error_dict(self, tmp_path, monkeypatch):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 16)
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_ha_cloud._normalize_to_16k_mono_pcm",
                  side_effect=HaCloudError("bad audio")):
            from tools.transcription_tools import _transcribe_ha_cloud
            result = _transcribe_ha_cloud(str(audio), "")
        assert result["success"] is False
        assert result["error"] == "bad audio"

    def test_transport_failure_is_error_dict(self, tmp_path, monkeypatch):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 16)
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_ha_cloud._normalize_to_16k_mono_pcm", return_value=b"\x00" * 320), \
             patch("tools.transcription_ha_cloud._run_ha_stt",
                   side_effect=HaCloudError("ws down")):
            from tools.transcription_tools import _transcribe_ha_cloud
            result = _transcribe_ha_cloud(str(audio), "")
        assert result["success"] is False
        assert result["error"] == "ws down"

    def test_empty_transcript_is_no_speech_error(self, tmp_path, monkeypatch):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 16)
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_ha_cloud._normalize_to_16k_mono_pcm", return_value=b"\x00" * 320), \
             patch("tools.transcription_ha_cloud._run_ha_stt", return_value=""):
            from tools.transcription_tools import _transcribe_ha_cloud
            result = _transcribe_ha_cloud(str(audio), "")
        assert result["success"] is False
        assert result.get("no_speech") is True

    def test_success_envelope(self, tmp_path, monkeypatch):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 16)
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_tools._load_stt_config", return_value={}), \
             patch("tools.transcription_ha_cloud._normalize_to_16k_mono_pcm", return_value=b"\x00" * 320), \
             patch("tools.transcription_ha_cloud._run_ha_stt", return_value="hello home") as run:
            from tools.transcription_tools import _transcribe_ha_cloud
            result = _transcribe_ha_cloud(str(audio), "")
        assert result == {"success": True, "transcript": "hello home", "provider": "ha_cloud"}
        assert run.call_args[0][3] == "01j5gssf495tbq990ab6p10cv4"


# ============================================================================
# Dispatcher registration
# ============================================================================

class TestDispatcherRegistration:
    def test_explicit_provider_resolves_when_available(self, monkeypatch):
        monkeypatch.setenv("HASS_TOKEN", "test-token")
        with patch("tools.transcription_ha_cloud._ha_cloud_reachable", return_value=True), \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            from tools.transcription_tools import _get_provider
            assert _get_provider({"provider": "ha_cloud"}) == "ha_cloud"

    def test_explicit_provider_refused_without_credentials(self):
        from tools.transcription_tools import _get_provider
        assert _get_provider({"provider": "ha_cloud"}) == "none"

    def test_dispatch_routes_to_handler(self, tmp_path):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"\x00" * 16)
        with patch("tools.transcription_tools._transcribe_ha_cloud",
                   return_value={"success": True, "transcript": "hi", "provider": "ha_cloud"}) as handler, \
             patch("tools.transcription_tools._load_stt_config", return_value={}):
            from tools.transcription_tools import _route_stt_provider
            result = _route_stt_provider(str(audio), "ha_cloud", {"provider": "ha_cloud"},
                                         None, None, None)
        assert result["transcript"] == "hi"
        handler.assert_called_once()

    def test_builtin_model_key_uses_pipeline(self):
        from tools.transcription_common import STT_MODEL_CONFIG_KEY
        from tools.transcription_tools import _builtin_model_name
        assert STT_MODEL_CONFIG_KEY["ha_cloud"] == "pipeline"
        assert _builtin_model_name("ha_cloud", {"ha_cloud": {"pipeline": "abc"}}, None) == "abc"
        assert _builtin_model_name("ha_cloud", {}, None) == "01j5gssf495tbq990ab6p10cv4"


# ============================================================================
# Full pipeline exchange against a mocked websocket layer
# ============================================================================

class _FakeWs:
    """Scripted stand-in for a websockets client connection."""

    def __init__(self, script):
        self._script = list(script)
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def recv(self):
        if not self._script:
            raise AssertionError("client read past the scripted messages")
        return self._script.pop(0)


class _FakeConnect:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


def _event(msg_id, etype, data):
    return json.dumps({"id": msg_id, "type": "event", "event": {"type": etype, "data": data}})


def _run_exchange(pcm, script):
    """Drive _stt_exchange with a fake websockets module; return (transcript, sent)."""
    ws = _FakeWs(script)
    fake_mod = types.ModuleType("websockets")
    fake_mod.connect = lambda *a, **k: _FakeConnect(ws)
    with patch.dict("sys.modules", {"websockets": fake_mod}):
        from tools.transcription_ha_cloud import _stt_exchange
        transcript = asyncio.run(_stt_exchange(pcm, "ws://ha:8123/api/websocket", "tok", "pipe"))
    return transcript, ws.sent


class TestPipelineExchange:
    MSG_ID = 10
    HANDLER = 3

    def _script(self, transcript="turn on the lights"):
        return [
            json.dumps({"type": "auth_required", "ha_version": "2026.10.1"}),
            json.dumps({"type": "auth_ok", "ha_version": "2026.10.1"}),
            _event(self.MSG_ID, "run-start", {"runner_data": {"stt_binary_handler_id": self.HANDLER}}),
            _event(self.MSG_ID, "stt-start", {}),
            _event(self.MSG_ID, "stt-end", {"stt_output": {"text": transcript}}),
        ]

    def test_happy_path_frames_and_transcript(self):
        pcm = b"\x01\x02" * 10000  # 20000 bytes -> 8192 + 8192 + 3616 + termination
        transcript, sent = _run_exchange(pcm, self._script())
        assert transcript == "turn on the lights"
        binary = [m for m in sent if isinstance(m, bytes)]
        assert len(binary) == 4
        assert binary[0] == bytes([self.HANDLER]) + pcm[:8192]
        assert binary[1] == bytes([self.HANDLER]) + pcm[8192:16384]
        assert binary[2] == bytes([self.HANDLER]) + pcm[16384:]
        # Stream ends with the bare handler byte (empty payload).
        assert binary[3] == bytes([self.HANDLER])

    def test_auth_failure_raises(self):
        from tools.transcription_ha_cloud import HaCloudError
        script = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_invalid", "message": "bad token"}),
        ]
        with pytest.raises(HaCloudError, match="auth failed"):
            _run_exchange(b"\x00" * 100, script)

    def test_run_end_without_transcript_raises(self):
        from tools.transcription_ha_cloud import HaCloudError
        script = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            _event(self.MSG_ID, "run-start", {"runner_data": {"stt_binary_handler_id": 1}}),
            _event(self.MSG_ID, "stt-start", {}),
            _event(self.MSG_ID, "run-end", {}),
        ]
        with pytest.raises(HaCloudError, match="no transcript"):
            _run_exchange(b"\x00" * 100, script)

    def test_pipeline_error_event_raises(self):
        from tools.transcription_ha_cloud import HaCloudError
        script = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            _event(self.MSG_ID, "run-start", {"runner_data": {"stt_binary_handler_id": 1}}),
            _event(self.MSG_ID, "stt-start", {}),
            _event(self.MSG_ID, "error", {"code": "stt-failed", "message": "boom"}),
        ]
        with pytest.raises(HaCloudError, match="STT error"):
            _run_exchange(b"\x00" * 100, script)

    def test_legacy_transcript_shape(self):
        transcript, _ = _run_exchange(b"\x00" * 100, [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
            _event(self.MSG_ID, "run-start", {"runner_data": {"stt_binary_handler_id": 9}}),
            _event(self.MSG_ID, "stt-start", {}),
            _event(self.MSG_ID, "stt-end", {"stt_text": "legacy words"}),
        ])
        assert transcript == "legacy words"
