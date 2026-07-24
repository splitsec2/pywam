"""Regression tests for the pywam `all-fixes` branch.

Each test targets one concrete fix and is written so that it PASSES on
`all-fixes` and FAILS on `master` (the unfixed upstream). Running the
same suite against both checkouts (via the ``PYWAM_SRC`` env var in
``conftest.py``) proves the fix is real and observable, not cosmetic.

There is no real speaker; the network client / socket layer is mocked.
"""
from __future__ import annotations

import asyncio

import pytest

from pywam.client import WamClient
from pywam.lib import api_call
from pywam.lib.api_response import ApiResponse
from pywam.lib.exceptions import ApiCallError, ApiCallTimeoutError
from pywam.speaker import Speaker

TEST_IP = "192.168.1.100"


def _ok(method: str = "", data=None) -> ApiResponse:
    """Build a benign, successful ApiResponse."""
    if data is None:
        data = {"@result": "ok"}
    return ApiResponse(method=method, success=True, data=data)


# ======================================================================
# Fix 1: update() tolerates source-unsupported calls, keyed by source
# ======================================================================


class _RecordingClient:
    """Stand-in for WamClient.request that records calls and can time out.

    Methods listed in ``timeout_methods`` raise ApiCallTimeoutError (as a
    real speaker does when the query is meaningless for the active
    source, e.g. shuffle/repeat/presets on an HDMI input); every other
    call returns a benign successful response.
    """

    def __init__(self, timeout_methods: set[str]) -> None:
        self.timeout_methods = timeout_methods
        self.calls: list[str] = []

    async def request(self, call: api_call.ApiCall) -> ApiResponse:
        self.calls.append(call.method)
        if call.method in self.timeout_methods:
            raise ApiCallTimeoutError(f"({TEST_IP}) No response from speaker")
        return _ok(call.expected_response)


@pytest.mark.asyncio
async def test_update_player_info_tolerates_unanswered_calls_per_source():
    """update_player_info() must not abort when the speaker ignores the
    optional shuffle/repeat queries (HDMI source), and must cache the
    skip keyed by source so a source change retries it."""
    speaker = Speaker(TEST_IP)
    fake = _RecordingClient(timeout_methods={"GetShuffleMode", "GetRepeatMode"})
    speaker.client.request = fake.request  # type: ignore[assignment]

    # Active source is an external HDMI input.
    speaker.attribute._function = "hdmi"

    # On all-fixes this completes; on master the raw request() for
    # GetShuffleMode raises ApiCallTimeoutError straight out.
    await speaker.update_player_info()

    # The unanswered optional calls were cached, keyed by the source.
    assert ("hdmi", "GetShuffleMode") in speaker._unsupported
    assert ("hdmi", "GetRepeatMode") in speaker._unsupported

    # A second pass in the SAME source skips them (request not re-issued).
    calls_before = fake.calls.count("GetShuffleMode")
    await speaker.update_player_info()
    assert fake.calls.count("GetShuffleMode") == calls_before

    # Changing the source re-enables the query (per-source, not permanent).
    speaker.attribute._function = "wifi"
    await speaker.update_player_info()
    assert fake.calls.count("GetShuffleMode") == calls_before + 1


@pytest.mark.asyncio
async def test_update_speaker_settings_tolerates_unanswered_preset_call():
    """update_speaker_settings() must tolerate the speaker ignoring the
    preset-list query in a non-radio source."""
    speaker = Speaker(TEST_IP)
    fake = _RecordingClient(timeout_methods={"GetPresetList"})
    speaker.client.request = fake.request  # type: ignore[assignment]

    speaker.attribute._function = "hdmi"

    # Completes on all-fixes; raises ApiCallTimeoutError on master.
    await speaker.update_speaker_settings()

    assert ("hdmi", "GetPresetList") in speaker._unsupported


# ======================================================================
# Fix 2: listener does not deadlock on a full response queue
# ======================================================================


@pytest.mark.asyncio
async def test_return_response_does_not_block_on_full_queue():
    """_return_response() must not block the listener when the response
    queue is already at capacity (trailing message case)."""
    speaker = Speaker(TEST_IP)
    client = speaker.client

    # Pre-fill the single-slot queue to capacity.
    client._response_queue = asyncio.Queue(1)
    client._response_queue.put_nowait(_ok("MuteStatus"))

    trailing = _ok("MuteStatus")

    # On all-fixes put_nowait -> QueueFull is caught and the message is
    # dropped, so this returns immediately. On master `await queue.put()`
    # blocks forever on the full queue, so wait_for raises TimeoutError.
    await asyncio.wait_for(client._return_response(trailing), timeout=1.0)

    # The original item is still there (trailing message was dropped).
    assert client._response_queue.qsize() == 1


# ======================================================================
# Fix 3: no-response request path closes the per-request writer
# ======================================================================


class _FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.wait_closed_awaited = False
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_awaited = True


@pytest.mark.asyncio
async def test_no_response_request_closes_writer(monkeypatch):
    """A request with no expected response must close its per-request
    writer, otherwise every such call leaks a socket/fd."""
    speaker = Speaker(TEST_IP)
    client = speaker.client

    # Make the client look connected + listening so request() proceeds.
    client._event_reader = object()  # type: ignore[assignment]
    client._event_writer = object()  # type: ignore[assignment]
    client._listening.set()

    writer = _FakeWriter()

    async def fake_open_connection(host, port):
        return object(), writer

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    # get_feature() has expected_response == '' (no-response path).
    call = api_call.get_feature()
    assert call.expected_response == ""

    result = await client.request(call)
    assert isinstance(result, ApiResponse)

    # On all-fixes the writer is closed and awaited; on master it is not.
    assert writer.closed is True
    assert writer.wait_closed_awaited is True


# ======================================================================
# Fix 4: _dispatch_event notifies ALL subscribers, not just the first
# ======================================================================


@pytest.mark.asyncio
async def test_dispatch_event_notifies_all_subscribers():
    """Every subscriber must be notified of a state change, not just the
    first one in registration order."""
    speaker = Speaker(TEST_IP)
    events = speaker.events

    fired: list[str] = []

    def sub_one() -> None:
        fired.append("one")

    def sub_two() -> None:
        fired.append("two")

    events.register_subscriber(sub_one, info_level=0)
    events.register_subscriber(sub_two, info_level=0)

    # Force an observable state change: the new state differs from the
    # last-known state on a shared key.
    events._latest_known_state = {"volume": 1}
    events._attr.get_state_copy = lambda: {"volume": 2}  # type: ignore[assignment]

    events._dispatch_event(used=True, event=_ok("MuteStatus"))

    # On all-fixes both fire; on master only the first fires because the
    # loop advances _latest_known_state and early-returns for the rest.
    assert "one" in fired
    assert "two" in fired


# ======================================================================
# Fix 5: MusicInfo timelength parsed right-aligned (shared helper)
# ======================================================================


@pytest.mark.asyncio
async def test_music_info_timelength_short_format():
    """A timelength shorter than HH:MM:SS.uuu ("23:45.678" == 23m45s)
    must be parsed as 1425 s, not front-aligned into hours."""
    speaker = Speaker(TEST_IP)
    events = speaker.events

    response = ApiResponse(
        method="MusicInfo",
        success=True,
        data={"@result": "ok", "timelength": "23:45.678"},
    )

    used = events.event_MusicInfo(response)
    assert used is True

    # all-fixes: timelength_to_sec right-aligns -> 23*60 + 45 = 1425.
    # master: front-aligned zip -> 23*3600 + 45*60 + 678 = 86178.
    assert speaker.attribute._tracklength == "1425"


# ======================================================================
# Fix 6: get_tunein_presets matches the real cpname casing ("TuneIn")
# ======================================================================


@pytest.mark.asyncio
async def test_get_tunein_presets_cpname_case():
    """get_tunein_presets() must accept the speaker's actual cpname
    casing ("TuneIn") and return the preset list."""
    speaker = Speaker(TEST_IP)

    presets = [{"contentid": "0", "title": "Radio One"}]

    async def fake_request(call: api_call.ApiCall) -> ApiResponse:
        if call.method == "GetPresetList":
            return ApiResponse(
                method="PresetList",
                success=True,
                data={
                    "@result": "ok",
                    "cpname": "TuneIn",
                    "presetlist": {"preset": presets},
                },
            )
        return _ok(call.expected_response)

    speaker.client.request = fake_request  # type: ignore[assignment]

    # all-fixes: cpname == "TuneIn" matches -> returns presets.
    # master: compares against "tunein" -> raises ApiCallError.
    result = await speaker.get_tunein_presets()
    assert result == presets


def test_get_tunein_presets_raises_on_non_tunein():
    """Sanity guard (passes on both branches): a non-TuneIn cpname still
    raises. Kept so the fix isn't mistaken for 'always return'."""

    async def run():
        speaker = Speaker(TEST_IP)

        async def fake_request(call: api_call.ApiCall) -> ApiResponse:
            if call.method == "GetPresetList":
                return ApiResponse(
                    method="PresetList",
                    success=True,
                    data={"@result": "ok", "cpname": "Spotify"},
                )
            return _ok(call.expected_response)

        speaker.client.request = fake_request  # type: ignore[assignment]
        with pytest.raises(ApiCallError):
            await speaker.get_tunein_presets()

    asyncio.run(run())
