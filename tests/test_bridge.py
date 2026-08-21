"""Tests for the Unity bridge protocol.

No sockets, no camera, no model -- the protocol and the gate are pure, which is the whole
reason they live apart from `08_bridge.py`. The field names asserted here are a contract
with Unity's `JutsuMessage` / `JutsuControlMessage`: if a key is renamed on either side,
casts silently stop landing, so the shape is pinned deliberately.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from handsign.bridge import (
    PROTOCOL_VERSION,
    ControlMessage,
    RecognitionGate,
    UplinkEncoder,
    decode_control,
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def decode(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


# --------------------------------------------------------------------------- uplink


def test_cast_is_repeated_and_every_copy_is_identical():
    # A dropped cast costs the player the seconds they spent forming the sequence.
    encoder = UplinkEncoder(cast_repeat=3)
    payloads = encoder.cast("Chidori", ["ox", "hare", "monkey"])

    assert len(payloads) == 3
    assert len(set(payloads)) == 1, "Unity de-duplicates on seq, so copies must match"


def test_cast_sequence_numbers_increment_so_unity_can_deduplicate():
    encoder = UplinkEncoder(cast_repeat=1)

    first = decode(encoder.cast("Chidori")[0])
    second = decode(encoder.cast("Clone Jutsu")[0])

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert encoder.last_seq == 2


def test_cast_payload_matches_unitys_expected_field_names():
    encoder = UplinkEncoder()
    payload = decode(encoder.cast("Dragon Flame Jutsu", ["snake", "dragon"])[0])

    assert payload["v"] == PROTOCOL_VERSION
    assert payload["type"] == "cast"
    assert payload["jutsu"] == "Dragon Flame Jutsu"
    assert payload["signs"] == ["snake", "dragon"]
    assert "seq" in payload


def test_sign_and_progress_and_heartbeat_shapes():
    encoder = UplinkEncoder()

    sign = decode(encoder.sign("tiger", 0.9123456))
    assert sign["type"] == "sign"
    assert sign["sign"] == "tiger"
    assert sign["conf"] == pytest.approx(0.9123, abs=1e-4)

    progress = decode(encoder.progress("Water Bullet Jutsu", 2, 4))
    assert progress["type"] == "progress"
    assert (progress["matched"], progress["total"]) == (2, 4)

    beat = decode(encoder.heartbeat(31.234))
    assert beat["type"] == "heartbeat"
    assert beat["fps"] == pytest.approx(31.23, abs=1e-2)


def test_cast_repeat_must_be_positive():
    with pytest.raises(ValueError):
        UplinkEncoder(cast_repeat=0)


# --------------------------------------------------------------------------- downlink


def test_decode_control_reads_a_recognition_message():
    payload = json.dumps(
        {"v": 1, "type": "recognition", "enabled": False, "reason": "lock", "seconds": 6.0}
    ).encode()

    message = decode_control(payload)

    assert message == ControlMessage(
        type="recognition", enabled=False, reason="lock", seconds=6.0
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not json at all",
        b"[1, 2, 3]",            # valid JSON, wrong shape
        b'{"no_type": true}',
        b'{"type": ""}',
        b"\xff\xfe\x00",         # not even UTF-8
    ],
)
def test_decode_control_drops_unusable_packets_instead_of_raising(payload):
    # A corrupt datagram is an expected network condition, not a crash.
    assert decode_control(payload) is None


def test_decode_control_defaults_missing_fields():
    message = decode_control(b'{"type": "recognition"}')

    assert message is not None
    assert message.enabled is True
    assert message.reason == ""
    assert message.seconds == 0.0


# --------------------------------------------------------------------------- gate


def test_gate_starts_open():
    gate = RecognitionGate()
    assert gate.enabled is True
    assert gate.seconds_remaining(now=0.0) == 0.0


def test_gate_reports_the_moment_it_closes_so_the_caller_can_reset():
    # The return value is the caller's cue to clear the smoother and sequence buffer.
    gate = RecognitionGate()
    closing = ControlMessage(type="recognition", enabled=False, reason="lock", seconds=6.0)

    assert gate.apply(closing, now=100.0) is True
    assert gate.enabled is False
    assert gate.seconds_remaining(now=100.0) == pytest.approx(6.0)
    assert gate.seconds_remaining(now=104.0) == pytest.approx(2.0)
    assert gate.seconds_remaining(now=110.0) == 0.0


def test_gate_does_not_re_report_a_close_that_already_happened():
    gate = RecognitionGate()
    closing = ControlMessage(type="recognition", enabled=False, reason="lock", seconds=6.0)

    assert gate.apply(closing, now=0.0) is True
    assert gate.apply(closing, now=1.0) is False, "already closed -- nothing to reset"


def test_gate_reopens_on_resume():
    gate = RecognitionGate()
    gate.apply(ControlMessage(type="recognition", enabled=False, seconds=6.0), now=0.0)

    assert gate.apply(ControlMessage(type="recognition", enabled=True), now=6.0) is False
    assert gate.enabled is True
    assert gate.seconds_remaining(now=6.0) == 0.0


def test_match_over_closes_the_gate_and_a_new_match_reopens_it():
    gate = RecognitionGate()

    assert gate.apply(ControlMessage(type="match", state="over"), now=0.0) is True
    assert gate.enabled is False
    assert gate.match_running is False
    assert gate.reason == "match_over"

    gate.apply(ControlMessage(type="match", state="running"), now=1.0)
    assert gate.enabled is True
    assert gate.match_running is True


def test_gate_remembers_the_last_rejection_without_closing():
    gate = RecognitionGate()
    rejection = ControlMessage(type="rejected", jutsu="Chidori", reason="chakra")

    assert gate.apply(rejection, now=0.0) is False
    assert gate.enabled is True, "a rejection is feedback, not a lock"
    assert gate.last_rejection == rejection


def test_gate_ignores_message_types_it_does_not_know():
    gate = RecognitionGate()
    assert gate.apply(ControlMessage(type="something_new"), now=0.0) is False
    assert gate.enabled is True


def test_describe_is_useful_on_the_console():
    gate = RecognitionGate()
    assert gate.describe(now=0.0) == "recognising"

    gate.apply(ControlMessage(type="recognition", enabled=False, reason="stun", seconds=2.0), now=0.0)
    assert "stun" in gate.describe(now=0.0)


# --------------------------------------------------------------------------- entrypoint


@pytest.fixture(scope="module")
def bridge_module():
    return _load("bridge_entry", "08_bridge.py")


def test_resolve_jutsu_accepts_index_name_and_substring(bridge_module):
    from handsign import load_jutsu

    jutsu_list = load_jutsu(Path(__file__).resolve().parents[1] / "demo-short.csv")
    resolve = bridge_module.resolve_jutsu

    assert resolve(jutsu_list, "1") is jutsu_list[0]
    assert resolve(jutsu_list, "Chidori").name == "Chidori"
    assert resolve(jutsu_list, "chidori").name == "Chidori"
    assert resolve(jutsu_list, "vacuum").name == "Vacuum Wave"


def test_resolve_jutsu_rejects_nonsense(bridge_module):
    from handsign import load_jutsu

    jutsu_list = load_jutsu(Path(__file__).resolve().parents[1] / "demo-short.csv")
    resolve = bridge_module.resolve_jutsu

    assert resolve(jutsu_list, "") is None
    assert resolve(jutsu_list, "0") is None
    assert resolve(jutsu_list, "999") is None
    assert resolve(jutsu_list, "rasengan") is None


def test_every_demo_short_jutsu_name_exists_in_unity_battle_stats():
    """The one piece of vocabulary the two sides share is the jutsu name.

    Unity looks up damage and chakra by that string. A rename on either side would make
    casts silently do nothing, so the two tables are checked against each other here.
    """
    from handsign import load_jutsu

    repo = Path(__file__).resolve().parents[1]
    stats = repo.parent.parent.parent / "NinjaGame_Background" / "Assets" / "Resources" / "battle_stats.csv"
    if not stats.exists():
        pytest.skip(f"Unity project not found at {stats}")

    unity_names = set()
    for line in stats.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("name,"):
            continue
        unity_names.add(line.split(",")[0].strip())

    python_names = {j.name for j in load_jutsu(repo / "demo-short.csv")}

    assert python_names <= unity_names, (
        f"these jutsu have no combat stats in Unity: {sorted(python_names - unity_names)}"
    )
