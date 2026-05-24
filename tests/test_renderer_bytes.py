import struct
import numpy as np
from tactile_vision.tactile.renderer import to_bytes, MODE_IDS
from tactile_vision.types import PinMatrix


def _make_pm(mode="walk", rows=2, cols=3, levels=8):
    data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    return PinMatrix(rows=rows, cols=cols, levels=levels, data=data, timestamp=0.0, mode=mode)


def test_header_format():
    pm = _make_pm(mode="walk")
    payload = to_bytes(pm, frame_index=42)
    assert len(payload) == 14
    frame_idx, rows, cols, levels, mode_id = struct.unpack("<IBBBB", payload[:8])
    assert frame_idx == 42
    assert rows == 2
    assert cols == 3
    assert levels == 8
    assert mode_id == MODE_IDS["walk"]


def test_data_payload_is_row_major():
    pm = _make_pm(mode="scene")
    payload = to_bytes(pm, frame_index=0)
    data = payload[8:]
    assert list(data) == [0, 1, 2, 3, 4, 5]


def test_mode_id_differs_between_modes():
    assert MODE_IDS["walk"] != MODE_IDS["scene"]
