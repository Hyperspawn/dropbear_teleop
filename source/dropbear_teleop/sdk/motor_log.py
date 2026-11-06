"""Append-only motor log: what every motor was told and what it did, tick by tick.

The same file comes from the simulator (``tools/newton_bridge.py --motor-log``) and, later, from the real CAN bridge, so
the same gate (``tools/teleop_gate.py``) judges both. Rows go straight to disk: a killed process loses at most the last
second, and memory stays flat however long the session runs. ``<path>`` holds float32 rows, ``<path>.json`` the layout.

Stdlib + numpy only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIELDS = ("q", "dq", "tau", "q_des", "dq_des", "tau_ff", "kp", "kd")
HEAD = ("time_s", "tick", "damping")


class MotorLogWriter:
    """``add`` one row per kept tick. ``tau`` is the motor's output torque (after the torque-speed envelope)."""

    def __init__(self, path: str | Path, names, dt: float, source: str, flush_rows: int = 250, **meta):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = len(names)
        side = {"schema": "dropbear-motor-log-v1", "names": list(names), "dt": float(dt), "source": source,
                "head": list(HEAD), "fields": list(FIELDS), **meta}
        self.path.with_name(self.path.name + ".json").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8")
        self._f = open(self.path, "wb", buffering=1 << 16)
        self._row = np.zeros(len(HEAD) + len(FIELDS) * self.n, dtype=np.float32)
        self._flush_rows = int(flush_rows)
        self.rows = 0

    def add(self, time_s: float, tick: int, damping: bool, **fields) -> None:
        r = self._row
        r[0], r[1], r[2] = time_s, tick, float(damping)
        for k, name in enumerate(FIELDS):
            r[len(HEAD) + k * self.n:len(HEAD) + (k + 1) * self.n] = fields[name]
        self._f.write(r.tobytes())
        self.rows += 1
        if self.rows % self._flush_rows == 0:
            self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def read_motor_log(path: str | Path) -> dict:
    """{"meta": sidecar, "time_s", "tick", "damping", <field>: (T, n)}. A truncated last row (killed writer) is dropped."""
    path = Path(path)
    meta = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    n = len(meta["names"])
    width = len(meta["head"]) + len(meta["fields"]) * n
    raw = np.fromfile(path, dtype=np.float32)
    raw = raw[:len(raw) // width * width].reshape(-1, width)
    out: dict = {"meta": meta}
    for i, k in enumerate(meta["head"]):
        out[k] = raw[:, i].astype(np.float64)
    o = len(meta["head"])
    for j, k in enumerate(meta["fields"]):
        out[k] = raw[:, o + j * n:o + (j + 1) * n].astype(np.float64)
    return out
