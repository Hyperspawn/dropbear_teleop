"""The arm gate (``tools/teleop_gate.py``) on synthetic motor logs, and the crash-proof log itself."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from dropbear_teleop.robots.hw_motor_specs import MOTOR_MODELS, line_at_supply  # noqa: E402
from dropbear_teleop.sdk import motors  # noqa: E402
from dropbear_teleop.sdk.motor_log import FIELDS, MotorLogWriter, read_motor_log  # noqa: E402
from teleop_gate import ARM_MOTORS, gate  # noqa: E402

DT = 0.004
X8 = MOTOR_MODELS["RMD-X8-Pro-V2-1:9"]


def write_log(path: Path, seconds: float = 30.0, fn=None) -> Path:
    """A log of a gentle arm motion; ``fn(t, i, row)`` changes motor ``i``'s fields at time ``t``."""
    log = MotorLogWriter(path, motors.MOTOR_NAMES, DT, "sim", profile="hw_cem6")
    n = len(motors.MOTOR_NAMES)
    arm = [motors.MOTOR_NAMES.index(m) for m in ARM_MOTORS]
    for k in range(int(seconds / DT)):
        t = k * DT
        row = {f: np.zeros(n) for f in FIELDS}
        for i in arm:
            row["q_des"][i] = 0.2 * np.sin(0.5 * t)
            row["q"][i] = row["q_des"][i]
            row["dq"][i] = 0.1 * np.cos(0.5 * t)
            row["dq_des"][i] = row["dq"][i]
            row["tau"][i] = 1.0
            row["tau_ff"][i] = 1.0
            row["kp"][i], row["kd"][i] = 50.0, 2.0
            if fn is not None:
                fn(t, motors.MOTOR_NAMES[i], row, i)
        log.add(t, k, False, **row)
    log.close()
    return path


def test_gentle_motion_passes(tmp_path):
    g = gate(write_log(tmp_path / "a.motorlog"))
    assert g["verdict"] == "PASS"
    assert [r["motor"] for r in g["motors"]] == list(ARM_MOTORS)
    assert g["motors"][0]["can_id"] == 35 and g["motors"][0]["model"] == "RMD-X10-V3-1:7"


def test_demand_above_peak_fails_clipped(tmp_path):
    def fn(t, name, row, i):
        if name == "RH_elbow_joint":
            row["tau_ff"][i] = 2.0 * X8.peak_torque  # the motor cannot give this
            row["tau"][i] = X8.peak_torque
    g = gate(write_log(tmp_path / "a.motorlog", fn=fn))
    r = next(r for r in g["motors"] if r["motor"] == "RH_elbow_joint")
    assert g["verdict"] == "FAIL" and "clipped" in r["high"] and r["clipped_frac"] == 1.0


def test_one_tick_of_motion_is_not_clipping(tmp_path):
    """Logged torque and this row's demand differ by a tick of motion: within the line, that is not clipping."""
    def fn(t, name, row, i):
        row["tau"][i] = row["tau_ff"][i] + 3.0
    g = gate(write_log(tmp_path / "a.motorlog", fn=fn))
    assert all(r["clipped_frac"] == 0.0 for r in g["motors"])


def test_rms_above_rated_fails_thermal(tmp_path):
    def fn(t, name, row, i):
        if name == "LH_pitch":
            row["tau"][i] = row["tau_ff"][i] = 1.5 * X8.rated_torque
    g = gate(write_log(tmp_path / "a.motorlog", fn=fn))
    assert "thermal" in next(r for r in g["motors"] if r["motor"] == "LH_pitch")["high"]


def test_speed_margin_depends_on_the_supply(tmp_path):
    """10.2 rad/s on an X8: fine at 48 V (no-load 16.8), above 95 % of no-load at 30 V (10.5)."""
    def fn(t, name, row, i):
        if name == "LH_wrist_roll":
            row["dq"][i] = row["dq_des"][i] = 10.2
    p = write_log(tmp_path / "a.motorlog", fn=fn)
    assert "speed" in next(r for r in gate(p, supply_v=30.0)["motors"] if r["motor"] == "LH_wrist_roll")["high"]
    assert gate(p, supply_v=48.0)["verdict"] == "PASS"


def test_oscillation_while_holding_fails_stability(tmp_path):
    def fn(t, name, row, i):
        row["q_des"][i] = 0.1
        row["q"][i] = 0.1
        row["dq_des"][i] = 0.0
        row["dq"][i] = 0.0
        if name == "RH_roll":
            row["dq"][i] = 0.5 * np.sin(2 * np.pi * 30.0 * t)
            row["tau"][i] = row["tau_ff"][i] = 8.0 * np.sin(2 * np.pi * 30.0 * t)
    g = gate(write_log(tmp_path / "a.motorlog", seconds=10.0, fn=fn))
    r = next(r for r in g["motors"] if r["motor"] == "RH_roll")
    assert g["verdict"] == "FAIL" and {"hold_speed", "torque_ripple"} <= set(r["high"])
    assert next(r for r in g["motors"] if r["motor"] == "LH_roll")["verdict"] != "FAIL"


def test_line_at_supply_keeps_the_slope():
    v0, sat = line_at_supply(X8, 30.0)
    assert v0 == pytest.approx(X8.no_load_speed * 0.625) and sat == pytest.approx(X8.saturation_effort * 0.625)
    assert sat / v0 == pytest.approx(X8.saturation_effort / X8.no_load_speed)


def test_killed_writer_loses_only_the_tail(tmp_path):
    """No close(): what reached the disk reads back, and a half-written row is dropped."""
    p = tmp_path / "a.motorlog"
    log = MotorLogWriter(p, ["a", "b"], 0.01, "sim", flush_rows=10)
    for k in range(25):
        log.add(k * 0.01, k, False, **{f: np.full(2, float(k)) for f in FIELDS})
    log._f.flush()
    with open(p, "ab") as f:
        f.write(b"\x00\x00\x00")  # a row cut off mid-write
    d = read_motor_log(p)
    assert d["q"].shape == (25, 2) and d["tick"][-1] == 24.0
    np.testing.assert_allclose(d["kp"][7], [7.0, 7.0])
    log.close()
