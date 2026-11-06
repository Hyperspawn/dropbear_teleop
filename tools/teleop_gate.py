"""Would these arm motions work on the real motors? A motor log against the datasheet, motor by motor.

Input: a ``dropbear-motor-log-v1`` file (:mod:`dropbear_teleop.sdk.motor_log`): the simulator's today (``tools/quest_session.py``
writes one per session), the real CAN bridge's later. Default: the newest ``logs/teleop/*.motorlog``.

Two questions:

* **Possible?** The hardware gate of ``tools/telemetry_issue_scan.py`` (thermal, clipped, speed, stop load) on the 10 arm
  motors, with the datasheet torque-speed line moved to the supply voltage (``--supply-v``; the motors are rated at 48 V).
* **Stable?** Fast torque ripple and joint speed while the target is still (a limit cycle shows in both). Also reported:
  tracking error and the damping ratio the gains give on paper with the motor's reflected inertia and the arm's.

    python tools/teleop_gate.py
    python tools/teleop_gate.py logs/teleop/quest_2026-11-06_14-00.motorlog --supply-v 48

A simulated log ran on the 48 V line, so the torque the motor would give at a lower voltage is the logged torque clipped
to the lower line: the motion itself is not re-simulated. A pass in simulation says the motion fits the datasheet with the
CAD's masses. It does not prove the real arm: friction, backlash, printed-part flex, the real masses and how the motor
firmware applies kd are not in it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "tools")]

import numpy as np  # noqa: E402

from dropbear_teleop.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG  # noqa: E402
from dropbear_teleop.robots.hw_motor_specs import (  # noqa: E402
    ARM_MOTOR_LABELS, HW_PROFILE_MAPS, JOINT_MOTOR_MAPS, MOTOR_CAN_IDS, MOTOR_MODELS, line_at_supply, link_inertia_at_rest,
)
from dropbear_teleop.sdk.motor_log import read_motor_log  # noqa: E402
from telemetry_issue_scan import HW_GATE, scan  # noqa: E402

ARM_MOTORS = tuple(ARM_MOTOR_LABELS)
STABILITY = {"torque_ripple": (0.05, 0.15),  # RMS of torque minus its 100 ms average, / peak
             "hold_speed": (0.05, 0.20)}     # rad/s, RMS joint speed once the target has been still for 0.5 s
TRACKING_WARN = 0.15  # rad, p95 |target - position|: a quality warning, the motor is not at risk
ZETA_LOW = 0.2        # damping ratio on paper: below this, expect ringing
HOLD_AFTER_S = 0.5
RIPPLE_WINDOW_S = 0.1


def envelope(dq: np.ndarray, peak: np.ndarray, sat: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest motor torque available at speed ``dq`` (``hw_motor_torque``'s line)."""
    v = np.clip(dq, -v0 * (1.0 + peak / sat), v0 * (1.0 + peak / sat))
    return np.maximum(sat * (-1.0 - v / v0), -peak), np.minimum(sat * (1.0 - v / v0), peak)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centred average along time, same length (shrinks to a one-sided window at the ends)."""
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    lo = np.clip(np.arange(len(x)) - width // 2, 0, len(x))
    hi = np.clip(np.arange(len(x)) + width // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def hold_speed(dq: np.ndarray, q_des: np.ndarray, dt: float) -> np.ndarray:
    """RMS joint speed at rows whose target has not moved for ``HOLD_AFTER_S``, per motor (NaN: under 0.1 s of such rows)."""
    moved = np.abs(np.diff(q_des, axis=0, prepend=q_des[:1])) > 1e-5
    rows = np.arange(len(q_des))[:, None]
    last = np.maximum.accumulate(np.where(moved, rows, 0), axis=0)
    still = rows - last > HOLD_AFTER_S / dt
    out = np.full(dq.shape[1], np.nan)
    for i in range(dq.shape[1]):
        if still[:, i].sum() * dt >= 0.1:
            out[i] = float(np.sqrt((dq[still[:, i], i] ** 2).mean()))
    return out


def damping_ratio(name: str, armature: float, kp: float, kd: float) -> float:
    try:
        link = link_inertia_at_rest(name)
    except (KeyError, FileNotFoundError):
        link = 0.0
    return float(kd / (2.0 * np.sqrt(max(kp, 1e-9) * (armature + link))))


def sliding_rms(x: np.ndarray, width: int) -> np.ndarray:
    """Largest RMS of ``x`` over any ``width`` rows, per column (the whole-run RMS when the run is shorter)."""
    if len(x) <= width:
        return np.sqrt((x ** 2).mean(0))
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x ** 2, axis=0)])
    return np.sqrt((c[width:] - c[:-width]).max(0) / width)


def gate(path: Path, supply_v: float = 30.0, skip_s: float = 0.0) -> dict:
    log = read_motor_log(path)
    meta = log["meta"]
    keep = (log["damping"] < 0.5) & (log["time_s"] >= log["time_s"][0] + skip_s)
    if keep.sum() < 100:
        raise SystemExit(f"{path}: only {int(keep.sum())} rows while following commands")
    idx = [meta["names"].index(n) for n in ARM_MOTORS]
    t = log["time_s"][keep]
    q, dq, tau, q_des, dq_des, tau_ff, kp, kd = (log[k][keep][:, idx] for k in ("q", "dq", "tau", "q_des", "dq_des", "tau_ff", "kp", "kd"))
    dt = float(np.median(np.diff(t)))
    profile = meta.get("profile") or "hw_cem6"
    motor_map = JOINT_MOTOR_MAPS[HW_PROFILE_MAPS[profile]]
    models = [MOTOR_MODELS[motor_map[n]] for n in ARM_MOTORS]
    peak = np.array([m.peak_torque for m in models])
    rated = np.array([m.rated_torque for m in models])
    line = [line_at_supply(m, supply_v) for m in models]
    v0, sat = np.array([x[0] for x in line]), np.array([x[1] for x in line])
    lo, hi = envelope(dq, peak, sat, v0)
    demand = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
    # a real motor reports its own torque, and even the simulator's differs from this row's demand by one tick of motion:
    # "clipped" is the demand outside the line, not a difference of the two
    tau_out = np.clip(tau, lo, hi)
    over = (demand > hi) | (demand < lo)
    tau_pd = np.where(over, demand, tau_out)
    lim = np.deg2rad([MOTOR_HARD_LIMITS_DEG[n] for n in ARM_MOTORS])

    stride = max(1, int(round(0.02 / dt)))
    tel = path.with_name(path.stem + "_telemetry.npz")
    np.savez_compressed(
        tel, dt=dt * stride, motor_names=np.array(ARM_MOTORS), peak_torque=peak, rated_torque=rated, no_load_speed=v0,
        joint_limits=lim, q=q[::stride], qd=dq[::stride], q_target=q_des[::stride], tau=tau_out[::stride],
        tau_pd=tau_pd[::stride], tau_sub=tau_out,
        meta=json.dumps({"source": meta.get("source"), "actuator_profile": profile, "supply_v": supply_v}))
    res = scan(tel)

    ripple = np.sqrt(((tau_out - moving_average(tau_out, max(3, int(round(RIPPLE_WINDOW_S / dt))))) ** 2).mean(0)) / peak
    hold = hold_speed(dq, q_des, dt)
    track = np.percentile(np.abs(q_des - q), 95, axis=0)
    worst5 = sliding_rms(tau_out, int(round(5.0 / dt)))
    rows = []
    for i, n in enumerate(ARM_MOTORS):
        found = [f for f in res["findings"] if f.get("motor") == n and f["check"] != "asymmetry"]
        high = {f["check"] for f in found if f["severity"] == "HIGH" and f["check"] in HW_GATE}
        warn = {f["check"] for f in found} - high
        for key, value in (("torque_ripple", ripple[i]), ("hold_speed", hold[i])):
            if np.isfinite(value):
                if value >= STABILITY[key][1]:
                    high.add(key)
                elif value >= STABILITY[key][0]:
                    warn.add(key)
        if track[i] >= TRACKING_WARN:
            warn.add("tracking")
        zeta = damping_ratio(n, models[i].armature, float(np.median(kp[:, i])), float(np.median(kd[:, i])))
        if zeta < ZETA_LOW:
            warn.add("low damping on paper")
        rows.append({
            "motor": n, "label": ARM_MOTOR_LABELS[n], "can_id": MOTOR_CAN_IDS[n], "model": models[i].name,
            "rated_Nm": float(rated[i]), "peak_Nm": float(peak[i]), "no_load_rad_s": round(float(v0[i]), 2),
            "max_tau_Nm": round(float(np.abs(tau_out[:, i]).max()), 2),
            "max_demand_Nm": round(float(np.abs(demand[:, i]).max()), 2),
            "rms_tau_Nm": round(float(np.sqrt((tau_out[:, i] ** 2).mean())), 2),
            "worst_5s_rms_Nm": round(float(worst5[i]), 2),
            "max_speed_rad_s": round(float(np.abs(dq[:, i]).max()), 2),
            "clipped_frac": round(float(over[:, i].mean()), 4),
            "range_deg": [round(float(np.degrees(q[:, i].min())), 1), round(float(np.degrees(q[:, i].max())), 1)],
            "limits_deg": [round(float(np.degrees(lim[i, 0]))), round(float(np.degrees(lim[i, 1])))],
            "ripple_of_peak": round(float(ripple[i]), 4),
            "hold_speed_rad_s": None if not np.isfinite(hold[i]) else round(float(hold[i]), 4),
            "tracking_deg": round(float(np.degrees(track[i])), 2), "zeta": round(zeta, 2),
            "kp": float(np.median(kp[:, i])), "kd": float(np.median(kd[:, i])),
            "high": sorted(high), "warn": sorted(warn), "verdict": "FAIL" if high else "warn" if warn else "pass",
        })
    power = np.clip(tau_out * dq, 0.0, None).sum(1)
    return {"log": str(path), "source": meta.get("source"), "profile": profile, "supply_v": float(supply_v),
            "seconds": round(float(t[-1] - t[0]), 1), "rate_hz": round(1.0 / dt, 1),
            "verdict": "FAIL" if any(r["high"] for r in rows) else "PASS", "motors": rows,
            "shaft_power_W": {"mean": round(float(power.mean()), 1),
                              "peak_100ms": round(float(moving_average(power[:, None], max(3, int(round(0.1 / dt)))).max()), 1)},
            "findings": res["findings"], "telemetry": str(tel),
            "_plot": {"dq": dq[::stride], "tau": tau_out[::stride], "peak": peak, "rated": rated, "sat": sat, "v0": v0}}


def table(g: dict) -> str:
    out = ["| Motor | CAN | Peak torque (limit) | RMS / worst 5 s (rated) | Max speed (no-load) | Range used (limits) | "
           "Ripple | Hold speed | Tracking | Damping | Verdict |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"]
    for r in g["motors"]:
        hold = "-" if r["hold_speed_rad_s"] is None else f"{r['hold_speed_rad_s']:.3f}"
        why = ", ".join(r["high"] + r["warn"])
        out.append(f"| {r['label']} | {r['can_id']} | {r['max_tau_Nm']:.1f} ({r['peak_Nm']:.0f}) N·m | "
                   f"{r['rms_tau_Nm']:.1f} / {r['worst_5s_rms_Nm']:.1f} ({r['rated_Nm']:.0f}) | "
                   f"{r['max_speed_rad_s']:.1f} ({r['no_load_rad_s']:.1f}) rad/s | "
                   f"{r['range_deg'][0]:.0f}..{r['range_deg'][1]:.0f}° ({r['limits_deg'][0]}..{r['limits_deg'][1]}) | "
                   f"{100 * r['ripple_of_peak']:.1f}% | {hold} | {r['tracking_deg']:.1f}° | {r['zeta']:.2f} | "
                   f"{r['verdict']}{': ' + why if why else ''} |\n")
    return "".join(out)


def report(g: dict) -> str:
    p = g["shaft_power_W"]
    return (f"# Arm gate: {g['verdict']}\n\n{g['source']} motor log, {g['seconds']} s at {g['rate_hz']} Hz, `{g['profile']}`, "
            f"torque-speed line at {g['supply_v']:.0f} V. Shaft power, both arms: {p['mean']} W mean, {p['peak_100ms']} W peak "
            f"(mechanical only; the motors' own losses come on top).\n\n{table(g)}")


def plot(g: dict, out: Path) -> bool:
    """Every motor's operating points on its datasheet torque-speed line (needs matplotlib; skipped without it)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    p = g["_plot"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True, constrained_layout=True)
    point, ink, muted, rule = "#2a6fbb", "#20262e", "#6b7480", "#c9ced6"
    for i, (ax, r) in enumerate(zip(axes.flat, g["motors"])):
        w = np.linspace(0.0, p["v0"][i], 50)
        line = np.minimum(p["sat"][i] * (1.0 - w / p["v0"][i]), p["peak"][i])
        ax.plot(w, line, color=ink, lw=1.6)
        ax.axhline(p["rated"][i], color=muted, lw=1.0, ls=(0, (4, 3)))
        ax.scatter(np.abs(p["dq"][:, i]), np.abs(p["tau"][:, i]), s=5, color=point, alpha=0.35, linewidths=0, rasterized=True)
        ax.set_title(f"{r['label']} (ID {r['can_id']})", fontsize=10, color=ink, loc="left")
        ax.set_ylim(0.0, p["peak"][i] * 1.08)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(color=rule, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("|speed| [rad/s]", fontsize=9, color=muted)
    for ax in axes[:, 0]:
        ax.set_ylabel("|torque| [N·m]", fontsize=9, color=muted)
    fig.suptitle(f"Motor operating points vs datasheet at {g['supply_v']:.0f} V: solid = torque-speed limit, dashed = rated "
                 f"(continuous). Gate: {g['verdict']}", fontsize=11, color=ink, x=0.01, ha="left")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=Path, default=None, help="default: the newest logs/teleop/*.motorlog")
    ap.add_argument("--supply-v", type=float, default=30.0, help="bus voltage for the torque-speed line (rated: 48 V)")
    ap.add_argument("--skip-s", type=float, default=0.0, help="ignore the first seconds of the log")
    a = ap.parse_args()
    path = a.log
    if path is None:
        logs = sorted((ROOT / "logs" / "teleop").glob("*.motorlog"), key=lambda p: p.stat().st_mtime)
        if not logs:
            raise SystemExit("no motor log in logs/teleop")
        path = logs[-1]
    g = gate(path, a.supply_v, a.skip_s)
    md = report(g)
    base = path.with_name(path.stem + "_gate")
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    plotted = plot(g, base.with_suffix(".png"))
    g.pop("_plot")
    base.with_suffix(".json").write_text(json.dumps(g, indent=1, default=float) + "\n", encoding="utf-8")
    print(md, flush=True)
    print(f"[gate] {base.with_suffix('.md')}" + (f"\n[gate] {base.with_suffix('.png')}" if plotted else ""), flush=True)
    return 0 if g["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
