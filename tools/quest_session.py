"""One command: Quest 3 drives Dropbear's arms in simulation.

Starts the simulated robot (``tools/newton_bridge.py``: hanging, real time, one CPU core, the real arm motors), the
teleop client (``tools/teleop_arm.py --device quest``), and maps the headset's ``localhost:8012`` to this PC over USB
(``adb reverse``), so no certificate is needed and nothing listens on the network.

    .venv-teleop/Scripts/python.exe tools/quest_session.py

Headset: USB-C cable to the PC, developer mode on, allow USB debugging when asked. Open the Quest browser at
http://localhost:8012 and press "Enter passthrough". The same page on the PC (http://localhost:8012) shows the robot too.
Hold a grip to move that arm; press both sticks in to stop. Ctrl-C here also stops. ``--record NAME`` saves the session.

Every session writes a motor log, and when it stops ``tools/teleop_gate.py`` judges it against the datasheets. If this
window is closed instead, the log is still there: run ``python tools/teleop_gate.py``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source")]

from dropbear_teleop.teleop.quest import ARM_BODIES  # noqa: E402

NEWTON_PY = ROOT / ".venv-newton" / "Scripts" / "python.exe"
TELEOP_PY = ROOT / ".venv-teleop" / "Scripts" / "python.exe"


def find_adb() -> str | None:
    sdk = Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe"
    return shutil.which("adb") or (str(sdk) if sdk.exists() else None)


def link_headset(port: int) -> bool:
    """``adb reverse`` if a headset is connected and allowed. Returns True when the headset can reach this PC."""
    adb = find_adb()
    if adb is None:
        print("[session] adb not found: the PC view still works, the headset needs Android platform-tools", flush=True)
        return False
    out = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=20).stdout
    states = [ln.split()[1] for ln in out.splitlines()[1:] if len(ln.split()) >= 2]
    if "device" not in states:
        why = ("allow USB debugging in the headset" if "unauthorized" in states else
               "connect the USB-C cable and turn developer mode on")
        print(f"[session] no headset on adb ({why}). Running anyway; re-run this command when it is connected.", flush=True)
        return False
    subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"], check=True, timeout=20, capture_output=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--motor-profile", default="hw_cem6", help="arms: X10 1:7 shoulder pitch, X8 Pro the rest, as built")
    ap.add_argument("--xr-scale", type=float, default=1.0, help="robot hand motion / your hand motion")
    ap.add_argument("--max-joint-acc", type=float, default=30.0, help="semantic joint acceleration limit [rad/s^2]")
    ap.add_argument("--max-joint-vel", type=float, default=4.0, help="semantic joint velocity limit [rad/s]")
    ap.add_argument("--stale-s", type=float, default=0.25, help="headset data older than this holds the arms")
    ap.add_argument("--supply-v", type=float, default=30.0, help="bench supply voltage for the motor display and the gate")
    ap.add_argument("--no-mirror", action="store_true", help="never mirror, even when standing in front of the robot")
    ap.add_argument("--record", default=None, help="save the session under data/teleop/<date>_<name>")
    ap.add_argument("--gpu", action="store_true", help="MuJoCo Warp on the GPU instead of one CPU core")
    a = ap.parse_args()
    (ROOT / "logs" / "teleop").mkdir(parents=True, exist_ok=True)
    motor_log = ROOT / "logs" / "teleop" / f"quest_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}.motorlog"

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    bridge = [str(NEWTON_PY), "-u", str(ROOT / "tools" / "newton_bridge.py"), "--fixed-base", "--realtime", "--duration", "0",
              "--motor-profile", a.motor_profile, "--sim-bodies", ",".join(ARM_BODIES), "--wait-timeout-s", "900",
              "--motor-log", str(motor_log)]
    bridge_env = dict(env)
    if not a.gpu:
        bridge += ["--device", "cpu", "--mujoco-cpu", "--no-gpu-lock"]
        bridge_env["CUDA_VISIBLE_DEVICES"] = "-1"
    teleop = [str(TELEOP_PY), "-u", str(ROOT / "tools" / "teleop_arm.py"), "--device", "quest", "--clock", "wall",
              "--duration", "0", "--xr-port", str(a.port), "--xr-scale", str(a.xr_scale),
              "--max-joint-vel", str(a.max_joint_vel), "--max-joint-acc", str(a.max_joint_acc), "--xr-stale-s", str(a.stale_s), "--connect-timeout-s", "300",
              "--xr-motor-profile", a.motor_profile, "--xr-supply-v", str(a.supply_v),
              "--summary", str(ROOT / "logs" / "teleop" / "quest_session.json")]
    if a.no_mirror:
        teleop.append("--xr-no-mirror")
    if a.record:
        teleop += ["--record", str(ROOT / "data" / "teleop" / f"{dt.datetime.now():%Y-%m-%d_%H-%M}_{a.record}"),
                   "--task", f"dropbear arm teleop ({a.record})"]

    linked = link_headset(a.port)
    log = open(ROOT / "logs" / "teleop" / "quest_session_bridge.log", "w", encoding="utf-8")
    print("[session] starting the simulated robot (about 30 s)...", flush=True)
    p_bridge = subprocess.Popen(bridge, cwd=ROOT, env=bridge_env, stdout=log, stderr=subprocess.STDOUT)
    p_teleop = subprocess.Popen(teleop, cwd=ROOT, env=env)
    time.sleep(2.0)
    print(f"[session] PC view: http://localhost:{a.port}", flush=True)
    if linked:
        print(f"[session] Headset: open the browser at http://localhost:{a.port} and press Enter passthrough", flush=True)
    rc = 0
    try:
        while p_teleop.poll() is None:
            if p_bridge.poll() is not None:
                print(f"[session] the simulator stopped (rc {p_bridge.returncode}); see logs/teleop/quest_session_bridge.log",
                      flush=True)
                rc = 1
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    for p in (p_teleop, p_bridge):
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    log.close()
    print("[session] stopped", flush=True)
    if motor_log.exists() and motor_log.stat().st_size:
        print(f"[session] motors against the datasheets at {a.supply_v:.0f} V:", flush=True)
        # the system Python first: it usually has matplotlib, for the torque-speed picture (the gate exits 0 pass, 2 fail)
        for py in filter(None, (shutil.which("python"), sys.executable)):
            gate = subprocess.run([py, "-u", str(ROOT / "tools" / "teleop_gate.py"), str(motor_log), "--supply-v", str(a.supply_v)],
                                  cwd=ROOT, env=env)
            if gate.returncode in (0, 2):
                break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
