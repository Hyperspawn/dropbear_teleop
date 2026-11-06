"""Simulated Quest for testing ``tools/teleop_arm.py --device quest`` without a headset: the page's websocket packets.

Script (seconds since connect): both grips close at ``--grip-at`` and the hands move on a figure-8; the left grip opens
at ``--release-at`` while the left hand keeps moving (the left arm must stay put), then closes again at ``--regrip-at``
with the hand somewhere else (the arm must not jump); both thumbsticks at ``--stop-at``. The head yaws throughout (it
must not move the arms). ``--robot-yaw`` turns the robot where the page would show it (pi: it faces the operator, who is
then in front of it, so it mirrors).

    .venv-teleop/Scripts/python.exe tools/sim_quest_client.py --port 8012 --duration 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "source")]

import numpy as np  # noqa: E402

from dropbear_teleop.teleop.frames import T_OPENXR_ROBOT, T_ROBOT_OPENXR, xr_flat  # noqa: E402
from dropbear_teleop.teleop.quest import BUTTONS, RX_LEN, RX_ROBOT  # noqa: E402


def pose_xr(p_robot, yaw: float = 0.0) -> list[float]:
    """A pose given in the robot basis -> the page's column-major OpenXR matrix."""
    c, s = math.cos(yaw), math.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = p_robot
    return xr_flat(T_OPENXR_ROBOT @ m @ T_ROBOT_OPENXR)


async def run(a) -> dict:
    import aiohttp

    url = f"ws://{a.host}:{a.port}/ws"
    stats: dict = {"sent": 0, "received": 0, "url": url}
    deadline = time.time() + a.connect_timeout_s
    while True:
        session = aiohttp.ClientSession()
        try:
            ws = await session.ws_connect(url)
            break
        except Exception:  # noqa: BLE001 - server not up yet
            await session.close()
            if time.time() > deadline:
                raise
            await asyncio.sleep(0.5)
    t0 = time.time()
    server_t = 0.0
    n = len(BUTTONS)
    try:
        while (t := time.time() - t0) <= a.duration:
            w = 2 * math.pi / a.period
            head = pose_xr([0.0, 0.0, 1.65], a.head_yaw_amp * math.sin(2 * math.pi * t / 7.0))
            pkt = np.full(RX_LEN, 0.0)
            pkt[:16] = head
            for k, sgn in enumerate((1.0, -1.0)):
                u = max(0.0, min(1.0, (t - a.grip_at) / 2.0))
                off = u * np.array([a.amp[0] * math.sin(0.5 * w * t), sgn * a.amp[1] * math.sin(w * t),
                                    a.amp[2] * math.sin(2 * w * t)])
                pkt[16 + 16 * k:32 + 16 * k] = pose_xr(np.array([0.30, sgn * 0.20, 1.20]) + off)
                held = t >= a.grip_at and not (k == 0 and a.release_at <= t < a.regrip_at)
                pkt[48 + n * k] = 1.0 if held else 0.0
                pkt[48 + n * k + 4] = 1.0 if a.stop_at <= t < a.stop_at + 0.3 else 0.0
            pkt[RX_ROBOT:RX_ROBOT + 3] = a.robot_yaw, a.base_xz[0], a.base_xz[1]
            pkt[-1] = server_t
            try:
                await ws.send_bytes(pkt.tobytes())
                stats["sent"] += 1
                msg = await ws.receive(timeout=0.001)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    server_t = float(np.frombuffer(msg.data, dtype=np.float64)[0])
                    stats["received"] += 1
            except asyncio.TimeoutError:
                pass
            except (ConnectionResetError, aiohttp.ClientError) as e:
                stats["closed_by_server_at_s"], stats["close_reason"] = round(t, 3), repr(e)
                break
            await asyncio.sleep(1.0 / a.hz)
    finally:
        await ws.close()
        await session.close()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--hz", type=float, default=72.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--period", type=float, default=6.0)
    ap.add_argument("--amp", type=float, nargs=3, default=(0.03, 0.07, 0.06), help="operator motion [m]")
    ap.add_argument("--head-yaw-amp", type=float, default=0.4, help="head yaw [rad] (must not move the arms)")
    ap.add_argument("--robot-yaw", type=float, default=0.0, help="where the page shows the robot facing [rad]")
    ap.add_argument("--base-xz", type=float, nargs=2, default=(0.0, -1.0), help="where the page shows the robot (XR x, z) [m]")
    ap.add_argument("--grip-at", type=float, default=5.0)
    ap.add_argument("--release-at", type=float, default=12.0)
    ap.add_argument("--regrip-at", type=float, default=15.0)
    ap.add_argument("--stop-at", type=float, default=18.0)
    ap.add_argument("--connect-timeout-s", type=float, default=120.0)
    ap.add_argument("--summary", type=Path, default=None)
    a = ap.parse_args()
    stats = asyncio.run(run(a))
    print(json.dumps(stats), flush=True)
    if a.summary:
        a.summary.write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
