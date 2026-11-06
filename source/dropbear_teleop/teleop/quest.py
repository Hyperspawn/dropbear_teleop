"""Quest 3 (any WebXR headset with controllers) as a teleop device: one web page, grip to move.

The headset opens ``tools/quest_web/index.html`` served by this module. The page streams head and controller poses over
a websocket and shows the robot in the room through passthrough, posed from the bridge's state. Over USB
(``adb reverse tcp:8012 tcp:8012``, then ``http://localhost:8012`` in the headset) WebXR needs no certificate and
nothing listens on the LAN; the server binds 127.0.0.1 by default.

:class:`ClutchMapping` is the operator mapping: hold a grip and that arm follows the *motion* of that hand from where
the arm is now; let go and the arm stays. There is no calibration pose and no start button. The mapping lives in the room:
the robot stands where the page shows it, and its hand moves the way yours does, wherever you stand and look. If you stand
in front of the robot when the first grip closes, it mirrors you: the arm on your side follows, reaching toward it reaches
toward you. The grip is also the dead-man switch: a released grip, a lost controller or stale data all hold the arm, and
it only moves again on a new grip.

Layout of the page's packet (float64): head, left, right 4x4 column-major in the XR reference space (OpenXR basis, NaN
= no pose), then for each controller ``BUTTONS``, then where the page shows the robot (yaw about up [rad], base x, base z
in the XR frame), then the last server time received (for the round trip).
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from dropbear_teleop.robots.dropbear_names import MOTOR_HARD_LIMITS_DEG, MOTOR_NAMES
from dropbear_teleop.robots.hw_motor_specs import (
    ARM_MOTOR_LABELS, HW_PROFILE_MAPS, JOINT_MOTOR_MAPS, MOTOR_CAN_IDS, MOTOR_MODELS, line_at_supply,
)

from .devices import WristTargets
from .frames import to_robot_basis, xr_matrix

REPO = Path(__file__).resolve().parents[3]
WEB_DIR = REPO / "tools" / "quest_web"
GLB_PATH = WEB_DIR / "dropbear.glb"
STAND_NPZ = REPO / "data" / "motions" / "smoke" / "dropbear_static_stand.npz"

ARM_BODIES: tuple[str, ...] = tuple(
    f"{s}_{b}" for s in ("LH", "RH") for b in (
        "RMD_X8_Pro_MIR8_MIR1__1__1", "RMD_X8_Pro_MIR8_MIR1__3__1", "elbow_rear_hanger__1__1", "elbow_rear_hanger_1",
        "bicep_motor_mate_1", "6mm_bearing__9__1", "bicep_1", "6mm_bearing__4__1", "shoulder_ex_al_interface_1"))
"""The 18 bodies that move when only the arms do; the page holds the other 72 at the standing pose."""

BUTTONS = ("grip", "trigger", "a", "b", "stick")
RX_ROBOT = 48 + 2 * len(BUTTONS)  # robot yaw, base x, base z
RX_LEN = RX_ROBOT + 3 + 1
HUD_MOTORS: tuple[str, ...] = tuple(ARM_MOTOR_LABELS)
TX_HEAD = 13  # server time [s], phase, left / right engaged, left / right IK error [mm], left / right EE point (root frame), mirror
TX_BODIES = TX_HEAD + 3 * len(HUD_MOTORS)  # before the bodies: q, dq, torque of each HUD motor
TX_LEN = TX_BODIES + 7 * len(ARM_BODIES)
PHASES = {"waiting": 0.0, "move_in": 1.0, "settle": 2.0, "teleop": 3.0, "home": 4.0}
GRIP_ON, GRIP_OFF = 0.6, 0.4
DRIVER_TIMEOUT_S = 1.0
"""Only one page drives at a time: another one takes over once the driving page has been silent this long."""
MIRROR = np.diag([-1.0, 1.0, 1.0])
"""Reflection through the robot's left-right plane, in the robot frame."""
OTHER_ARM = {"left": "right", "right": "left"}


def rot_z(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def operator_in_front(head: np.ndarray, robot_yaw: float, base_xz) -> bool:
    """Is the head on the robot's front side? ``head`` 4x4 in the robot basis, ``base_xz`` the page's base x, z (XR frame)."""
    to_head = head[:2, 3] - np.array([-base_xz[1], -base_xz[0]])  # XR (x, z) -> robot basis (x, y) = (-z, -x)
    return bool(np.cos(robot_yaw) * to_head[0] + np.sin(robot_yaw) * to_head[1] > 0.0)


class ClutchMapping:
    """Hand motion -> wrist target, relative to where hand and arm were when the grip closed.

    ``hand`` is 4x4 in the robot basis (x forward, y left, z up), XR world frame; the wrist pose is in the torso frame;
    ``robot_yaw`` is where the robot faces in that world (``Rz``). Position: ``p = p_wrist0 + scale * Rz^T (p_hand - p_hand0)``:
    the robot's hand moves the way the operator's does in the room. Orientation: the hand's rotation since the grip is applied
    to the wrist, ``R = (Rz^T R_hand R_hand0^T Rz) R_wrist0``, so the fixed angle between a controller and a robot wrist does
    not matter. ``mirror``: both are reflected through the robot's left-right plane (``M``): ``M Rz^T d`` and ``M (...) M``."""

    def __init__(self, scale: float = 0.70):
        self.scale = float(scale)
        self._held: dict[str, dict] = {}

    def engaged(self, side: str) -> bool:
        return side in self._held

    def engage(self, side: str, hand: np.ndarray, robot_yaw: float, wrist: np.ndarray, mirror: bool = False) -> None:
        self._held[side] = {"yaw": rot_z(robot_yaw), "hand": np.array(hand, dtype=float),
                            "wrist": np.array(wrist, dtype=float), "m": MIRROR if mirror else np.eye(3)}

    def release(self, side: str) -> None:
        self._held.pop(side, None)

    def target(self, side: str, hand: np.ndarray) -> np.ndarray:
        h = self._held[side]
        rz, m = h["yaw"], h["m"]
        out = np.eye(4)
        out[:3, 3] = h["wrist"][:3, 3] + self.scale * m @ rz.T @ (hand[:3, 3] - h["hand"][:3, 3])
        out[:3, :3] = m @ rz.T @ hand[:3, :3] @ h["hand"][:3, :3].T @ rz @ m @ h["wrist"][:3, :3]
        return out


class JointLimiter:
    """Velocity- and acceleration-limited joint command following a target without overshoot.

    Whatever the IK asks for, the command changes speed by at most ``a_max * dt`` per step and brakes in time for the
    target (the speed whose discrete stopping distance ``v^2 / 2a + v dt / 2`` is the remaining error): a hand-tracking
    glitch or a fast IK move becomes a smooth motion."""

    def __init__(self, v_max: float, a_max: float, dt: float, n: int = 10):
        self.v_max, self.a_max, self.dt = float(v_max), float(a_max), float(dt)
        self.v = np.zeros(n)

    def step(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        e = np.asarray(target, dtype=float) - q
        half = 0.5 * self.a_max * self.dt
        brake = np.sqrt(half * half + 2.0 * self.a_max * np.abs(e)) - half
        want = np.sign(e) * np.minimum(np.minimum(self.v_max, brake), np.abs(e) / self.dt)
        self.v = np.clip(want, self.v - self.a_max * self.dt, self.v + self.a_max * self.dt)
        return q + self.v * self.dt

    def stop(self, mask: np.ndarray | None = None) -> None:
        """Forget the velocity (all joints, or where ``mask``): the next step starts from rest."""
        self.v = np.zeros_like(self.v) if mask is None else np.where(mask, 0.0, self.v)


class _Shared:
    """Latest page packet and latest state for the page, shared with the server process."""

    def __init__(self, ctx):
        self.rx = ctx.Array("d", RX_LEN, lock=True)
        self.tx = ctx.Array("d", TX_LEN, lock=True)
        self.meta = ctx.Array("d", 4, lock=True)  # last packet wall time, packets, clients, round trip [ms]
        self.rx[:] = [float("nan")] * RX_LEN
        self.tx[:] = [0.0] * TX_LEN


def stand_payload(stand_npz: Path = STAND_NPZ) -> dict:
    """Every body's pose in the standing pose (world frame, feet on the floor) for the page, once at load."""
    z = np.load(stand_npz, allow_pickle=True)
    return {"names": [str(n) for n in z["body_names"]], "p": np.round(z["body_pos_w"][0], 5).tolist(),
            "q": np.round(z["body_quat_w"][0], 5).tolist(), "arm": list(ARM_BODIES)}


def motors_payload(profile: str = "hw_cem6", supply_v: float = 30.0) -> dict:
    """The datasheet limits the page's motor display draws against (the same numbers as ``tools/teleop_gate.py``)."""
    motor_map = JOINT_MOTOR_MAPS[HW_PROFILE_MAPS[profile]]
    out = []
    for n in HUD_MOTORS:
        m = MOTOR_MODELS[motor_map[n]]
        no_load, sat = line_at_supply(m, supply_v)
        lo, hi = MOTOR_HARD_LIMITS_DEG[n]
        out.append({"name": n, "label": ARM_MOTOR_LABELS[n], "can_id": MOTOR_CAN_IDS[n], "model": m.name, "rated": m.rated_torque,
                    "peak": m.peak_torque, "no_load": no_load, "saturation": sat, "limits_deg": [lo, hi]})
    return {"supply_v": supply_v, "profile": profile, "tx_motors": TX_HEAD, "tx_bodies": TX_BODIES, "motors": out}


def _serve(cfg: dict, shared: _Shared, ready) -> None:
    """The page and websocket server (blocking; its own process, so it never competes with the control loop)."""
    try:
        from aiohttp import WSMsgType, web

        t0 = time.perf_counter()
        driver: dict = {"ws": None, "t": 0.0}  # the page whose packets count: two pages must never fight over the grips
        stand = json.dumps(stand_payload())
        motors = json.dumps({**motors_payload(cfg["motor_profile"], cfg["supply_v"]), "mirror": cfg["allow_mirror"]})

        async def index(_):
            return web.FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})

        async def glb(_):
            return web.FileResponse(GLB_PATH)

        async def stand_json(_):
            return web.Response(text=stand, content_type="application/json")

        async def motors_json(_):
            return web.Response(text=motors, content_type="application/json")

        async def ws_handler(request):
            ws = web.WebSocketResponse(max_msg_size=1 << 16)
            await ws.prepare(request)
            with shared.meta.get_lock():
                shared.meta[2] += 1

            async def sender():
                while not ws.closed:
                    tx = np.array(shared.tx[:], dtype=np.float64)
                    tx[0] = time.perf_counter() - t0
                    await ws.send_bytes(tx.tobytes())
                    await asyncio.sleep(1.0 / cfg["state_hz"])

            task = asyncio.ensure_future(sender())
            try:
                async for msg in ws:
                    if msg.type != WSMsgType.BINARY or len(msg.data) != 8 * RX_LEN:
                        continue
                    if driver["ws"] is not ws and time.time() - driver["t"] < DRIVER_TIMEOUT_S:
                        continue
                    driver.update(ws=ws, t=time.time())
                    pkt = np.frombuffer(msg.data, dtype=np.float64)
                    with shared.meta.get_lock():
                        shared.rx[:] = pkt.tolist()
                        shared.meta[0] = time.time()
                        shared.meta[1] += 1
                        if np.isfinite(pkt[-1]) and pkt[-1] > 0:
                            shared.meta[3] = 1e3 * (time.perf_counter() - t0 - pkt[-1])
            finally:
                task.cancel()
                with shared.meta.get_lock():
                    shared.meta[2] -= 1
            return ws

        app = web.Application()
        app.add_routes([web.get("/", index), web.get("/dropbear.glb", glb), web.get("/stand.json", stand_json),
                        web.get("/motors.json", motors_json), web.get("/ws", ws_handler)])
        ssl_ctx = None
        if cfg.get("cert") and cfg.get("key"):
            import ssl

            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(certfile=cfg["cert"], keyfile=cfg["key"])

        async def run():
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, cfg["host"], cfg["port"], ssl_context=ssl_ctx).start()
            ready.set()
            await asyncio.Event().wait()

        asyncio.run(run())
    except BaseException as e:  # noqa: BLE001
        print(f"[quest] server failed: {e!r}", flush=True)
        ready.set()


class QuestSource:
    """Teleop device: wrist targets from the page, ``None`` for an arm that is not gripped (the loop holds it).

    Set :attr:`robot_wrists` (callable -> side -> torso-frame 4x4, the commanded wrists) and :attr:`armed` (True while
    the loop is in TELEOP; grips do nothing before). Events: ``"stop"`` when both thumbsticks are pressed.

    When no grip is held, the next grip decides :attr:`mirror` from where the operator stands (``allow_mirror``), and both
    hands keep that mode until both are released, so two hands can never ask for the same arm."""

    name = "quest"

    def __init__(self, scale: float = 0.70, host: str = "127.0.0.1", port: int = 8012, cert: str | None = None,
                 key: str | None = None, stale_s: float = 0.25, state_hz: float = 30.0, allow_mirror: bool = True,
                 motor_profile: str = "hw_cem6", supply_v: float = 30.0):
        self.clutch = ClutchMapping(scale)
        self.host, self.port, self.cert, self.key = host, int(port), cert, key
        self.stale_s, self.state_hz = float(stale_s), float(state_hz)
        self.allow_mirror, self.mirror = bool(allow_mirror), False
        self.motor_profile, self.supply_v = motor_profile, float(supply_v)
        self._arm_of: dict[str, str] = {}  # gripping hand -> the arm it drives
        self._hud_slots = [MOTOR_NAMES.index(n) for n in HUD_MOTORS]
        self.robot_wrists = None
        self.armed = False
        self._shared: _Shared | None = None
        self._proc = None
        self._gripping = {"left": False, "right": False}
        self._sticks = False
        self._grips_seen = 0

    def start(self) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._shared = _Shared(ctx)
        ready = ctx.Event()
        cfg = {"host": self.host, "port": self.port, "cert": self.cert, "key": self.key, "state_hz": self.state_hz,
               "motor_profile": self.motor_profile, "supply_v": self.supply_v, "allow_mirror": self.allow_mirror}
        self._proc = ctx.Process(target=_serve, args=(cfg, self._shared, ready), name="quest-page", daemon=True)
        self._proc.start()
        if not ready.wait(timeout=60.0) or not self._proc.is_alive():
            raise RuntimeError(f"quest page server did not start (alive={self._proc.is_alive()})")

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(timeout=3.0)

    @property
    def url(self) -> str:
        return f"{'https' if self.cert else 'http'}://{'localhost' if self.host == '127.0.0.1' else self.host}:{self.port}"

    @property
    def counts(self) -> dict:
        if self._shared is None:
            return {}
        m = self._shared.meta[:]
        return {"packets": int(m[1]), "clients": int(m[2]), "round_trip_ms": round(float(m[3]), 2),
                "grips": self._grips_seen}

    def _update_grip(self, side: str, value: float) -> bool:
        held = self._gripping[side]
        self._gripping[side] = value >= GRIP_ON if not held else value > GRIP_OFF
        return self._gripping[side]

    def get(self, t: float) -> WristTargets:
        with self._shared.meta.get_lock():
            pkt, stamp, packets = np.array(self._shared.rx[:]), self._shared.meta[0], self._shared.meta[1]
        age = time.time() - stamp
        stamp_ns = time.perf_counter_ns() - int(max(age, 0.0) * 1e9)
        fresh = packets > 0 and age <= self.stale_s
        head, robot = pkt[:16], pkt[RX_ROBOT:RX_ROBOT + 3]
        out: dict = {"left": None, "right": None}
        n = len(BUTTONS)
        hands: dict[str, np.ndarray] = {}
        for k, side in enumerate(("left", "right")):
            hand = pkt[16 + 16 * k:32 + 16 * k]
            grip = pkt[48 + n * k]
            usable = self.armed and fresh and all(np.all(np.isfinite(x)) for x in (head, robot, hand))
            if usable and self._update_grip(side, float(grip)):
                hands[side] = to_robot_basis(xr_matrix(hand))
            else:
                self._gripping[side] = False
                if side in self._arm_of:
                    self.clutch.release(self._arm_of.pop(side))
        if hands and not self._arm_of:
            self.mirror = self.allow_mirror and operator_in_front(to_robot_basis(xr_matrix(head)), robot[0], robot[1:])
        for side, hand_r in hands.items():
            if side not in self._arm_of:
                self._arm_of[side] = OTHER_ARM[side] if self.mirror else side
                arm = self._arm_of[side]
                self.clutch.engage(arm, hand_r, float(robot[0]), self.robot_wrists()[arm], mirror=self.mirror)
                self._grips_seen += 1
            out[self._arm_of[side]] = self.clutch.target(self._arm_of[side], hand_r)
        sticks = bool(fresh and pkt[48 + 4] > 0.5 and pkt[48 + n + 4] > 0.5)
        events = ["stop"] if sticks and not self._sticks else []
        self._sticks = sticks
        return WristTargets(t=t, stamp_ns=stamp_ns, left=out["left"], right=out["right"], valid=fresh, events=events)

    def publish(self, phase: str, st=None, ik_res=None, ee: dict | None = None, torso_in_root=None) -> None:
        """Latest state for the page: phase, which arms follow, IK error, end-effector points (``ee``: side -> torso-frame
        4x4; ``torso_in_root``: the torso origin in the root frame), what each arm motor reports, arm body poses relative to
        the root."""
        tx = np.zeros(TX_LEN)
        tx[1] = PHASES.get(phase, 0.0)
        tx[2], tx[3] = float(self.clutch.engaged("left")), float(self.clutch.engaged("right"))
        if ik_res is not None:
            tx[4], tx[5] = 1e3 * ik_res.arms["left"].pos_err_m, 1e3 * ik_res.arms["right"].pos_err_m
        if ee is not None:
            tx[6:9], tx[9:12] = ee["left"][:3, 3] + torso_in_root, ee["right"][:3, 3] + torso_in_root
        tx[12] = float(self.mirror and bool(self._arm_of))
        if st is not None:
            tx[TX_HEAD:TX_BODIES:3] = np.asarray(st.motor.q, dtype=float)[self._hud_slots]
            tx[TX_HEAD + 1:TX_BODIES:3] = np.asarray(st.motor.dq, dtype=float)[self._hud_slots]
            tx[TX_HEAD + 2:TX_BODIES:3] = np.asarray(st.motor.tau_est, dtype=float)[self._hud_slots]
        sim = st.sim if st is not None else None
        if sim is not None and sim.body_names:
            from dropbear_teleop.motion.rotations import quat_conj, quat_mul, quat_to_matrix

            qr_inv = quat_conj(np.asarray(sim.root_quat_w, dtype=float))
            rr, pr = quat_to_matrix(np.asarray(sim.root_quat_w, dtype=float)), np.asarray(sim.root_pos_w, dtype=float)
            for i, name in enumerate(ARM_BODIES):
                if name in sim.body_names:
                    p, q = sim.body_pose(name)
                    tx[TX_BODIES + 7 * i:TX_BODIES + 7 * i + 3] = rr.T @ (np.asarray(p, dtype=float) - pr)
                    tx[TX_BODIES + 7 * i + 3:TX_BODIES + 7 * i + 7] = quat_mul(qr_inv, np.asarray(q, dtype=float))
        self._shared.tx[:] = tx.tolist()
