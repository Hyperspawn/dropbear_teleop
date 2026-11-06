"""CPU tests of the Quest teleop device (``dropbear_teleop.teleop.quest``): the grip clutch and the page packet."""
from __future__ import annotations

import math

import numpy as np
import pytest

import sdk_test_paths  # noqa: F401  (adds source/ and third_party/pydeps)

from dropbear_teleop.teleop.frames import T_OPENXR_ROBOT, T_ROBOT_OPENXR, xr_flat  # noqa: E402
from dropbear_teleop.teleop.quest import (  # noqa: E402
    ARM_BODIES, BUTTONS, HUD_MOTORS, RX_LEN, RX_ROBOT, STAND_NPZ, TX_BODIES, TX_HEAD, TX_LEN, ClutchMapping, QuestSource,
    motors_payload, operator_in_front, stand_payload,
)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def pose(p=(0.0, 0.0, 0.0), r=None) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = p
    if r is not None:
        m[:3, :3] = r
    return m


HEAD = pose((0.0, 0.0, 1.6))
YAW = 0.0
WRIST = pose((0.3, 0.2, 0.3), rot_y(0.4))
WRIST_R = pose((0.3, -0.2, 0.3), rot_y(0.4))
HAND = pose((0.4, 0.2, 1.2), rot_z(0.7) @ rot_y(-0.3))


def test_engaging_does_not_move_the_arm():
    c = ClutchMapping(0.7)
    c.engage("left", HAND, YAW, WRIST)
    np.testing.assert_allclose(c.target("left", HAND), WRIST, atol=1e-12)


def test_position_follows_scaled_hand_motion():
    c = ClutchMapping(0.5)
    c.engage("left", HAND, YAW, WRIST)
    moved = HAND.copy()
    moved[:3, 3] += [0.1, -0.2, 0.04]
    np.testing.assert_allclose(c.target("left", moved)[:3, 3], WRIST[:3, 3] + 0.5 * np.array([0.1, -0.2, 0.04]))


@pytest.mark.parametrize("yaw", [0.0, 0.7, math.pi / 2, math.pi, -2.0])
def test_the_robots_hand_moves_the_way_yours_does_in_the_room(yaw):
    """Whatever way the robot faces: the wrist's motion, seen in the room, is the hand's motion."""
    c = ClutchMapping(1.0)
    c.engage("left", HAND, yaw, WRIST)
    d = np.array([0.1, -0.05, 0.08])
    np.testing.assert_allclose(rot_z(yaw) @ (c.target("left", _moved(HAND, d))[:3, 3] - WRIST[:3, 3]), d, atol=1e-12)


@pytest.mark.parametrize("yaw", [0.0, math.pi, 0.7])
def test_rotation_seen_in_the_room_is_the_hands(yaw):
    """A room-frame rotation of the hand turns the wrist by the same room-frame rotation, whatever way the robot faces."""
    c = ClutchMapping(1.0)
    c.engage("left", HAND, yaw, WRIST)
    d = rot_y(0.3) @ rot_z(0.5)
    moved = HAND.copy()
    moved[:3, :3] = d @ HAND[:3, :3]
    rz = rot_z(yaw)
    np.testing.assert_allclose(rz @ c.target("left", moved)[:3, :3], d @ rz @ WRIST[:3, :3], atol=1e-12)


def test_mirror_when_facing_the_operator():
    """Robot faces the operator (yaw pi). Reach toward it: its hand comes toward you. Move to your right: its hand goes to
    your right too. Up is up."""
    c = ClutchMapping(1.0)
    c.engage("left", HAND, math.pi, WRIST, mirror=True)

    def room(d):
        return rot_z(math.pi) @ (c.target("left", _moved(HAND, d))[:3, 3] - WRIST[:3, 3])

    np.testing.assert_allclose(room([0.1, 0.0, 0.0]), [-0.1, 0.0, 0.0], atol=1e-12)  # world +x: toward the robot
    np.testing.assert_allclose(room([0.0, -0.1, 0.0]), [0.0, -0.1, 0.0], atol=1e-12)  # world -y: the operator's right
    np.testing.assert_allclose(room([0.0, 0.0, 0.1]), [0.0, 0.0, 0.1], atol=1e-12)


def test_mirror_keeps_a_proper_rotation_and_turns_the_other_way():
    c = ClutchMapping(1.0)
    c.engage("left", HAND, 0.0, WRIST, mirror=True)
    moved = HAND.copy()
    moved[:3, :3] = rot_z(0.5) @ HAND[:3, :3]
    r = c.target("left", moved)[:3, :3]
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    np.testing.assert_allclose(r, rot_z(-0.5) @ WRIST[:3, :3], atol=1e-12)  # about up: the mirror image turns the other way


def _moved(m: np.ndarray, d) -> np.ndarray:
    out = m.copy()
    out[:3, 3] += d
    return out


def test_operator_in_front_is_about_the_robots_front_not_the_head_direction():
    # robot 1 m ahead of the operator (XR z = -1): facing away, then facing the operator; the head's own yaw plays no part
    for head in (pose((0.0, 0.0, 1.6)), pose((0.0, 0.0, 1.6), rot_z(2.0))):
        assert not operator_in_front(head, 0.0, (0.0, -1.0))
        assert operator_in_front(head, math.pi, (0.0, -1.0))


def test_rotation_is_the_hand_rotation_seen_by_the_operator():
    """A world-frame rotation of the hand, whatever the hand's own axes, turns the wrist by the same world rotation."""
    c = ClutchMapping(0.7)
    c.engage("left", HAND, YAW, WRIST)
    d = rot_z(0.5)
    moved = HAND.copy()
    moved[:3, :3] = d @ HAND[:3, :3]
    np.testing.assert_allclose(c.target("left", moved)[:3, :3], d @ WRIST[:3, :3], atol=1e-12)


def test_release_and_regrip_restart_from_the_arm():
    c = ClutchMapping(1.0)
    c.engage("left", HAND, YAW, WRIST)
    assert c.engaged("left") and not c.engaged("right")
    c.release("left")
    assert not c.engaged("left")
    far = HAND.copy()
    far[:3, 3] += [0.5, 0.5, 0.5]
    c.engage("left", far, YAW, WRIST)
    np.testing.assert_allclose(c.target("left", far), WRIST, atol=1e-12)


def _xr(m_robot: np.ndarray) -> list[float]:
    return xr_flat(T_OPENXR_ROBOT @ m_robot @ T_ROBOT_OPENXR)


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeArray(list):
    def get_lock(self):
        return _FakeLock()


def _source() -> tuple:
    """A QuestSource with its shared memory faked in-process (no server)."""
    import time
    from types import SimpleNamespace

    q = QuestSource(scale=1.0)
    q._shared = SimpleNamespace(rx=_FakeArray([float("nan")] * RX_LEN), tx=_FakeArray([0.0] * TX_LEN),
                                meta=_FakeArray([0.0, 0.0, 0.0, 0.0]))
    q.armed = True
    q.robot_wrists = lambda: {"left": WRIST, "right": WRIST_R}

    def send(left=HAND, right=HAND, grips=(0.0, 0.0), sticks=(0.0, 0.0), age=0.0, head=HEAD, robot=(0.0, 0.0, -1.0)):
        n = len(BUTTONS)
        pkt = np.zeros(RX_LEN)
        pkt[:16] = _xr(head)
        pkt[16:32] = _xr(left) if left is not None else np.nan
        pkt[32:48] = _xr(right) if right is not None else np.nan
        pkt[48], pkt[48 + n] = grips
        pkt[48 + 4], pkt[48 + n + 4] = sticks
        pkt[RX_ROBOT:RX_ROBOT + 3] = robot
        q._shared.rx[:] = pkt.tolist()
        q._shared.meta[0] = time.time() - age
        q._shared.meta[1] += 1
    return q, send


def test_only_the_gripped_arm_gets_a_target():
    q, send = _source()
    send(grips=(0.9, 0.0))
    out = q.get(0.0)
    assert out.left is not None and out.right is None and out.valid
    np.testing.assert_allclose(out.left, WRIST, atol=1e-9)


def test_grip_hysteresis():
    q, send = _source()
    send(grips=(0.5, 0.0))
    assert q.get(0.0).left is None  # below the closing threshold
    send(grips=(0.7, 0.0))
    assert q.get(0.0).left is not None
    send(grips=(0.5, 0.0))
    assert q.get(0.0).left is not None  # above the opening threshold: still held
    send(grips=(0.3, 0.0))
    assert q.get(0.0).left is None


def test_nothing_moves_until_armed():
    q, send = _source()
    q.armed = False
    send(grips=(1.0, 1.0))
    out = q.get(0.0)
    assert out.left is None and out.right is None and not q.clutch.engaged("left")


def test_stale_data_releases_and_needs_a_new_engage():
    q, send = _source()
    send(grips=(1.0, 0.0))
    assert q.get(0.0).left is not None
    send(grips=(1.0, 0.0), age=1.0)
    out = q.get(0.0)
    assert out.left is None and not out.valid and not q.clutch.engaged("left")


def test_lost_controller_releases_that_arm_only():
    q, send = _source()
    send(grips=(1.0, 1.0))
    q.get(0.0)
    send(left=None, grips=(1.0, 1.0))
    out = q.get(0.0)
    assert out.left is None and out.right is not None and out.valid


def test_both_sticks_stop_once():
    q, send = _source()
    send(sticks=(1.0, 0.0))
    assert q.get(0.0).events == []
    send(sticks=(1.0, 1.0))
    assert q.get(0.0).events == ["stop"]
    send(sticks=(1.0, 1.0))
    assert q.get(0.0).events == []


FACING = (math.pi, 0.0, -1.0)  # 1 m ahead of the operator at the XR origin, facing the operator


def test_behind_the_robot_each_hand_drives_its_own_arm():
    q, send = _source()
    send(grips=(1.0, 0.0))
    out = q.get(0.0)
    assert out.left is not None and out.right is None and not q.mirror


def test_in_front_of_the_robot_each_hand_drives_the_arm_on_its_side():
    q, send = _source()
    send(grips=(1.0, 0.0), robot=FACING)  # the operator's left hand
    out = q.get(0.0)
    assert q.mirror and out.left is None
    np.testing.assert_allclose(out.right, WRIST_R, atol=1e-9)  # no jump: it starts from the robot's right wrist
    send(grips=(1.0, 1.0), robot=FACING)
    out = q.get(0.0)
    np.testing.assert_allclose(out.left, WRIST, atol=1e-9)


def test_mode_holds_until_both_hands_let_go():
    """Walking around the robot while one hand holds cannot make two hands ask for one arm."""
    q, send = _source()
    send(grips=(1.0, 0.0))
    q.get(0.0)
    send(grips=(1.0, 1.0), robot=FACING)  # the operator now stands in front; the left hand still holds
    out = q.get(0.0)
    assert not q.mirror and out.left is not None and out.right is not None
    send(grips=(0.0, 0.0), robot=FACING)
    assert q.get(0.0).left is None
    send(grips=(1.0, 0.0), robot=FACING)
    assert q.get(0.0).right is not None and q.mirror


def test_no_mirror_option():
    q, send = _source()
    q.allow_mirror = False
    send(grips=(1.0, 0.0), robot=FACING)
    out = q.get(0.0)
    assert not q.mirror and out.left is not None


def test_page_without_a_robot_pose_moves_nothing():
    q, send = _source()
    send(grips=(1.0, 1.0), robot=(float("nan"),) * 3)
    out = q.get(0.0)
    assert out.left is None and out.right is None


def test_publish_sends_each_motor_where_the_page_reads_it():
    from types import SimpleNamespace

    from dropbear_teleop.robots.dropbear_names import MOTOR_NAMES

    q, _ = _source()
    n = len(MOTOR_NAMES)
    st = SimpleNamespace(motor=SimpleNamespace(q=np.arange(n, dtype=float), dq=100.0 + np.arange(n), tau_est=200.0 + np.arange(n)),
                         sim=None)
    q.publish("teleop", st)
    tx = np.array(q._shared.tx[:])
    i = HUD_MOTORS.index("RH_elbow_joint")
    slot = MOTOR_NAMES.index("RH_elbow_joint")
    assert list(tx[TX_HEAD + 3 * i:TX_HEAD + 3 * i + 3]) == [slot, 100.0 + slot, 200.0 + slot]
    assert TX_BODIES == TX_HEAD + 30 and len(tx) == TX_LEN


def test_motors_payload_uses_the_supply():
    p = motors_payload("hw_cem6", 30.0)
    m = {x["name"]: x for x in p["motors"]}
    assert m["LH_yaw"]["model"] == "RMD-X10-V3-1:7" and m["LH_yaw"]["can_id"] == 35 and m["LH_yaw"]["peak"] == 40.0
    assert m["RH_wrist_roll"]["no_load"] == pytest.approx(16.7552 * 30.0 / 48.0)
    assert p["tx_bodies"] == TX_BODIES


def test_only_one_page_drives():
    """Two pages sending at once: the second is ignored until the first has gone silent."""
    aiohttp = pytest.importorskip("aiohttp")
    import asyncio
    import socket

    from dropbear_teleop.teleop.quest import DRIVER_TIMEOUT_S

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    q = QuestSource(port=port)
    q.start()

    async def scenario():
        async with aiohttp.ClientSession() as http:
            first = await http.ws_connect(f"ws://127.0.0.1:{port}/ws")
            second = await http.ws_connect(f"ws://127.0.0.1:{port}/ws")

            async def send(ws, value):
                await ws.send_bytes(np.full(RX_LEN, value).tobytes())
                await asyncio.sleep(0.1)
                return q._shared.rx[0]

            assert await send(first, 1.0) == 1.0
            assert await send(second, 2.0) == 1.0, "the second page must not count while the first is driving"
            assert await send(first, 3.0) == 3.0
            await asyncio.sleep(DRIVER_TIMEOUT_S + 0.1)
            assert await send(second, 4.0) == 4.0, "the second page takes over once the first has gone silent"
            assert await send(first, 5.0) == 4.0

    try:
        asyncio.run(scenario())
    finally:
        q.close()


def test_stand_payload_has_every_arm_body():
    if not STAND_NPZ.exists():
        pytest.skip("standing NPZ not present")
    s = stand_payload()
    assert len(s["names"]) == len(s["p"]) == len(s["q"]) == 90
    assert set(ARM_BODIES) <= set(s["names"]) and len(ARM_BODIES) == 18
    assert "world" in s["names"]


def test_limiter_respects_velocity_and_acceleration_and_lands():
    from dropbear_teleop.teleop.quest import JointLimiter

    lim = JointLimiter(4.0, 30.0, 0.02, n=1)
    q, target = np.zeros(1), np.array([1.0])
    qs = [q.copy()]
    for _ in range(200):
        q = lim.step(q, target)
        qs.append(q.copy())
    qs = np.array(qs)[:, 0]
    v = np.diff(qs) / 0.02
    assert np.abs(v).max() <= 4.0 + 1e-9
    assert np.abs(np.diff(v) / 0.02).max() <= 30.0 + 1e-6
    assert qs.max() <= 1.0 + 2e-3, "overshoot only from the time step (< 2 mrad)"
    assert abs(qs[-1] - 1.0) < 1e-6


def test_tool_point_jacobian_and_shell():
    """The IK's tool point: FK along the forearm, an exact Jacobian, a thicker reachable shell."""
    from dropbear_teleop.teleop.arm_ik import DropbearArmIK

    plain, tool = DropbearArmIK(None), DropbearArmIK(None, tool_m=0.2)
    ch0, ch = plain.chains["left"], tool.chains["left"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(ch.lower, ch.upper)
        p0, r0 = ch0.fk(q)
        p, r, jac = ch.fk(q, jacobian=True)
        np.testing.assert_allclose(p, p0 + 0.2 * r0[:, 0], atol=1e-12)
        for i in range(5):
            dq = np.zeros(5)
            dq[i] = 1e-6
            num = (ch.fk(q + dq)[0] - ch.fk(q - dq)[0]) / 2e-6
            np.testing.assert_allclose(jac[:3, i], num, atol=2e-3)  # 2e-3: the elbow table is piecewise linear
    assert ch.shell[1] - ch.shell[0] > 2.0 * (ch0.shell[1] - ch0.shell[0])
