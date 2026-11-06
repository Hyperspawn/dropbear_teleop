# dropbear_teleop

Drive the arms of [Dropbear](https://hyperspawn.co), an open 3D-printed humanoid, with a Meta Quest 3. The robot stands in your
room through passthrough. Hold a grip and that arm follows your hand; let go and it stays. Every arm motor is shown live against
its datasheet, and every session is judged against the datasheets when it ends.

**Status:** this runs in simulation, with models of the motors built into the robot. It has not been run on real motors yet.
The same motor log format is meant for the real CAN bridge, so the same gate will judge real motors.

Everything it needs is in this repository: code, calibration, the robot model (421 MB, through [Git LFS](https://git-lfs.com)) and
the page's 3D library. Only the pinned Python packages are installed from `requirements/`.

## Run it

Windows, Python 3.12, [uv](https://docs.astral.sh/uv/), Git LFS. Once:

```bash
git lfs install
git clone https://github.com/Hyperspawn/dropbear_teleop
cd dropbear_teleop
uv venv --python 3.12 .venv-newton
uv pip install --python .venv-newton/Scripts/python.exe -r requirements/newton-windows.txt
uv venv --python 3.12 .venv-teleop
uv pip install --python .venv-teleop/Scripts/python.exe -r requirements/teleop.txt
```

Headset, once: turn on developer mode (Meta Horizon phone app > your headset > Developer Mode), connect a USB-C data cable to the
PC, and allow USB debugging in the headset. You need `adb` (Android platform-tools).

Then, every time:

```bash
.venv-teleop/Scripts/python.exe tools/quest_session.py
```

In the headset browser open `http://localhost:8012` and press **Enter passthrough**. The connection is USB only (`adb reverse`), so
nothing listens on your network, no certificate is needed, and the headset needs no internet. The same address on the PC shows the
robot and the motors without a headset.

## Controls

| Control | What it does |
|---|---|
| Hold a grip | That arm follows your hand, from where the arm is now. Let go and it stays. This is also the dead-man switch. |
| Stand in front of the robot | It mirrors you: the arm on your side follows, and reaching toward it reaches toward you. `--no-mirror` turns this off. |
| Left stick / right stick | Move / turn the robot in the room, while no grip is held |
| A or X | Show or hide the motor panel |
| Both sticks pressed in | Stop. The arms go home and the gate runs. |

The ball at each hand is the tracked point: grey idle, green following, red out of reach (let go and re-grip). A lost controller or
headset data older than 0.25 s holds the arm until a new grip. Only one page drives at a time.

## Motors against the datasheets

Every session writes `logs/teleop/quest_<date>.motorlog`. When it stops, `tools/teleop_gate.py` prints a verdict per motor and saves
a chart of its operating points on the datasheet torque-speed line. If the window was closed instead, the log is intact:

```bash
.venv-teleop/Scripts/python.exe tools/teleop_gate.py
```

The gate asks two things, at the voltage you give it (`--supply-v`, default 30 V; the motors are rated at 48 V):

- **Possible?** RMS torque against rated, demand above the torque-speed line, speed against no-load, load on the joint stops.
- **Stable?** Fast torque ripple, and joint speed while the target is still.

A pass on a simulated log means the motion fits the datasheets with the CAD's masses. It does not prove the real arm: friction,
backlash, printed-part flex, real masses and how the motor firmware applies its gains are not in it. Results and open points are
in [docs/TELEOP.md](docs/TELEOP.md), section 0.

## Test it without a headset

```bash
.venv-teleop/Scripts/python.exe -m pytest tests -q
```

```bash
.venv-teleop/Scripts/python.exe scripts/verify_teleop.py --quest-sim --tag test --no-record --duration 22 --bridge-extra "--motor-profile hw_cem6"
```

The second command runs the simulated robot with simulated Quest packets (`tools/sim_quest_client.py`) and writes its motor log to
`logs/teleop/quest_test.motorlog`. Two tests of the simulator's kernels skip here because they need Warp; run them with
`.venv-newton/Scripts/python.exe -m pytest tests/test_newton_bridge_units.py -q`.

## What is here

| Path | What it is |
|---|---|
| `tools/quest_session.py` | The one command: simulated robot, teleop, `adb reverse`, gate at the end |
| `tools/quest_web/` | The page the headset opens, with the robot's 3D model and three.js |
| `source/dropbear_teleop/teleop/quest.py` | The Quest device: grip clutch, mirror, joint limiter, page server |
| `tools/teleop_arm.py`, `source/dropbear_teleop/teleop/` | Teleop loop, arm IK, gravity feed-forward, recorder |
| `tools/newton_bridge.py`, `source/dropbear_teleop/newton_sim/` | The simulated robot, speaking the same commands a real robot would |
| `source/dropbear_teleop/sdk/motor_log.py`, `tools/teleop_gate.py` | The motor log and the gate |
| `assets/dropbear.usd`, `data/` | The robot model; calibration, motor data, standing pose |
| `docs/TELEOP.md`, `docs/CONTRACTS.md` | Details and results; the motor and joint contract |

This started as a copy of parts of [dropbear-wbc](https://github.com/Hyperspawn/dropbear-wbc) at commit `7315226` and does not depend
on it. The older Vuer client (`--device webxr`) is still in the code; the Quest path above does not use it.

## License

Noncommercial use only: PolyForm Noncommercial 1.0.0 (`LICENSE`). For commercial use, contact Hyperspawn
([hyperspawn.co](https://hyperspawn.co), priyanshu@hyperspawn.co). Third-party parts keep their own licenses
(`THIRD_PARTY_NOTICES.md`).
