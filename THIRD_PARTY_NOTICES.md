# Third-party code

This project's own code, data and robot model are licensed for **noncommercial use only** under the PolyForm Noncommercial
License 1.0.0 (`LICENSE`); for commercial use, contact Hyperspawn ([hyperspawn.co](https://hyperspawn.co), priyanshu@hyperspawn.co).
It builds on the following; their own licenses apply to those parts.

## In this repository

| component | where | license | notes |
|---|---|---|---|
| three.js 0.160.0 | `tools/quest_web/vendor/three/` | MIT (`LICENSE` there) | the page's 3D view; unmodified files from the npm package |
| televuer (Unitree) | `source/dropbear_teleop/teleop/` | MIT | adapted constants / transforms (`third_party/televuer_NOTICE.md`) |

## Installed separately (`requirements/`)

NVIDIA Newton, Warp, MuJoCo Warp (Apache-2.0), MuJoCo (Apache-2.0) and the other pinned Python packages.
