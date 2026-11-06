# Third-party code

This project's own code, data and robot model are licensed for **noncommercial use only** under the PolyForm Noncommercial
License 1.0.0 (`LICENSE`); for commercial use, contact Hyperspawn ([hyperspawn.co](https://hyperspawn.co), priyanshu@hyperspawn.co).
It builds on the following; their own licenses apply to those parts.

## Adapted code (in this repository)

| component | where | license | notes |
|---|---|---|---|
| televuer (Unitree) | `source/dropbear_teleop/teleop/` | MIT | adapted constants / transforms (`third_party/televuer_NOTICE.md`) |

## Used but not in this repository

| component | license | used for |
|---|---|---|
| NVIDIA Newton, Warp, MuJoCo Warp | Apache-2.0 | the simulated robot |
| MuJoCo | Apache-2.0 | the simulated robot |
| three.js (loaded by the page from jsDelivr) | MIT | the page's 3D view |
| `tools/newton_dropbear_asset.py` from [robit-man/dropbear_control](https://github.com/robit-man/dropbear_control), cloned separately | see that repository | prepares the USD stage for Newton |
