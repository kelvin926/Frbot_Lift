from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdPhysics


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect selected joints from a USD asset.")
    parser.add_argument("--usd", required=True, help="USD/USDa path.")
    parser.add_argument("--artifact", required=True, help="Path to write JSON output.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {args.usd}")

    rows: list[dict] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        if not any(key in name for key in ("steer", "Wheel", "Link_")):
            continue
        row = {
            "path": str(prim.GetPath()),
            "name": name,
            "type": prim.GetTypeName(),
            "lower_limit": prim.GetAttribute("physics:lowerLimit").Get(),
            "upper_limit": prim.GetAttribute("physics:upperLimit").Get(),
            "drive_type": prim.GetAttribute("drive:angular:physics:type").Get(),
            "drive_target_pos": prim.GetAttribute("drive:angular:physics:targetPosition").Get(),
            "drive_target_vel": prim.GetAttribute("drive:angular:physics:targetVelocity").Get(),
            "drive_stiffness": prim.GetAttribute("drive:angular:physics:stiffness").Get(),
            "drive_damping": prim.GetAttribute("drive:angular:physics:damping").Get(),
            "drive_max_force": prim.GetAttribute("drive:angular:physics:maxForce").Get(),
        }
        rows.append(row)

    Path(args.artifact).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
