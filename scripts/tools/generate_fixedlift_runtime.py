from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdPhysics


DEFAULT_PRIM = "tn__AMRTotalassy_step_qH9y3"
SOURCE_USD = "./lift_amr_new.usd"

LOCKED_JOINT_LIMITS = {
    "Link_1to7_2_joint": (5399.99, 5400.01),
    "Link_1to2_2_joint": (899.99, 900.01),
    "Link_2to3_2_joint": (899.99, 900.01),
    "Link_3to4_2_joint": (899.99, 900.01),
    "Link_4to5_2_joint": (-900.01, -899.99),
    "Link_5to6_2_joint": (-900.01, -899.99),
    "Link_6to7_2_joint": (-900.01, -899.99),
    "steering/FL_steer": (179.99, 180.01),
    "steering/FR_steer": (179.99, 180.01),
    "steering/RL_steer": (-0.01, 0.01),
    "steering/RR_steer": (-0.01, 0.01),
}

DISABLED_COLLISION_ROOTS = (
    f"/{DEFAULT_PRIM}/AMR/tn__WA24AV10100_TelescopicliftRev10_sW0RhD69Ch1",
    f"/{DEFAULT_PRIM}/AMR/tn__WA24AV106AMR_Lift_cover_frame_assy_iZ0c8Z0m0k1e469",
)


def _quoted_path_components(path: str) -> list[str]:
    return [component for component in path.strip("/").split("/") if component]


def _insert_path(tree: dict, path: str, attr_line: str) -> None:
    node = tree
    for component in _quoted_path_components(path):
        node = node.setdefault(component, {})
    node.setdefault("__attrs__", [])
    if attr_line not in node["__attrs__"]:
        node["__attrs__"].append(attr_line)


def _emit_tree(tree: dict, indent: int = 0) -> list[str]:
    lines: list[str] = []
    for name in sorted(key for key in tree.keys() if key != "__attrs__"):
        child = tree[name]
        lines.append("    " * indent + f'over "{name}"')
        lines.append("    " * indent + "{")
        for attr_line in child.get("__attrs__", []):
            lines.append("    " * (indent + 1) + attr_line)
        lines.extend(_emit_tree(child, indent + 1))
        lines.append("    " * indent + "}")
        lines.append("")
    return lines


def _find_collision_prims(stage: Usd.Stage) -> list[str]:
    collision_paths: list[str] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if any(prim_path.startswith(root) for root in DISABLED_COLLISION_ROOTS):
            collision_paths.append(prim_path)
    return sorted(set(collision_paths))


def generate_runtime_usda(source_usd: Path, output_usda: Path) -> None:
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {source_usd}")

    collision_paths = _find_collision_prims(stage)

    lines: list[str] = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{DEFAULT_PRIM}"',
        "    metersPerUnit = 0.001",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{DEFAULT_PRIM}" (',
        f"    prepend references = @{SOURCE_USD}@",
        ")",
        "{",
        '    over Xform "GroundPlane" (',
        "        active = false",
        "    )",
        "    {",
        '        token visibility = "invisible"',
        "",
        '        over Mesh "CollisionMesh"',
        "        {",
        '            token visibility = "invisible"',
        "            bool physics:collisionEnabled = 0",
        "        }",
        "",
        '        over Plane "CollisionPlane"',
        "        {",
        '            token visibility = "invisible"',
        "            bool physics:collisionEnabled = 0",
        "        }",
        "    }",
        "",
        '    over "Joint"',
        "    {",
    ]

    joint_override_tree: dict = {}
    for joint_path, (lower, upper) in LOCKED_JOINT_LIMITS.items():
        _insert_path(joint_override_tree, joint_path, f"float physics:lowerLimit = {lower}")
        _insert_path(joint_override_tree, joint_path, f"float physics:upperLimit = {upper}")
    lines.extend(_emit_tree(joint_override_tree, indent=2))
    if lines[-1] == "":
        lines.pop()
    lines.extend(["    }", ""])

    override_tree: dict = {}
    for collision_path in collision_paths:
        relative_path = collision_path.removeprefix(f"/{DEFAULT_PRIM}/")
        _insert_path(override_tree, relative_path, "bool physics:collisionEnabled = 0")

    lines.extend(_emit_tree(override_tree, indent=1))
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("}")
    lines.append("")

    output_usda.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed-lift Frbot runtime USDA.")
    parser.add_argument("--source", required=True, help="Path to the source robot USD.")
    parser.add_argument("--output", required=True, help="Path to the generated runtime USDA.")
    args = parser.parse_args()

    source_usd = Path(args.source)
    output_usda = Path(args.output)
    output_usda.parent.mkdir(parents=True, exist_ok=True)
    generate_runtime_usda(source_usd=source_usd, output_usda=output_usda)
    print(output_usda)


if __name__ == "__main__":
    main()
