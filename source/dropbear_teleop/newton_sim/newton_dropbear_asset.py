"""Explicit, in-memory USD adaptations for Newton's importer."""
from collections import deque


def reverse_joint(joint):
    """Exchange body frames and negate the scalar coordinate convention."""
    from pxr import UsdPhysics
    a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    joint.GetBody0Rel().SetTargets(b)
    joint.GetBody1Rel().SetTargets(a)
    for left, right in ((joint.GetLocalPos0Attr(), joint.GetLocalPos1Attr()),
                        (joint.GetLocalRot0Attr(), joint.GetLocalRot1Attr())):
        lv, rv = left.Get(), right.Get()
        left.Set(rv)
        right.Set(lv)
    prim = joint.GetPrim()
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        lo, hi = prim.GetAttribute("physics:lowerLimit"), prim.GetAttribute("physics:upperLimit")
        lv, hv = lo.Get(), hi.Get()
        lo.Set(-hv)
        hi.Set(-lv)
        for name in ("angular", "linear"):
            if prim.HasAPI(UsdPhysics.DriveAPI, name):
                drive = UsdPhysics.DriveAPI(prim, name)
                for attr in (drive.GetTargetPositionAttr(), drive.GetTargetVelocityAttr()):
                    attr.Set(-attr.Get())


def prepare_stage(source):
    from pxr import Usd, UsdGeom, UsdPhysics, Work
    # USD 26.3's Windows native physics parser can crash during parallel parsing
    # of this large instanced assembly. Simulation itself remains GPU-parallel.
    Work.SetConcurrencyLimit(1)
    stage = Usd.Stage.Open(str(source))
    stage.SetEditTarget(stage.GetSessionLayer())
    changes = {"reversedJoints": [], "expandedInstances": [], "collisionMeshes": 0,
               "collisionMeshApproximation": "convexHull", "sourceNonpositiveInertias": []}
    # Expanding references makes collision meshes visible to the USD physics parser.
    while True:
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            changes["expandedInstances"].append(str(prim.GetPath()))
            prim.SetInstanceable(False)
    for prim in list(stage.Traverse()):
        if prim.HasAPI(UsdPhysics.MassAPI):
            inertia = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get()
            if inertia is not None and any(value <= 0 for value in inertia):
                changes["sourceNonpositiveInertias"].append({"body": str(prim.GetPath()), "diagonal": list(inertia)})
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Xform):
            for descendant in Usd.PrimRange(prim):
                if descendant.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(descendant)
                    UsdPhysics.MeshCollisionAPI.Apply(descendant).CreateApproximationAttr("convexHull")
                    changes["collisionMeshes"] += 1
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
    adjacency = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetExcludeFromArticulationAttr().Get() or not joint.GetJointEnabledAttr().Get():
            continue
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(a) == len(b) == 1:
            a, b = str(a[0]), str(b[0])
            adjacency.setdefault(a, []).append((b, joint, False))
            adjacency.setdefault(b, []).append((a, joint, True))
    visited = {"/humanoid/world"}
    queue = deque(visited)
    while queue:
        parent = queue.popleft()
        for child, joint, reverse in adjacency.get(parent, []):
            if child in visited:
                continue
            visited.add(child)
            queue.append(child)
            if reverse:
                reverse_joint(joint)
                changes["reversedJoints"].append(str(joint.GetPath()))
    return stage, changes
