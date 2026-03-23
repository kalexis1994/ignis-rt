"""Check Floor object: modifiers, UV layers, transform, and per-material UV ranges."""
import bpy

obj = bpy.data.objects.get("Floor")
if not obj:
    print("Floor not found")
else:
    print(f"=== Object: {obj.name} ===")
    print(f"  Type: {obj.type}")
    print(f"  Location: ({obj.location.x:.3f}, {obj.location.y:.3f}, {obj.location.z:.3f})")
    print(f"  Scale: ({obj.scale.x:.3f}, {obj.scale.y:.3f}, {obj.scale.z:.3f})")
    print(f"  Rotation: ({obj.rotation_euler.x:.4f}, {obj.rotation_euler.y:.4f}, {obj.rotation_euler.z:.4f})")

    # Modifiers
    print(f"\n  Modifiers ({len(obj.modifiers)}):")
    for mod in obj.modifiers:
        print(f"    - {mod.name} (type={mod.type})")

    # UV layers
    mesh = obj.data
    print(f"\n  UV Layers ({len(mesh.uv_layers)}):")
    for uv in mesh.uv_layers:
        print(f"    - '{uv.name}' active={uv.active}")

    # Materials
    print(f"\n  Materials ({len(obj.material_slots)}):")
    for i, slot in enumerate(obj.material_slots):
        name = slot.material.name if slot.material else "None"
        print(f"    [{i}] {name}")

    # Per-material UV ranges (using evaluated mesh)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data
    eval_mesh.calc_loop_triangles()
    uv_layer = eval_mesh.uv_layers.active

    if uv_layer:
        for mat_idx, slot in enumerate(obj.material_slots):
            mat_name = slot.material.name if slot.material else "None"
            u_vals = []
            v_vals = []
            tri_count = 0
            for tri in eval_mesh.loop_triangles:
                if tri.material_index == mat_idx:
                    tri_count += 1
                    for loop_idx in tri.loops:
                        uv = uv_layer.data[loop_idx].uv
                        u_vals.append(uv[0])
                        v_vals.append(uv[1])
            if u_vals:
                print(f"\n    Material [{mat_idx}] '{mat_name}': {tri_count} tris")
                print(f"      U: [{min(u_vals):.6f}, {max(u_vals):.6f}] span={max(u_vals)-min(u_vals):.6f}")
                print(f"      V: [{min(v_vals):.6f}, {max(v_vals):.6f}] span={max(v_vals)-min(v_vals):.6f}")

    # Check if evaluated mesh differs from original (modifiers applied)
    print(f"\n  Original verts: {len(mesh.vertices)}, Evaluated verts: {len(eval_mesh.vertices)}")
    print(f"  Original loops: {len(mesh.loops)}, Evaluated loops: {len(eval_mesh.loops)}")
