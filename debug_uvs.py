"""Check UV range for ONLY the faces using laminate_floor_02."""
import bpy
import numpy as np

obj = bpy.data.objects.get("Floor")
if not obj or obj.type != 'MESH':
    print("Floor object not found")
else:
    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    mesh.calc_loop_triangles()

    uv_layer = mesh.uv_layers.active
    if not uv_layer:
        print("No UV layer")
    else:
        # Find material index for laminate_floor_02
        lam_idx = -1
        for i, slot in enumerate(obj.material_slots):
            if slot.material and 'laminate' in slot.material.name.lower():
                lam_idx = i
                print(f"Found '{slot.material.name}' at slot {i}")

        if lam_idx < 0:
            print("laminate_floor_02 not found in material slots")
        else:
            # Collect UVs for triangles with this material
            lam_u = []
            lam_v = []
            tri_count = 0
            for tri in mesh.loop_triangles:
                if tri.material_index == lam_idx:
                    tri_count += 1
                    for loop_idx in tri.loops:
                        uv = uv_layer.data[loop_idx].uv
                        lam_u.append(uv[0])
                        lam_v.append(uv[1])

            if lam_u:
                print(f"\nLaminate faces: {tri_count} triangles, {len(lam_u)} UV vertices")
                print(f"  U range: [{min(lam_u):.6f}, {max(lam_u):.6f}]")
                print(f"  V range: [{min(lam_v):.6f}, {max(lam_v):.6f}]")
                print(f"  U span: {max(lam_u) - min(lam_u):.6f}")
                print(f"  V span: {max(lam_v) - min(lam_v):.6f}")

                # With Scale=15, effective UV range:
                u_span = (max(lam_u) - min(lam_u)) * 15
                v_span = (max(lam_v) - min(lam_v)) * 15
                print(f"\n  With Scale=15: U tiles ≈ {u_span:.1f}, V tiles ≈ {v_span:.1f}")
            else:
                print("No faces with laminate material")
