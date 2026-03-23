"""Compare UVs: what Blender sees vs what our export would produce."""
import bpy
import numpy as np

obj = bpy.data.objects.get("Floor")
if not obj:
    print("Floor not found")
else:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.data
    mesh.calc_loop_triangles()

    uv_layer = mesh.uv_layers.active
    if not uv_layer:
        print("No UV layer")
    else:
        # Method 1: Direct loop_triangle UV access (what Blender shows)
        print("=== Method 1: Direct UV from loop_triangles ===")
        for i, tri in enumerate(mesh.loop_triangles[:5]):
            uvs = [uv_layer.data[li].uv for li in tri.loops]
            mat = obj.material_slots[tri.material_index].material.name if tri.material_index < len(obj.material_slots) else "?"
            print(f"  tri[{i}] mat={mat}: UV0=({uvs[0][0]:.6f},{uvs[0][1]:.6f}) UV1=({uvs[1][0]:.6f},{uvs[1][1]:.6f}) UV2=({uvs[2][0]:.6f},{uvs[2][1]:.6f})")

        # Method 2: Our export method (foreach_get + indexing)
        print("\n=== Method 2: Our export (foreach_get + tri_loops indexing) ===")
        tri_count = len(mesh.loop_triangles)
        raw_vert_count = tri_count * 3
        tri_loops = np.empty(raw_vert_count, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)

        uv_data = mesh.uv_layers.active.data
        all_loop_uvs = np.empty(len(uv_data) * 2, dtype=np.float32)
        uv_data.foreach_get("uv", all_loop_uvs)
        all_loop_uvs = all_loop_uvs.reshape(-1, 2)
        uvs_export = all_loop_uvs[tri_loops]

        for i in range(5):
            u0, v0 = uvs_export[i*3]
            u1, v1 = uvs_export[i*3+1]
            u2, v2 = uvs_export[i*3+2]
            tri = mesh.loop_triangles[i]
            mat = obj.material_slots[tri.material_index].material.name if tri.material_index < len(obj.material_slots) else "?"
            print(f"  tri[{i}] mat={mat}: UV0=({u0:.6f},{v0:.6f}) UV1=({u1:.6f},{v1:.6f}) UV2=({u2:.6f},{v2:.6f})")

        # Check: do they match?
        print("\n=== Comparison ===")
        match = True
        for i, tri in enumerate(mesh.loop_triangles):
            for j, li in enumerate(tri.loops):
                blender_uv = uv_layer.data[li].uv
                our_uv = uvs_export[i*3+j]
                if abs(blender_uv[0] - our_uv[0]) > 0.001 or abs(blender_uv[1] - our_uv[1]) > 0.001:
                    print(f"  MISMATCH tri[{i}] corner[{j}]: Blender=({blender_uv[0]:.6f},{blender_uv[1]:.6f}) Export=({our_uv[0]:.6f},{our_uv[1]:.6f})")
                    match = False
        if match:
            print("  All UVs MATCH between Blender and our export method")
