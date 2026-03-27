"""Debug particle systems. Run in Blender Text Editor (Alt+P)."""
import bpy

print("=" * 60)
print("PARTICLE SYSTEM DEBUG")
print("=" * 60)

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    for ps_idx, ps in enumerate(eval_obj.particle_systems):
        if ps.settings.type != 'HAIR':
            continue
        s = ps.settings
        print(f"\nObject: '{obj.name}' PS[{ps_idx}]: '{ps.name}'")
        print(f"  type={s.type} render_as={s.render_type}")
        print(f"  parents={s.count}")
        print(f"  child_type={s.child_type}")
        print(f"  child_nbr={getattr(s, 'child_nbr', getattr(s, 'child_percent', '?'))}")
        print(f"  rendered_child_count={getattr(s, 'rendered_child_count', getattr(s, 'child_render_percent', '?'))}")
        print(f"  display_step={s.display_step} render_step={s.render_step}")
        print(f"  radius_scale={s.radius_scale}")
        print(f"  root_radius={getattr(s, 'root_radius', '?')}")
        print(f"  tip_radius={getattr(s, 'tip_radius', '?')}")
        print(f"  shape={getattr(s, 'shape', '?')}")
        print(f"  clump_factor={getattr(s, 'clump_factor', '?')}")
        print(f"  clump_shape={getattr(s, 'clump_shape', '?')}")
        print(f"  clump_noise_size={getattr(s, 'clump_noise_size', '?')}")
        print(f"  roughness_1={getattr(s, 'roughness_1', '?')}")
        print(f"  roughness_1_size={getattr(s, 'roughness_1_size', '?')}")
        print(f"  roughness_2={getattr(s, 'roughness_2', '?')}")
        print(f"  roughness_2_size={getattr(s, 'roughness_2_size', '?')}")
        print(f"  roughness_endpoint={getattr(s, 'roughness_endpoint', '?')}")
        print(f"  kink={getattr(s, 'kink', '?')}")
        print(f"  kink_amplitude={getattr(s, 'kink_amplitude', '?')}")
        print(f"  len(particles)={len(ps.particles)}")
        n_child = len(ps.child_particles) if hasattr(ps, 'child_particles') else 0
        print(f"  len(child_particles)={n_child}")

        if len(ps.particles) > 0:
            p = ps.particles[0]
            n_keys = len(p.hair_keys)
            print(f"  parent[0] keys={n_keys}")
            ps_mod = None
            for mod in eval_obj.modifiers:
                if mod.type == 'PARTICLE_SYSTEM' and mod.particle_system == ps:
                    ps_mod = mod
                    break
            if n_keys >= 2 and ps_mod:
                root = p.hair_keys[0].co_object(eval_obj, ps_mod, p)
                tip = p.hair_keys[-1].co_object(eval_obj, ps_mod, p)
                print(f"    co_object root=({root[0]:.4f},{root[1]:.4f},{root[2]:.4f})")
                print(f"    co_object tip=({tip[0]:.4f},{tip[1]:.4f},{tip[2]:.4f})")
            if n_keys >= 2:
                print(f"    co root=({p.hair_keys[0].co[0]:.4f},{p.hair_keys[0].co[1]:.4f},{p.hair_keys[0].co[2]:.4f})")
                print(f"    co tip=({p.hair_keys[-1].co[0]:.4f},{p.hair_keys[-1].co[1]:.4f},{p.hair_keys[-1].co[2]:.4f})")

        if n_child > 0:
            c0 = ps.child_particles[0]
            print(f"  child[0] attrs: {[a for a in dir(c0) if not a.startswith('_')]}")

# Check for Curves objects
print(f"\nCURVES objects:")
for obj in bpy.data.objects:
    if obj.type == 'CURVES':
        curves = obj.data
        print(f"  '{obj.name}' parent='{obj.parent.name if obj.parent else None}'")
        print(f"    curves={len(curves.curves)} points={len(curves.points)}")
        if len(curves.curves) > 0:
            c0 = curves.curves[0]
            print(f"    curve[0]: start={c0.first_point_index} len={c0.points_length}")
            if c0.points_length > 0:
                p0 = curves.points[c0.first_point_index]
                print(f"    point[0]: pos=({p0.position[0]:.4f},{p0.position[1]:.4f},{p0.position[2]:.4f}) r={p0.radius:.6f}")

# Dump ALL keys of first 3 strands
print(f"\n--- First 3 strands key dump ---")
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    for ps_idx, ps in enumerate(eval_obj.particle_systems):
        if ps.settings.type != 'HAIR':
            continue
        ps_mod = None
        for mod in eval_obj.modifiers:
            if mod.type == 'PARTICLE_SYSTEM' and mod.particle_system == ps:
                ps_mod = mod
                break
        print(f"  '{obj.name}' PS[{ps_idx}]:")
        for pi in range(min(3, len(ps.particles))):
            p = ps.particles[pi]
            hk = p.hair_keys
            print(f"    strand[{pi}] ({len(hk)} keys):")
            for ki in range(len(hk)):
                co = hk[ki].co
                line = f"      [{ki}] co=({co[0]:.6f},{co[1]:.6f},{co[2]:.6f})"
                if ps_mod:
                    cobj = hk[ki].co_object(eval_obj, ps_mod, p)
                    line += f" co_obj=({cobj[0]:.6f},{cobj[1]:.6f},{cobj[2]:.6f})"
                print(line)
        break

print("=" * 60)
