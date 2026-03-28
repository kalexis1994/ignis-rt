"""Debug: compare co_object vs co_hair spaces. Run in Blender Text Editor (Alt+P)."""
import bpy
import numpy as np

dg = bpy.context.evaluated_depsgraph_get()
obj = bpy.context.active_object
oe = obj.evaluated_get(dg)
ps = oe.particle_systems[0]
s = ps.settings
n_par = len(ps.particles)
render_step = s.render_step
total_steps = (1 << render_step)

mod = None
for m in oe.modifiers:
    if m.type == 'PARTICLE_SYSTEM' and m.particle_system == ps:
        mod = m
        break

print("render_step=%d total_steps=%d" % (render_step, total_steps))
print("parents=%d children=%d" % (n_par, len(ps.child_particles)))
print("")

# Compare parent 0 ALL keys in both spaces
p = ps.particles[0]
n_keys = len(p.hair_keys)
print("Parent 0: %d keys" % n_keys)

# Sample co_hair at indices matching co_object keys
sample = np.linspace(0, total_steps - 1, n_keys, dtype=int)

for ki in range(n_keys):
    co_obj = p.hair_keys[ki].co_object(oe, mod, p)
    si = int(sample[ki])
    co_h = ps.co_hair(oe, particle_no=0, step=si)
    print("  key[%d] step=%d" % (ki, si))
    print("    co_object: %.4f, %.4f, %.4f" % (co_obj[0], co_obj[1], co_obj[2]))
    print("    co_hair:   %.4f, %.4f, %.4f" % (co_h[0], co_h[1], co_h[2]))

print("")
print("Object matrix_world:")
for r in range(4):
    row = [oe.matrix_world[r][c] for c in range(4)]
    print("  [%.4f, %.4f, %.4f, %.4f]" % tuple(row))
