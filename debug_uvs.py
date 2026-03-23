"""Test: apply Mapping transform to a UV in Python and compare with expected result."""
import math

# Laminate Mapping: Scale=(15,15,15), Rotation Z=90° (1.5708 rad), Point mode
scale_x, scale_y = 15.0, 15.0
rot = 1.5708  # 90 degrees

# Test UV (first triangle of laminate)
u, v = 0.532530, 0.421151

# Cycles POINT mode: result = Rotation * (UV * Scale) + Location
# Step 1: Scale
scaled_u = u * scale_x  # 0.53253 * 15 = 7.98795
scaled_v = v * scale_y  # 0.421151 * 15 = 6.31727

# Step 2: Rotate Z by 90°
c = math.cos(rot)  # ≈ 0
s = math.sin(rot)  # ≈ 1
rot_u = scaled_u * c - scaled_v * s  # ≈ -6.317
rot_v = scaled_u * s + scaled_v * c  # ≈ 7.988

print(f"Input UV: ({u:.6f}, {v:.6f})")
print(f"Scaled:   ({scaled_u:.6f}, {scaled_v:.6f})")
print(f"Rotated:  ({rot_u:.6f}, {rot_v:.6f})")
print(f"Wrapped:  ({rot_u % 1.0:.6f}, {rot_v % 1.0:.6f})")

print()

# Now simulate what our shader does with V-flip:
v_vk = 1.0 - v  # V-flip in shader = 0.578849

print(f"=== Shader simulation ===")
print(f"Vulkan UV: ({u:.6f}, {v_vk:.6f})")

# UV_TRANSFORM: undo flip, scale, redo flip
bl_u = u
bl_v = 1.0 - v_vk  # = v = 0.421151
t_u = bl_u * scale_x  # 7.98795
t_v = bl_v * scale_y  # 6.31727
out_u = t_u
out_v = 1.0 - t_v  # = 1 - 6.31727 = -5.31727
print(f"After UV_TRANSFORM: ({out_u:.6f}, {out_v:.6f})")

# UV_ROTATE: undo flip, rotate, redo flip
bl2_u = out_u  # 7.98795
bl2_v = 1.0 - out_v  # = 1 - (-5.31727) = 6.31727
r_u = bl2_u * c - bl2_v * s  # ≈ -6.317
r_v = bl2_u * s + bl2_v * c  # ≈ 7.988
final_u = r_u
final_v = 1.0 - r_v  # = 1 - 7.988 = -6.988
print(f"After UV_ROTATE:    ({final_u:.6f}, {final_v:.6f})")
print(f"Wrapped:            ({final_u % 1.0:.6f}, {final_v % 1.0:.6f})")

print()
print(f"=== Comparison ===")
print(f"Cycles wrapped:  ({rot_u % 1.0:.6f}, {rot_v % 1.0:.6f})")
print(f"Shader wrapped:  ({final_u % 1.0:.6f}, {final_v % 1.0:.6f})")
print(f"Match: {abs(rot_u % 1.0 - final_u % 1.0) < 0.001 and abs(rot_v % 1.0 - final_v % 1.0) < 0.001}")
