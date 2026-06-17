"""Local Vulkan bring-up test for the Rust ignis_rt.dll.

Calls ignis_set_log_path + ignis_create on this machine (Intel iGPU, no hardware RT)
to verify instance + logical device + allocator come up. RT extensions/features are
skipped when unsupported, so create() should still succeed here. The full RT path is
verified on the RTX box via the same log file.

Run:  python rust/create_test.py
"""
import ctypes
import os
import tempfile
from ctypes import c_bool, c_char_p, c_uint32

DLL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "target", "release", "ignis_rt.dll")

log_path = os.path.join(tempfile.gettempdir(), "ignis_rs_create_test.log")
if os.path.exists(log_path):
    os.remove(log_path)

lib = ctypes.CDLL(DLL)
lib.ignis_set_log_path.argtypes = [c_char_p]
lib.ignis_set_log_path.restype = None
lib.ignis_create.argtypes = [c_uint32, c_uint32]
lib.ignis_create.restype = c_bool
lib.ignis_destroy.argtypes = []
lib.ignis_destroy.restype = None

lib.ignis_get_int.argtypes = [c_char_p]
lib.ignis_get_int.restype = ctypes.c_int
lib.ignis_render_frame.argtypes = []
lib.ignis_render_frame.restype = None
lib.ignis_readback_float.argtypes = [ctypes.POINTER(ctypes.c_float), c_uint32]
lib.ignis_readback_float.restype = c_bool

lib.ignis_set_log_path(log_path.encode("utf-8"))
ok = lib.ignis_create(1920, 1080)
print(f"ignis_create -> {ok}")

rw = lib.ignis_get_int(b"render_width")
rh = lib.ignis_get_int(b"render_height")
print(f"render_width={rw} render_height={rh}")

# GPU path: render_frame clears the offscreen image on the GPU, readback copies it back.
# Frame 0 clear color ~ [0.5, 0.933, 0.067, 1.0].
lib.ignis_render_frame()
n = rw * rh  # full frame, matches what the addon requests
buf = (ctypes.c_float * (n * 4))()
rb = lib.ignis_readback_float(buf, n)
print(f"readback_float -> {rb}, frame0 px RGBA={[round(buf[k],3) for k in range(4)]}")

lib.ignis_destroy()

print("\n--- DLL log ---")
if os.path.exists(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        print(f.read())
else:
    print("(no log written)")

print("PASS" if ok else "FAIL (see log)")
