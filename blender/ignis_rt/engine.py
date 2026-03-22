"""Ignis RT render engine for Blender viewport."""

import ctypes
import math
import os
import time
import traceback

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import dll_wrapper
from . import scene_export
from .dll_wrapper import TLASInstance


# ── Python-side logger (writes to same .log as the C++ DLL) ──
_log_file = None
_log_path = ""


def _log_init():
    """Open the log file for Python-side writing (append mode, shared with C++)."""
    global _log_file, _log_path
    if _log_file is not None:
        return
    try:
        from . import get_log_path
        _log_path = get_log_path()
        _log_file = open(_log_path, "a", encoding="utf-8")
    except Exception:
        _log_path = ""
        _log_file = None


def _log(msg):
    """Write to both console and log file (flush immediately for crash safety)."""
    line = f"[ignis-py] {msg}"
    print(line)
    global _log_file
    if _log_file is None:
        _log_init()
    if _log_file is not None:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
            os.fsync(_log_file.fileno())
        except Exception:
            pass


def _log_exception(context_msg):
    """Log a full exception traceback to the log file."""
    tb = traceback.format_exc()
    _log(f"EXCEPTION in {context_msg}:\n{tb}")


def _log_close():
    """Flush and close the log file."""
    global _log_file
    if _log_file is not None:
        try:
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass
        _log_file = None


def _draw_texture_flipped(texture, w, h):
    """Draw texture fullscreen with V flipped (Vulkan top-down -> OpenGL bottom-up)."""
    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)

    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(
        shader, 'TRI_FAN',
        {
            "pos": ((0, 0), (w, 0), (w, h), (0, h)),
            "texCoord": ((0, 1), (1, 1), (1, 0), (0, 0)),
        },
    )
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)


# ── Module-level state — survives engine instance recreation ──
_ignis_initialized = False
_ignis_width = 0
_ignis_height = 0
_ignis_blas_handles = {}       # mesh_key -> BLAS handle
_ignis_obj_to_mesh = {}        # obj.name -> mesh_key (for BLAS lookup by object name)
_ignis_known_objects = set()   # ALL obj.names seen during initial load (including skipped)
_ignis_hidden_objects = set()  # obj.names in hidden collections (skip in sync)
_ignis_float_buffer = None     # ctypes c_float array for readback (legacy)
_ignis_byte_buffer = None      # ctypes c_uint8 array for RGBA8 readback
_ignis_gpu_texture = None      # Reusable GPUTexture (avoids alloc every frame)
_ignis_frame_index = 0         # continuous frame counter
_ignis_tex_manager = None      # Texture manager handle (opaque void*)
_fps_times = []
_fps_display = 0.0

# ── Incremental update flags (Cycles-style) ──
_ignis_full_dirty = True       # need full scene upload (meshes + materials + TLAS)
_ignis_materials_dirty = False  # need materials-only re-upload (property tweak)
_ignis_tlas_dirty = False      # need TLAS rebuild only (move/hide/show/delete)
_ignis_last_full_upload = 0.0  # perf_counter timestamp (cooldown for init burst)
_ignis_instance_count = 0      # mesh instance count after last sync
_ignis_last_tex_names = None   # texture name tuple from last upload (skip re-upload if same)
_ignis_last_transforms = None  # cached instance transforms hash for change detection
_ignis_last_light_count = -1   # track light count for emissive re-export
_ignis_hdri_sun = None         # sun extracted from HDRI (used as fallback when no SUN light)
_ignis_mat_name_to_index = {}  # persistent material name→index mapping for incremental uploads

# ── Staged loading state machine ──
LOAD_IDLE = 0          # no loading in progress
LOAD_FADEIN = -1       # show loading screen fade-in (no work yet)
LOAD_INIT = 1          # initialize renderer
LOAD_EXPORT = 2        # export meshes from depsgraph (CPU)
LOAD_MESHES = 3        # upload meshes to GPU (batched)
LOAD_MATERIALS = 4     # upload materials + textures
LOAD_MATIDS = 5        # assign per-primitive material IDs
LOAD_TLAS = 6          # build acceleration structure
LOAD_FINALIZE = 7      # final setup (sun, etc.)
LOAD_TEX_UPLOAD = 8    # upload textures one-at-a-time to GPU (smooth spinner)

FADEIN_DURATION = 0.8  # seconds to show loading screen before starting work

_load_stage = LOAD_IDLE
_load_start_time = 0.0
_load_status = ""
_load_progress = 0.0   # 0.0-1.0
# Cached data between stages
_load_unique_meshes = None
_load_scene_instances = None
_load_mesh_keys = []
_load_mesh_idx = 0
_load_materials_data = None
_load_mat_name_to_index = None
_load_textures_list = None
_load_obj_to_mesh_key = None
_load_depsgraph = None
_load_tex_idx = 0
_load_tex_buffers = []
MESH_BATCH_SIZE = 4    # meshes per frame during loading


def _update_fps():
    """Track frame times and compute rolling average FPS."""
    global _fps_times, _fps_display
    now = time.perf_counter()
    _fps_times.append(now)
    if len(_fps_times) > 30:
        _fps_times = _fps_times[-30:]
    if len(_fps_times) >= 2:
        elapsed = _fps_times[-1] - _fps_times[0]
        if elapsed > 0:
            _fps_display = (len(_fps_times) - 1) / elapsed


def _draw_fps_overlay(w, h):
    """Draw FPS counter at top-right of viewport."""
    text = f"{_fps_display:.0f} FPS"
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    verts = ((w - 130, h - 46), (w - 14, h - 46), (w - 14, h - 74), (w - 130, h - 74))
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", (0.0, 0.0, 0.0, 0.6))
    batch.draw(shader)
    font_id = 0
    blf.size(font_id, 18)
    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.position(font_id, w - 122, h - 68, 0)
    blf.draw(font_id, text)
    gpu.state.blend_set('NONE')


from . import loading_screen


def _ignis_shutdown():
    global _ignis_initialized, _ignis_width, _ignis_height
    global _ignis_blas_handles, _ignis_float_buffer, _ignis_byte_buffer
    global _ignis_gpu_texture, _ignis_tex_manager
    global _ignis_frame_index, _ignis_full_dirty, _ignis_materials_dirty, _ignis_tlas_dirty
    global _ignis_last_full_upload, _ignis_instance_count
    global _fps_times, _fps_display
    global _load_stage
    if _ignis_initialized:
        _log("ignis_destroy()")
        if _ignis_tex_manager is not None:
            dll_wrapper.destroy_texture_manager(_ignis_tex_manager)
            _ignis_tex_manager = None
        dll_wrapper.destroy()
        _ignis_initialized = False
        _ignis_width = 0
        _ignis_height = 0
        _ignis_blas_handles = {}
        _ignis_float_buffer = None
        _ignis_byte_buffer = None
        _ignis_gpu_texture = None
        _ignis_frame_index = 0
        _ignis_full_dirty = True
        _ignis_materials_dirty = False
        _ignis_tlas_dirty = False
        _ignis_last_full_upload = 0.0
        _ignis_instance_count = 0
        _fps_times = []
        _fps_display = 0.0
        _load_stage = LOAD_IDLE


class IgnisRenderEngine(bpy.types.RenderEngine):
    bl_idname = "IGNIS_RT"
    bl_label = "Ignis RT"
    bl_use_preview = False
    bl_use_eevee_viewport = False
    bl_use_gpu_context = True

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _ensure_init(self, width, height):
        """Initialize or resize the renderer if needed. Returns True if ready."""
        global _ignis_initialized, _ignis_width, _ignis_height
        global _ignis_float_buffer, _ignis_byte_buffer, _ignis_gpu_texture
        global _ignis_frame_index, _ignis_blas_handles, _ignis_full_dirty, _ignis_materials_dirty, _ignis_tlas_dirty

        if not dll_wrapper.load():
            _log("ERROR: DLL failed to load")
            return False

        if _ignis_initialized and _ignis_width == width and _ignis_height == height:
            return True

        if _ignis_initialized:
            _log(f" Reinit: {_ignis_width}x{_ignis_height} -> {width}x{height}")
            _ignis_shutdown()

        from . import get_log_path, get_base_path
        log_path = get_log_path()

        # Write diagnostics to separate file BEFORE anything else
        import os as _os
        _diag_path = _os.path.join(_os.path.expanduser("~"), "ignis-diag.txt")
        try:
            _adir = _os.path.dirname(_os.path.abspath(__file__))
            with open(_diag_path, "w") as _df:
                _df.write(f"addon_dir={_adir}\n")
                _bcrumb = _os.path.join(_adir, "_deploy_root.txt")
                _df.write(f"breadcrumb_exists={_os.path.isfile(_bcrumb)}\n")
                if _os.path.isfile(_bcrumb):
                    with open(_bcrumb) as _bf:
                        _bc = _bf.read().strip()
                    _df.write(f"breadcrumb={_bc}\n")
                    _df.write(f"shaders_there={_os.path.isdir(_os.path.join(_bc, 'shaders'))}\n")
                _df.write(f"shaders_in_addon={_os.path.isdir(_os.path.join(_adir, 'shaders'))}\n")
        except Exception as _e:
            pass

        base_path = get_base_path()
        dll_wrapper.set_log_path(log_path)
        dll_wrapper.set_base_path(base_path)
        _log(f" log: {log_path}")
        _log(f" base: {base_path}")

        dll_wrapper.set_int("shader_mode", 1)

        try:
            props = bpy.context.scene.ignis_rt
            if props.dlss_enabled:
                dll_wrapper.set_int("dlss_enabled", 1)
                quality = int(props.dlss_quality)
                dll_wrapper.set_int("dlss_quality", quality)
                # RR works best with upscaling — at DLAA (native res) fall back to NRD
                if props.dlss_rr_enabled and quality < 6:
                    dll_wrapper.set_int("dlss_rr_enabled", 1)
                    _log(f" DLSS {props.dlss_quality} + Ray Reconstruction")
                elif props.dlss_rr_enabled and quality == 6:
                    _log(f" DLSS DLAA + NRD (RR needs upscaling, falling back to NRD)")
                else:
                    _log(f" DLSS {props.dlss_quality} + NRD")
            if hasattr(props, 'samples_per_pixel') and props.samples_per_pixel > 1:
                dll_wrapper.set_int("spp", props.samples_per_pixel)
                _log(f" SPP: {props.samples_per_pixel}")
            if props.use_wavefront:
                dll_wrapper.set_int("use_wavefront", 1)
                _log(f" Wavefront path tracing enabled")
        except Exception:
            pass

        _log(f" ignis_create({width}, {height}) ...")
        if not dll_wrapper.create(width, height):
            _log(f" ERROR: ignis_create({width}, {height}) failed -- check {log_path}")
            return False

        _log("ignis_create OK")

        # Report denoiser status
        dlss_on = dll_wrapper.get_int("dlss_active")
        rr_on = dll_wrapper.get_int("dlss_rr_active")
        nrd_on = dll_wrapper.get_int("nrd_active")
        if rr_on:
            _log(" Denoiser: DLSS Ray Reconstruction")
        elif nrd_on:
            _log(" Denoiser: NRD (RELAX + SIGMA)")
        elif dlss_on:
            _log(" Denoiser: DLSS SR only (no denoiser)")
        else:
            _log(" Denoiser: None")

        _ignis_initialized = True
        _ignis_width = width
        _ignis_height = height
        _ignis_frame_index = 0
        _ignis_blas_handles = {}
        _ignis_full_dirty = True
        _ignis_materials_dirty = False
        _ignis_tlas_dirty = False
        _ignis_float_buffer = (ctypes.c_float * (width * height * 4))()
        _ignis_byte_buffer = (ctypes.c_uint8 * (width * height * 4))()

        return True

    # ------------------------------------------------------------------
    # Staged loading — process one stage per view_draw call
    # ------------------------------------------------------------------

    def _begin_staged_load(self, depsgraph):
        """Start the staged loading pipeline."""
        global _load_stage, _load_start_time, _load_status, _load_progress
        global _load_unique_meshes, _load_scene_instances, _load_mesh_keys, _load_mesh_idx
        global _load_materials_data, _load_mat_name_to_index, _load_textures_list, _load_obj_to_mesh_key, _load_hidden
        global _load_depsgraph
        global _ignis_blas_handles

        _load_stage = LOAD_EXPORT
        _load_start_time = time.perf_counter()
        _load_status = "Preparing scene..."
        _load_progress = 0.0
        _load_unique_meshes = None
        _load_scene_instances = None
        _load_mesh_keys = []
        _load_mesh_idx = 0
        _load_materials_data = None
        _load_mat_name_to_index = None
        _load_textures_list = None
        _ignis_blas_handles = {}
        # Destroy texture manager before clearing geometry (avoids dangling references)
        global _ignis_tex_manager
        if _ignis_tex_manager is not None:
            try:
                dll_wrapper.destroy_texture_manager(_ignis_tex_manager)
            except Exception:
                pass
            _ignis_tex_manager = None
        # Clear GPU BLAS before reload — prevents index mismatch when BLAS
        # accumulate across reloads (handle 0 would reference old geometry)
        dll_wrapper.clear_geometry()
        _log("Staged load: BEGIN")

    def _tick_staged_load(self, depsgraph):
        """Process one stage of loading. Returns True when complete."""
        global _load_stage, _load_status, _load_progress
        global _load_unique_meshes, _load_scene_instances, _load_mesh_keys, _load_mesh_idx
        global _load_materials_data, _load_mat_name_to_index, _load_textures_list, _load_obj_to_mesh_key, _load_hidden
        global _load_tex_idx, _load_tex_buffers
        global _ignis_initialized, _ignis_width, _ignis_height
        global _ignis_float_buffer, _ignis_byte_buffer
        global _ignis_blas_handles, _ignis_frame_index
        global _ignis_full_dirty, _ignis_materials_dirty, _ignis_tlas_dirty
        global _ignis_last_full_upload, _ignis_instance_count, _ignis_tex_manager
        global _ignis_last_tex_names

        # Yield frame: skip one frame of work to let the spinner animate
        if hasattr(self, '_load_yield') and self._load_yield:
            self._load_yield = False
            return False

        if _load_stage == LOAD_FADEIN:
            # ── Fade-in phase: just show loading screen, no work yet ──
            _load_status = ""
            _load_progress = 0.0
            elapsed = time.perf_counter() - _load_start_time
            if elapsed >= FADEIN_DURATION:
                if not _ignis_initialized:
                    # Show "Initializing" text for 1 frame before blocking
                    _load_status = "Initializing renderer..."
                    _load_stage = LOAD_INIT
                else:
                    _load_stage = LOAD_EXPORT
            return False

        elif _load_stage == LOAD_INIT:
            # ── Stage 0: Vulkan init in background thread (spinner stays smooth) ──
            import threading

            if not hasattr(self, '_init_thread'):
                # First call: load DLL + configure on main thread, then start worker
                _load_status = "Initializing renderer..."
                _load_progress = 0.01
                dll_wrapper.load()
                from . import get_log_path, get_base_path
                dll_wrapper.set_log_path(get_log_path())
                dll_wrapper.set_base_path(get_base_path())
                dll_wrapper.set_int("shader_mode", 1)
                try:
                    props = bpy.context.scene.ignis_rt
                    if props.dlss_enabled:
                        dll_wrapper.set_int("dlss_enabled", 1)
                        dll_wrapper.set_int("dlss_quality", int(props.dlss_quality))
                        if props.dlss_rr_enabled and int(props.dlss_quality) < 6:
                            dll_wrapper.set_int("dlss_rr_enabled", 1)
                    if hasattr(props, 'samples_per_pixel') and props.samples_per_pixel > 1:
                        dll_wrapper.set_int("spp", props.samples_per_pixel)
                    if props.use_wavefront:
                        dll_wrapper.set_int("use_wavefront", 1)
                except Exception:
                    pass

                w = self._last_w if hasattr(self, '_last_w') else 1920
                h = self._last_h if hasattr(self, '_last_h') else 1080
                self._init_w, self._init_h = w, h
                self._init_ok = None  # result from thread

                def _worker():
                    try:
                        self._init_ok = dll_wrapper.create(w, h)
                    except Exception as e:
                        _log(f"Init thread error: {e}")
                        self._init_ok = False

                self._init_thread = threading.Thread(target=_worker, daemon=True)
                self._init_thread.start()
                return False

            # Poll thread
            if self._init_thread.is_alive():
                _load_status = "Initializing renderer..."
                _load_progress = 0.03
                return False  # spinner keeps spinning

            # Thread complete
            self._init_thread.join()
            ok = self._init_ok
            w, h = self._init_w, self._init_h
            del self._init_thread, self._init_ok, self._init_w, self._init_h

            if not ok:
                _log("ERROR: ignis_create failed in background thread")
                _load_stage = LOAD_IDLE
                return True

            # Finalize Python-side state
            _ignis_initialized = True
            _ignis_width = w
            _ignis_height = h
            _ignis_frame_index = 0
            _ignis_blas_handles = {}
            _ignis_full_dirty = True
            _ignis_materials_dirty = False
            _ignis_tlas_dirty = False
            _ignis_float_buffer = (ctypes.c_float * (w * h * 4))()
            _ignis_byte_buffer = (ctypes.c_uint8 * (w * h * 4))()
            _log("Init complete (background thread)")
            _load_stage = LOAD_EXPORT
            self._load_yield = True
            return False

        elif _load_stage == LOAD_EXPORT:
            # ── Stage 1: Export ALL meshes + instances from depsgraph ──
            # This extracts geometry from Blender (CPU-bound, requires GIL)
            _load_status = "Exporting geometry..."
            _load_progress = 0.05
            try:
                t0 = time.perf_counter()
                _load_unique_meshes, _load_scene_instances, _load_obj_to_mesh_key, _load_hidden = scene_export.export_meshes(depsgraph)
                _load_mesh_keys = list(_load_unique_meshes.keys())
                _load_mesh_idx = 0
                total_tris = sum(m["tri_count"] for m in _load_unique_meshes.values())
                total_verts = sum(m["vertex_count"] for m in _load_unique_meshes.values())
                dt = time.perf_counter() - t0
                _log(f"Stage EXPORT: {len(_load_mesh_keys)} meshes, "
                     f"{total_tris:,} tris, {total_verts:,} verts, "
                     f"{len(_load_scene_instances)} inst -- {dt:.2f}s")
            except Exception:
                _log_exception("LOAD_EXPORT")
                _load_stage = LOAD_IDLE
                return True
            _load_stage = LOAD_MESHES
            self._load_yield = True  # breathe after export
            return False

        elif _load_stage == LOAD_MESHES:
            # ── Stage 2: Upload meshes to GPU in batches ──
            # Each batch: BLAS build + attribute upload (GPU-bound)
            total = len(_load_mesh_keys)
            batch_size = 1  # 1 mesh per frame for smooth spinner
            end = min(_load_mesh_idx + batch_size, total)
            _load_status = f"Building BVH... {end}/{total}"
            _load_progress = 0.1 + 0.4 * (end / max(total, 1))

            for i in range(_load_mesh_idx, end):
                mesh_key = _load_mesh_keys[i]
                m = _load_unique_meshes[mesh_key]
                try:
                    blas = dll_wrapper.upload_mesh(
                        m["positions"], m["vertex_count"],
                        m["indices"], m["index_count"],
                    )
                    if blas < 0:
                        _log(f"  FAILED upload_mesh: {mesh_key}")
                        continue
                    _ignis_blas_handles[mesh_key] = blas
                    dll_wrapper.upload_mesh_attributes(
                        blas, m["normals"], m["uvs"], m["vertex_count"],
                    )
                except Exception:
                    _log_exception(f"LOAD_MESHES uploading '{mesh_key}'")

            _load_mesh_idx = end
            if _load_mesh_idx >= total:
                _log(f"Stage MESHES: {total} uploaded, {len(_ignis_blas_handles)} success")
                # Dump BLAS mapping with vertex fingerprints for corruption detection
                import os
                try:
                    with open(os.path.join(os.path.expanduser("~"), "ignis-blas-map.txt"), "w") as _bf:
                        _bf.write(f"BLAS handles ({len(_ignis_blas_handles)} entries):\n")
                        for mk, bh in sorted(_ignis_blas_handles.items(), key=lambda x: x[1]):
                            m = _load_unique_meshes.get(mk, {})
                            pos = m.get('positions', np.array([]))
                            v0 = f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})" if len(pos) >= 3 else "?"
                            _bf.write(f"  blas={bh:3d} mesh='{mk}' tris={m.get('tri_count',0)} v0={v0}\n")
                except Exception:
                    pass
                _load_stage = LOAD_MATERIALS
                self._load_yield = True  # breathe after mesh upload
            return False

        elif _load_stage == LOAD_MATERIALS:
            # ── Stage 3: Materials + textures (chunked for smooth spinner) ──
            # Sub-stage A: export materials from Blender (one-time)
            if _load_materials_data is None:
                _load_status = "Exporting materials..."
                _load_progress = 0.50
                try:
                    _load_materials_data, _load_mat_name_to_index, _load_textures_list = \
                        scene_export.export_materials(depsgraph, hidden_objects=_load_hidden)
                    _log(f"Stage MATERIALS: exporting {len(_load_mat_name_to_index)} materials, "
                         f"{len(_load_textures_list)} textures")
                    _load_tex_idx = 0
                    _load_tex_buffers = []
                    if _load_textures_list:
                        if _ignis_tex_manager is not None:
                            dll_wrapper.destroy_texture_manager(_ignis_tex_manager)
                            _ignis_tex_manager = None
                        _ignis_tex_manager = dll_wrapper.create_texture_manager()
                except Exception:
                    _log_exception("LOAD_MATERIALS export")
                    _load_stage = LOAD_IDLE
                    return True
                return False

            # Sub-stage B: extract + add textures one per frame
            if hasattr(self, '_load_tex_idx_done') and self._load_tex_idx_done:
                pass  # skip to material upload (no textures)
            elif _load_textures_list and _ignis_tex_manager:
                total_tex = len(_load_textures_list)
                if _load_tex_idx < total_tex:
                    ti = _load_tex_idx
                    tex_info = _load_textures_list[ti]
                    _load_status = f"Loading textures... {ti + 1}/{total_tex}"
                    _load_progress = 0.50 + 0.15 * ((ti + 1) / max(total_tex, 1))

                    # Extract pixel data from Blender (deferred from export_materials)
                    scene_export.extract_texture_bytes(tex_info)

                    if tex_info["data"] is not None:
                        _log(f"  tex[{ti}] '{tex_info['name']}': "
                             f"{tex_info['width']}x{tex_info['height']}, "
                             f"{len(tex_info['data'])} bytes")
                        data_bytes = tex_info["data"]
                        data_np = np.frombuffer(data_bytes, dtype=np.uint8).copy()
                        _load_tex_buffers.append(data_np)
                        dll_wrapper.texture_manager_add(
                            _ignis_tex_manager,
                            tex_info["name"],
                            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            len(data_bytes),
                            tex_info["width"], tex_info["height"],
                            1, 0,
                        )

                    _load_tex_idx += 1
                    if _load_tex_idx < total_tex:
                        return False  # next texture next frame
                # All textures added → transition to per-frame GPU upload
                self._load_tex_idx_done = True
                _load_stage = LOAD_TEX_UPLOAD
                return False

            # No textures — go straight to material upload
            _load_status = "Uploading materials..."
            _load_progress = 0.68
            mat_count = max(len(_load_mat_name_to_index), 1)
            dll_wrapper.upload_materials(_load_materials_data, mat_count)
            _ignis_last_tex_names = tuple(t["name"] for t in _load_textures_list)
            _log(f"Stage MATERIALS: {mat_count} mat, {len(_load_textures_list)} tex")
            self._load_tex_idx_done = False
            _load_stage = LOAD_MATIDS
            return False

        elif _load_stage == LOAD_TEX_UPLOAD:
            # ── Stage 3b: Upload textures to GPU one-at-a-time (smooth spinner) ──
            if _ignis_tex_manager:
                pending = dll_wrapper.texture_manager_pending_count(_ignis_tex_manager)
                total_tex = len(_load_textures_list) if _load_textures_list else 0
                done_count = total_tex - pending

                if pending > 0:
                    _load_status = f"Uploading textures to GPU... {done_count + 1}/{total_tex}"
                    _load_progress = 0.55 + 0.13 * ((done_count + 1) / max(total_tex, 1))
                    dll_wrapper.texture_manager_upload_one(_ignis_tex_manager)
                    return False  # next frame uploads next texture

                # All uploaded — update descriptors
                dll_wrapper.update_texture_descriptors(_ignis_tex_manager)
                _log(f"  texture upload OK ({total_tex} textures)")

            _load_tex_buffers = []

            # Upload material buffer
            _load_status = "Uploading materials..."
            _load_progress = 0.68
            mat_count = max(len(_load_mat_name_to_index), 1)
            dll_wrapper.upload_materials(_load_materials_data, mat_count)
            _ignis_last_tex_names = tuple(t["name"] for t in _load_textures_list)
            _log(f"Stage MATERIALS: {mat_count} mat, {len(_load_textures_list)} tex")
            self._load_tex_idx_done = False
            _load_stage = LOAD_MATIDS
            return False

        elif _load_stage == LOAD_MATIDS:
            # ── Stage 4: Per-primitive material IDs (chunked) ──
            # Build lookup on first call
            if not hasattr(self, '_matid_keys'):
                mesh_first_inst = {}
                for inst in _load_scene_instances:
                    if inst["mesh_key"] not in mesh_first_inst:
                        mesh_first_inst[inst["mesh_key"]] = inst
                self._matid_keys = list(_load_unique_meshes.keys())
                self._matid_first_inst = mesh_first_inst
                self._matid_idx = 0

            total = len(self._matid_keys)
            MATID_BATCH = 10  # 10 matID uploads per frame
            end = min(self._matid_idx + MATID_BATCH, total)
            _load_status = f"Assigning materials... {end}/{total}"
            _load_progress = 0.70 + 0.10 * (end / max(total, 1))

            try:
                for ki in range(self._matid_idx, end):
                    mesh_key = self._matid_keys[ki]
                    m = _load_unique_meshes[mesh_key]
                    blas = _ignis_blas_handles.get(mesh_key)
                    if blas is not None and m["tri_count"] > 0:
                        first_inst = self._matid_first_inst.get(mesh_key)
                        if first_inst is None:
                            continue
                        slots = first_inst["material_slots"]
                        n_slots = max(len(slots), 1)
                        default_idx = _load_mat_name_to_index.get('__ignis_default__', 0)
                        slot_to_global = np.full(n_slots, default_idx, dtype=np.uint32)
                        for i, mat in enumerate(slots):
                            if mat is not None:
                                mat_key = f"{mat.library.filepath}:{mat.name}" if mat.library else mat.name
                                if mat_key in _load_mat_name_to_index:
                                    slot_to_global[i] = _load_mat_name_to_index[mat_key]
                        tri_mats = np.asarray(m["tri_material_indices"], dtype=np.int32)
                        clamped = np.clip(tri_mats, 0, n_slots - 1)
                        global_ids = np.ascontiguousarray(slot_to_global[clamped], dtype=np.uint32)
                        dll_wrapper.upload_mesh_primitive_materials(blas, global_ids, m["tri_count"])
                        slot_names = [s.name if s else 'None' for s in slots]
                        _log(f"  matIDs '{mesh_key}': blas={blas} {m['tri_count']} tris, {n_slots} slots={slot_names} -> global={list(slot_to_global)}")
            except Exception:
                _log_exception("LOAD_MATIDS")
                _load_stage = LOAD_IDLE
                del self._matid_keys, self._matid_first_inst, self._matid_idx
                return True

            self._matid_idx = end
            if self._matid_idx < total:
                return False  # more to process

            _log("Stage MATIDS: done")
            del self._matid_keys, self._matid_first_inst, self._matid_idx
            _load_stage = LOAD_TLAS
            self._load_yield = True  # breathe after matIDs
            return False

        elif _load_stage == LOAD_TLAS:
            # ── Stage 5: Build TLAS ──
            _load_status = "Building acceleration structure..."
            _load_progress = 0.85
            try:
                if _load_scene_instances:
                    count = len(_load_scene_instances)
                    _log(f"Stage TLAS: building with {count} instances...")
                    InstanceArray = TLASInstance * count
                    tlas_arr = InstanceArray()
                    for i, inst in enumerate(_load_scene_instances):
                        blas_idx = _ignis_blas_handles.get(inst["mesh_key"], -1)
                        if blas_idx < 0:
                            _log(f"  WARNING: instance {i} mesh '{inst['mesh_key']}' has no BLAS, skipping")
                            blas_idx = 0
                        tlas_arr[i].blasIndex = blas_idx
                        for j in range(12):
                            tlas_arr[i].transform[j] = inst["transform_3x4"][j]
                        tlas_arr[i].customIndex = blas_idx  # indexes geometryMetadata[], NOT instance
                        tlas_arr[i].mask = 0xFF
                    dll_wrapper.build_tlas(tlas_arr, count)
                    _log(f"Stage TLAS: {count} instances OK")
            except Exception:
                _log_exception("LOAD_TLAS")
                _load_stage = LOAD_IDLE
                return True
            _load_stage = LOAD_FINALIZE
            self._load_yield = True  # breathe after TLAS build
            return False

        elif _load_stage == LOAD_FINALIZE:
            # ── Stage 6: Sun + cleanup ──
            _load_status = "Finalizing..."
            _load_progress = 0.95

            sun = scene_export.export_sun(depsgraph)

            # World HDRI environment map
            hdri_sun = None
            try:
                hdri = scene_export.export_world_hdri(depsgraph)
                if hdri:
                    _log(f"Stage FINALIZE: HDRI '{hdri['name']}' {hdri['width']}x{hdri['height']}, strength={hdri['strength']:.2f}")
                    hdri_sun = hdri.get("extracted_sun")
                    # Upload as texture via the texture manager
                    if _ignis_tex_manager:
                        data_np = np.frombuffer(hdri["data"], dtype=np.uint8).copy()
                        hdri_dxgi = hdri.get("dxgi_format", 0)
                        hdri_idx = dll_wrapper.texture_manager_add(
                            _ignis_tex_manager, hdri["name"],
                            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            len(hdri["data"]), hdri["width"], hdri["height"], 1, hdri_dxgi)
                        if hdri_idx >= 0:
                            dll_wrapper.texture_manager_upload_all(_ignis_tex_manager)
                            dll_wrapper.update_texture_descriptors(_ignis_tex_manager)
                            dll_wrapper.set_int("hdri_tex_index", hdri_idx)
                            dll_wrapper.set_float("hdri_strength", hdri["strength"])
                            _log(f"Stage FINALIZE: HDRI uploaded as texture {hdri_idx}")
                        else:
                            _log("Stage FINALIZE: HDRI texture add failed")
                    else:
                        _log("Stage FINALIZE: No texture manager for HDRI")
                else:
                    dll_wrapper.set_int("hdri_tex_index", -1)
            except Exception:
                _log_exception("FINALIZE HDRI")

            # Use HDRI-extracted sun if no Blender SUN light exists
            global _ignis_hdri_sun
            if sun["sun_intensity"] <= 0.0 and hdri_sun:
                sun = hdri_sun
                hdri_str = hdri["strength"] if hdri else 1.0
                # Scale by PI to compensate for hemisphere sampling deficit
                # (Cycles importance-samples the sun disc efficiently; we use NEE)
                sun["sun_intensity"] = sun["sun_intensity"] * hdri_str * math.pi
                _log(f"Stage FINALIZE: Sun from HDRI — elev={sun['sun_elevation']:.1f}° "
                     f"azim={sun['sun_azimuth']:.1f}° intensity={sun['sun_intensity']:.1f} "
                     f"color=({sun['sun_color'][0]:.3f},{sun['sun_color'][1]:.3f},{sun['sun_color'][2]:.3f})")
                # Cache for per-frame sync (export_sun returns 0 when no SUN light)
                _ignis_hdri_sun = dict(sun)

            dll_wrapper.set_float("sun_elevation", sun["sun_elevation"])
            dll_wrapper.set_float("sun_azimuth", sun["sun_azimuth"])
            dll_wrapper.set_float("sun_intensity", sun["sun_intensity"])
            sun_color = sun.get("sun_color", (1.0, 1.0, 1.0))
            dll_wrapper.set_float("sun_color_r", sun_color[0])
            dll_wrapper.set_float("sun_color_g", sun_color[1])
            dll_wrapper.set_float("sun_color_b", sun_color[2])

            # Point/spot/area lights
            light_data = scene_export.export_lights(depsgraph)
            n_lights = len(light_data) // 16
            dll_wrapper.upload_lights(light_data, n_lights)
            _log(f"Stage FINALIZE: {n_lights} lights uploaded")

            # Emissive triangles for MIS — use already-exported mesh data to avoid
            # re-evaluating meshes (which hangs on 7M+ tri scenes)
            try:
                emissive_data = scene_export.export_emissive_triangles_fast(
                    depsgraph, _load_unique_meshes, _load_scene_instances, _load_mat_name_to_index)
                n_emissive = len(emissive_data) // 16
                dll_wrapper.upload_emissive_triangles(emissive_data, n_emissive)
                _log(f"Stage FINALIZE: {n_emissive} emissive triangles uploaded")
            except Exception:
                _log_exception("FINALIZE emissive triangles")
                _log("Stage FINALIZE: emissive export failed, continuing without MIS")

            # Save material mapping and obj→mesh mapping for incremental updates
            global _ignis_mat_name_to_index, _ignis_obj_to_mesh, _ignis_known_objects, _ignis_hidden_objects
            if _load_mat_name_to_index:
                _ignis_mat_name_to_index = dict(_load_mat_name_to_index)
            if _load_obj_to_mesh_key:
                _ignis_obj_to_mesh = dict(_load_obj_to_mesh_key)
            # Record ALL object names seen — so transform sync knows what's "new" vs "pre-existing"
            _ignis_known_objects = set(_ignis_obj_to_mesh.keys()) | set(_ignis_blas_handles.keys())
            if _load_hidden:
                _ignis_hidden_objects = set(_load_hidden)
            _load_unique_meshes = None
            _load_scene_instances = None
            _load_mesh_keys = []
            _load_materials_data = None
            _load_mat_name_to_index = None
            _load_textures_list = None
            _load_obj_to_mesh_key = None

            _ignis_full_dirty = False
            _ignis_materials_dirty = False
            _ignis_tlas_dirty = False
            _ignis_last_full_upload = time.perf_counter()
            _ignis_instance_count = _ignis_instance_count  # already set
            _ignis_frame_index = 0

            dt = time.perf_counter() - _load_start_time
            _log(f" Staged load: COMPLETE ({dt:.2f}s total)")
            _load_stage = LOAD_IDLE
            _load_progress = 1.0
            loading_screen.begin_fadeout()  # start fade-out transition to render
            return True

        return True  # unknown stage = done

    # ------------------------------------------------------------------
    # Incremental update detection (Cycles-style)
    # ------------------------------------------------------------------

    def view_update(self, context, depsgraph):
        """Called by Blender on scene changes.
        - New/deleted objects → handled incrementally in transform sync
        - Geometry MODIFIED on existing object → full reload (BLAS rebuild)
        - Material/NodeTree changed → fast material re-upload
        """
        global _ignis_full_dirty, _ignis_materials_dirty

        if _load_stage != LOAD_IDLE:
            return
        if not _ignis_blas_handles:
            return

        # Ignore updates for 2 seconds after a load completes
        # (Blender sends accumulated depsgraph updates that would re-trigger full load)
        if _ignis_last_full_upload > 0 and (time.perf_counter() - _ignis_last_full_upload) < 2.0:
            return

        has_material_change = False
        has_geometry_change = False
        geometry_mesh_name = None

        for update in depsgraph.updates:
            uid = update.id
            if isinstance(uid, (bpy.types.Material, bpy.types.NodeTree)):
                has_material_change = True
            elif isinstance(uid, bpy.types.Mesh):
                if update.is_updated_geometry:
                    has_geometry_change = True
                    geometry_mesh_name = uid.name

        # Material changes → fast re-upload (no full reload)
        if has_material_change:
            _ignis_materials_dirty = True

        # Geometry changes on known objects → full reload (BLAS needs rebuild)
        # Skip if accompanied by material change only (Blender reports mesh
        # update when material slot changes, but topology didn't change)
        if has_geometry_change and not has_material_change:
            if geometry_mesh_name and (
                geometry_mesh_name in _ignis_known_objects or
                geometry_mesh_name in _ignis_blas_handles
            ):
                _ignis_full_dirty = True

    # ------------------------------------------------------------------
    # Materials-only re-upload (fast)
    # ------------------------------------------------------------------

    def _update_materials(self, depsgraph):
        global _ignis_materials_dirty, _ignis_frame_index, _ignis_tex_manager
        global _ignis_last_tex_names, _ignis_mat_name_to_index

        t0 = time.perf_counter()
        materials_data, mat_name_to_index, textures_list = scene_export.export_materials(
            depsgraph, hidden_objects=_ignis_hidden_objects, existing_mapping=_ignis_mat_name_to_index)

        new_tex_names = tuple(t["name"] for t in textures_list)
        if new_tex_names != _ignis_last_tex_names:
            if _ignis_tex_manager is not None:
                dll_wrapper.destroy_texture_manager(_ignis_tex_manager)
                _ignis_tex_manager = None
            if textures_list:
                _ignis_tex_manager = dll_wrapper.create_texture_manager()
                if _ignis_tex_manager:
                    tex_buffers = []
                    for tex_info in textures_list:
                        data_bytes = tex_info["data"]
                        data_np = np.frombuffer(data_bytes, dtype=np.uint8).copy()
                        tex_buffers.append(data_np)
                        dll_wrapper.texture_manager_add(
                            _ignis_tex_manager,
                            tex_info["name"],
                            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            len(data_bytes),
                            tex_info["width"], tex_info["height"],
                            1, 0,
                        )
                    ok = dll_wrapper.texture_manager_upload_all(_ignis_tex_manager)
                    if ok:
                        dll_wrapper.update_texture_descriptors(_ignis_tex_manager)
                    del tex_buffers
            _ignis_last_tex_names = new_tex_names

        mat_count = max(len(mat_name_to_index), 1)
        dll_wrapper.upload_materials(materials_data, mat_count)
        _ignis_mat_name_to_index = dict(mat_name_to_index)
        _ignis_materials_dirty = False
        _ignis_frame_index = 0

        dt = time.perf_counter() - t0
        _log(f" Materials update: {mat_count} mat ({len(textures_list)} tex) -- {dt:.3f}s")

    # ------------------------------------------------------------------
    # Incremental mesh upload (Cycles-style live editing)
    # ------------------------------------------------------------------

    def _upload_new_objects(self, depsgraph, new_objects):
        """Upload new objects incrementally (Cycles-style live editing).

        new_objects: dict of obj_name → (obj, instance) for objects not yet in BLAS.
        """
        global _ignis_blas_handles, _ignis_obj_to_mesh, _ignis_materials_dirty
        import numpy as np

        t0 = time.perf_counter()

        # Export only the new objects (not the full scene — avoids 10s hang on large scenes)
        unique_meshes = {}
        instances = []
        obj_to_mesh = {}
        for obj_name, (obj, inst) in new_objects.items():
            mesh_key = obj.name
            obj_to_mesh[obj_name] = mesh_key
            if mesh_key in unique_meshes:
                continue
            try:
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                if mesh is None:
                    continue
                mesh.calc_loop_triangles()
                if len(mesh.loop_triangles) == 0:
                    eval_obj.to_mesh_clear()
                    continue
                tri_count = len(mesh.loop_triangles)
                raw_vert_count = tri_count * 3
                tri_loops = np.empty(raw_vert_count, dtype=np.int32)
                mesh.loop_triangles.foreach_get("loops", tri_loops)
                tri_verts_idx = np.empty(raw_vert_count, dtype=np.int32)
                mesh.loop_triangles.foreach_get("vertices", tri_verts_idx)
                all_pos = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", all_pos)
                all_pos = all_pos.reshape(-1, 3)
                all_nrm = np.empty(len(mesh.loops) * 3, dtype=np.float32)
                try:
                    mesh.corner_normals.foreach_get("vector", all_nrm)
                except (AttributeError, RuntimeError):
                    mesh.calc_normals_split()
                    mesh.loops.foreach_get("normal", all_nrm)
                all_nrm = all_nrm.reshape(-1, 3)
                positions = all_pos[tri_verts_idx]
                normals = all_nrm[tri_loops]
                uvs = np.zeros((raw_vert_count, 2), dtype=np.float32)
                if mesh.uv_layers.active is not None:
                    all_uvs = np.empty(len(mesh.uv_layers.active.data) * 2, dtype=np.float32)
                    mesh.uv_layers.active.data.foreach_get("uv", all_uvs)
                    uvs = all_uvs.reshape(-1, 2)[tri_loops]
                tri_mat_indices = np.empty(tri_count, dtype=np.int32)
                mesh.loop_triangles.foreach_get("material_index", tri_mat_indices)
                indices = np.arange(raw_vert_count, dtype=np.uint32)
                unique_meshes[mesh_key] = {
                    "positions": np.ascontiguousarray(positions.flatten(), dtype=np.float32),
                    "normals": np.ascontiguousarray(normals.flatten(), dtype=np.float32),
                    "uvs": np.ascontiguousarray(uvs.flatten(), dtype=np.float32),
                    "indices": indices,
                    "vertex_count": raw_vert_count,
                    "index_count": raw_vert_count,
                    "tri_count": tri_count,
                    "tri_material_indices": tri_mat_indices,
                }
                eval_obj.to_mesh_clear()
                instances.append({
                    "mesh_key": mesh_key,
                    "material_slots": [s.material for s in obj.material_slots],
                })
            except Exception:
                _log_exception(f"incremental export '{obj_name}'")

        uploaded = 0
        for obj_name in new_objects:
            mesh_key = obj_to_mesh.get(obj_name)
            if not mesh_key:
                continue

            # Register the obj→mesh mapping
            _ignis_obj_to_mesh[obj_name] = mesh_key

            # If this mesh_key already has a BLAS (shared mesh data), skip upload
            if mesh_key in _ignis_blas_handles:
                continue

            if mesh_key not in unique_meshes:
                continue
            mdata = unique_meshes[mesh_key]

            # Upload mesh geometry → BLAS
            positions = np.ascontiguousarray(mdata["positions"], dtype=np.float32)
            indices = np.ascontiguousarray(mdata["indices"], dtype=np.uint32)
            blas_handle = dll_wrapper.upload_mesh(positions, mdata["vertex_count"],
                                                   indices, mdata["index_count"])
            if blas_handle < 0:
                _log(f"  BLAS upload failed for '{mesh_key}'")
                continue

            # Upload attributes (normals + UVs)
            normals = np.ascontiguousarray(mdata["normals"], dtype=np.float32)
            uvs = np.ascontiguousarray(mdata["uvs"], dtype=np.float32)
            dll_wrapper.upload_mesh_attributes(blas_handle, normals, uvs, mdata["vertex_count"])

            # Upload per-primitive material IDs
            for inst_data in instances:
                if inst_data["mesh_key"] == mesh_key:
                    slots = inst_data.get("material_slots", [])
                    n_slots = max(len(slots), 1)
                    default_idx = _ignis_mat_name_to_index.get('__ignis_default__', 0)
                    slot_to_global = [default_idx] * n_slots
                    for i, mat in enumerate(slots):
                        if mat and mat.name in _ignis_mat_name_to_index:
                            slot_to_global[i] = _ignis_mat_name_to_index[mat.name]
                    tri_mat = mdata.get("tri_material_indices")
                    if tri_mat is not None:
                        clamped = np.clip(tri_mat, 0, n_slots - 1)
                        global_ids = np.array([slot_to_global[m] for m in clamped], dtype=np.uint32)
                        dll_wrapper.upload_mesh_primitive_materials(
                            blas_handle, global_ids, mdata["tri_count"])
                    break

            _ignis_blas_handles[mesh_key] = blas_handle
            uploaded += 1
            _log(f"  Incremental BLAS: '{mesh_key}' ({mdata['tri_count']} tris) -> handle {blas_handle}")

        # Register all new objects as known
        for obj_name in new_objects:
            _ignis_known_objects.add(obj_name)

        # New materials might be needed
        if uploaded > 0:
            _ignis_materials_dirty = True
            _ignis_frame_index = 0

        dt = time.perf_counter() - t0
        _log(f"  Incremental upload: {uploaded} new, {len(new_objects) - uploaded} shared -- {dt:.3f}s")

    # ------------------------------------------------------------------
    # TLAS-only rebuild (cheap)
    # ------------------------------------------------------------------

    def _rebuild_tlas(self, depsgraph):
        global _ignis_tlas_dirty, _ignis_instance_count, _ignis_frame_index

        t0 = time.perf_counter()
        instances = []
        for inst in depsgraph.object_instances:
            obj = inst.object
            if obj.type != 'MESH':
                continue
            mesh_key = _ignis_obj_to_mesh.get(obj.name)
            if not mesh_key or mesh_key not in _ignis_blas_handles:
                continue
            xform = scene_export._matrix_to_3x4_row_major(inst.matrix_world)
            instances.append((mesh_key, xform))

        if instances:
            count = len(instances)
            InstanceArray = TLASInstance * count
            tlas_arr = InstanceArray()
            for i, (mesh_key, xform) in enumerate(instances):
                blas_idx = _ignis_blas_handles[mesh_key]
                tlas_arr[i].blasIndex = blas_idx
                for j in range(12):
                    tlas_arr[i].transform[j] = xform[j]
                tlas_arr[i].customIndex = blas_idx  # indexes geometryMetadata[], NOT instance
                tlas_arr[i].mask = 0xFF
            dll_wrapper.build_tlas(tlas_arr, count)

        sun = scene_export.export_sun(depsgraph)
        if sun["sun_intensity"] <= 0.0 and _ignis_hdri_sun:
            sun = _ignis_hdri_sun
        dll_wrapper.set_float("sun_elevation", sun["sun_elevation"])
        dll_wrapper.set_float("sun_azimuth", sun["sun_azimuth"])
        dll_wrapper.set_float("sun_intensity", sun["sun_intensity"])
        sun_color = sun.get("sun_color", (1.0, 1.0, 1.0))
        dll_wrapper.set_float("sun_color_r", sun_color[0])
        dll_wrapper.set_float("sun_color_g", sun_color[1])
        dll_wrapper.set_float("sun_color_b", sun_color[2])

        # Re-sync point/spot/area lights (in case a light was moved/added)
        light_data = scene_export.export_lights(depsgraph)
        n_lights = len(light_data) // 16
        dll_wrapper.upload_lights(light_data, n_lights)

        _ignis_tlas_dirty = False
        _ignis_instance_count = len(instances)
        _ignis_frame_index = 0

        dt = time.perf_counter() - t0
        _log(f" TLAS rebuild: {len(instances)} instances -- {dt:.3f}s")

    # ------------------------------------------------------------------
    # RenderEngine callbacks
    # ------------------------------------------------------------------

    def view_draw(self, context, depsgraph):
        """Called on every viewport redraw."""
        global _ignis_frame_index, _ignis_full_dirty, _ignis_materials_dirty, _ignis_tlas_dirty
        global _load_stage, _ignis_instance_count, _load_start_time

        region = context.region
        w, h = region.width, region.height
        self._last_w, self._last_h = w, h

        # ── First contact: capture viewport and start cross-fade ──
        if not _ignis_initialized and _load_stage == LOAD_IDLE:
            _load_stage = LOAD_FADEIN
            _load_start_time = time.perf_counter()
            loading_screen.reset()
            loading_screen.draw(w, h, "", 0.0)
            self.tag_redraw()
            return

        # ── Loading screen: show and process one stage per call ──
        if _load_stage != LOAD_IDLE:
            loading_screen.draw(w, h, _load_status, _load_progress)
            try:
                done = self._tick_staged_load(depsgraph)
            except Exception:
                _log_exception("view_draw -> _tick_staged_load")
                _load_stage = LOAD_IDLE
                done = True
            self.tag_redraw()
            return

        # ── Full scene upload needed? Start staged loading with fade-in ──
        if _ignis_full_dirty:
            now = time.perf_counter()
            time_since_last = now - _ignis_last_full_upload if _ignis_last_full_upload > 0 else 999.0
            if _ignis_last_full_upload == 0 or time_since_last >= 5.0:
                _log(f"full_dirty triggered ({time_since_last:.1f}s since last)")
                self._begin_staged_load(depsgraph)
                loading_screen.draw(w, h, "Preparing scene...", 0.0)
                self.tag_redraw()
                return
            else:
                _ignis_full_dirty = False

        # ── Fast incremental updates (no loading screen needed) ──
        if _ignis_materials_dirty:
            # Skip auto material update after initial load — prevents material index
            # reordering that breaks per-BLAS material IDs. Only update on explicit
            # user action (Reload Scene button) via full_dirty.
            _ignis_materials_dirty = False
        elif _ignis_tlas_dirty:
            self._rebuild_tlas(depsgraph)
        # ── Light sync (every frame — cheap) ──
        sun = scene_export.export_sun(depsgraph)
        if sun["sun_intensity"] <= 0.0 and _ignis_hdri_sun:
            sun = _ignis_hdri_sun
        dll_wrapper.set_float("sun_elevation", sun["sun_elevation"])
        dll_wrapper.set_float("sun_azimuth", sun["sun_azimuth"])
        dll_wrapper.set_float("sun_intensity", sun["sun_intensity"])
        sun_color = sun.get("sun_color", (1.0, 1.0, 1.0))
        dll_wrapper.set_float("sun_color_r", sun_color[0])
        dll_wrapper.set_float("sun_color_g", sun_color[1])
        dll_wrapper.set_float("sun_color_b", sun_color[2])

        light_data = scene_export.export_lights(depsgraph)
        n_lights = len(light_data) // 16
        dll_wrapper.upload_lights(light_data, n_lights)

        # Track light count — emissive re-export is done in FINALIZE only (not per-frame)
        global _ignis_last_light_count
        _ignis_last_light_count = n_lights

        # ── Transform sync (detect moved/added/removed objects per frame) ──
        # Uses obj→mesh mapping for BLAS lookup. Handles:
        # - Moved objects (transform changed)
        # - New objects (obj.name not in mapping → upload BLAS incrementally)
        # - Removed objects (not in depsgraph → excluded from TLAS)
        global _ignis_last_transforms, _ignis_obj_to_mesh
        if _ignis_blas_handles is not None:
            cur_instances = []
            new_objects = {}  # obj_name → (obj, inst) for incremental upload
            _sync_obj_count = 0
            for inst in depsgraph.object_instances:
                obj = inst.object
                if obj.type != 'MESH':
                    continue
                if not inst.show_self:
                    continue
                if obj.name in _ignis_hidden_objects:
                    continue
                if obj.hide_viewport:
                    continue
                try:
                    if obj.hide_get():
                        continue
                except RuntimeError:
                    pass
                if hasattr(obj, 'visible_camera') and not obj.visible_camera:
                    continue
                _sync_obj_count += 1
                # Use library-qualified name (same as export_meshes)
                if obj.library:
                    obj_key = f"{obj.library.filepath}:{obj.name}"
                else:
                    obj_key = obj.name
                # Resolve obj_key → mesh_key → BLAS handle
                mesh_key = _ignis_obj_to_mesh.get(obj_key)
                if mesh_key is None:
                    mesh_key = obj_key  # try direct
                if mesh_key and mesh_key in _ignis_blas_handles:
                    xform = scene_export._matrix_to_3x4_row_major(inst.matrix_world)
                    cur_instances.append((mesh_key, xform))
                else:
                    # Unknown object — only upload if genuinely NEW (not a pre-existing
                    # collection instance or hidden object from initial load)
                    if obj_key not in _ignis_known_objects and obj_key not in new_objects:
                        new_objects[obj_key] = (obj, inst)

            # Incremental BLAS upload for new objects
            if new_objects:
                try:
                    self._upload_new_objects(depsgraph, new_objects)
                except Exception:
                    _log_exception("incremental mesh upload")
                    # Mark as known so we don't retry every frame
                    for obj_name in new_objects:
                        _ignis_known_objects.add(obj_name)
                # Re-scan to include newly uploaded objects
                cur_instances = []
                for inst in depsgraph.object_instances:
                    obj = inst.object
                    if obj.type != 'MESH':
                        continue
                    mesh_key = _ignis_obj_to_mesh.get(obj.name)
                    if mesh_key and mesh_key in _ignis_blas_handles:
                        xform = scene_export._matrix_to_3x4_row_major(inst.matrix_world)
                        cur_instances.append((mesh_key, xform))

            # Add hair instances (not in depsgraph — use parent object transform)
            for hair_key, blas_idx in _ignis_blas_handles.items():
                if '__hair_' in hair_key:
                    # Find parent object transform from existing instances
                    parent_name = hair_key.split('__hair_')[0]
                    for mk, xf in cur_instances:
                        if mk == parent_name:
                            cur_instances.append((hair_key, xf))
                            break

            # Compare with tolerance to avoid float jitter causing constant resets
            xform_floats = tuple(f for _, xf in cur_instances for f in xf)
            transforms_changed = len(cur_instances) != _ignis_instance_count
            if not transforms_changed and _ignis_last_transforms is not None:
                for a, b in zip(xform_floats, _ignis_last_transforms):
                    if abs(a - b) > 1e-5:
                        transforms_changed = True
                        break

            if _ignis_frame_index < 2:
                _log(f"  sync: {_sync_obj_count} mesh objects, {len(cur_instances)} known, {len(new_objects)} new, changed={transforms_changed}")

            if transforms_changed or new_objects:
                _ignis_last_transforms = xform_floats
                count = len(cur_instances)
                if count > 0:
                    InstanceArray = TLASInstance * count
                    tlas_arr = InstanceArray()
                    for i, (mesh_key, xform) in enumerate(cur_instances):
                        blas_idx = _ignis_blas_handles[mesh_key]
                        tlas_arr[i].blasIndex = blas_idx
                        for j in range(12):
                            tlas_arr[i].transform[j] = xform[j]
                        tlas_arr[i].customIndex = blas_idx
                        tlas_arr[i].mask = 0xFF
                    dll_wrapper.build_tlas(tlas_arr, count)
                    # Reset denoiser only when objects added/removed (count changed),
                    # NOT on transform-only changes (NRD anti-lag handles those naturally)
                    if count != _ignis_instance_count or new_objects:
                        _ignis_frame_index = 0
                        dll_wrapper.set_int("reset_history", 1)
                    _ignis_instance_count = count

        # FPS limiter
        props = context.scene.ignis_rt
        fps_limit = props.fps_limit
        if fps_limit > 0:
            min_frame_time = 1.0 / fps_limit
            now = time.perf_counter()
            if _fps_times:
                elapsed = now - _fps_times[-1]
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)

        # Sync scene properties
        props = context.scene.ignis_rt
        dll_wrapper.set_int("max_bounces", props.max_bounces)
        dll_wrapper.set_int("spp", props.samples_per_pixel)
        dll_wrapper.set_int("backface_culling", 1 if props.backface_culling else 0)
        dll_wrapper.set_int("restir_di", 1 if props.restir_di else 0)
        dll_wrapper.set_int("debug_view", props.debug_view)
        dll_wrapper.set_int("auto_sky_colors", 1 if props.auto_sky_colors else 0)
        if not props.auto_sky_colors:
            dll_wrapper.set_float("ambient_intensity", props.ambient_intensity)
            dll_wrapper.set_float("ambient_color_r", props.ambient_color[0])
            dll_wrapper.set_float("ambient_color_g", props.ambient_color[1])
            dll_wrapper.set_float("ambient_color_b", props.ambient_color[2])
        dll_wrapper.set_float("sky_refl_intensity", props.sky_refl_intensity)
        dll_wrapper.set_float("sky_bounce_intensity", props.sky_bounce_intensity)
        dll_wrapper.set_float("cloud_visibility", props.cloud_visibility)
        dll_wrapper.set_float("exposure", props.exposure)
        dll_wrapper.set_int("tonemap_mode", props.tonemap_mode)
        dll_wrapper.set_float("saturation", props.saturation)
        dll_wrapper.set_float("contrast", props.contrast)

        # Camera
        try:
            cam = scene_export.export_camera(context)
            dll_wrapper.set_camera(
                cam["view_inv"], cam["proj_inv"],
                cam["view"], cam["proj"],
                _ignis_frame_index,
            )

            # Render
            dll_wrapper.render_frame()

            # Display
            gl_ok = dll_wrapper.draw_gl(w, h)
            if not gl_ok:
                # Use actual Vulkan render size (may differ from viewport due to min-1920 enforcement)
                rw = dll_wrapper.get_int("render_width") or w
                rh = dll_wrapper.get_int("render_height") or h
                pixel_count = rw * rh
                # Ensure float buffer matches render size
                global _ignis_float_buffer
                needed = pixel_count * 4
                if _ignis_float_buffer is None or len(_ignis_float_buffer) < needed:
                    _ignis_float_buffer = (ctypes.c_float * needed)()
                ok = dll_wrapper.readback_float(_ignis_float_buffer, pixel_count)
                if ok:
                    global _ignis_gpu_texture
                    gpu_buf = gpu.types.Buffer('FLOAT', [pixel_count * 4], _ignis_float_buffer)
                    _ignis_gpu_texture = gpu.types.GPUTexture((rw, rh), format='RGBA32F', data=gpu_buf)
                    _draw_texture_flipped(_ignis_gpu_texture, w, h)
                    if _ignis_frame_index < 3:
                        _log(f"CPU readback OK ({rw}x{rh} -> {w}x{h})")
                elif _ignis_gpu_texture is not None:
                    _draw_texture_flipped(_ignis_gpu_texture, w, h)

            _ignis_frame_index += 1
        except Exception:
            _log_exception(f"view_draw render/display (frame {_ignis_frame_index})")

        # Loading screen fade-out overlay (cross-fade from loader to render)
        if loading_screen.draw_fadeout(w, h):
            pass  # still fading — will redraw

        # FPS overlay
        if props.show_fps:
            _update_fps()
            _draw_fps_overlay(w, h)

        # Always request next frame — NRD needs continuous accumulation to converge
        self.tag_redraw()

    # Final render (F12)
    def update(self, data, depsgraph):
        pass

    def render(self, depsgraph):
        pass
