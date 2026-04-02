"""Performance overlay — FPS counter, frame time graph, and stats."""

import time

import blf
import gpu
from gpu_extras.batch import batch_for_shader

WINDOW_SEC = 5.0  # show last N seconds of frame history

# ── CPU section timing ───────────────────────────────────────────────
_cpu_marks = {}    # name → perf_counter (begin marks)
_cpu_times = {}    # name → ms (latest completed section)

def begin(name):
    """Mark the start of a CPU section."""
    _cpu_marks[name] = time.perf_counter()

def end(name):
    """Mark the end of a CPU section, compute elapsed ms."""
    t0 = _cpu_marks.get(name)
    if t0 is not None:
        _cpu_times[name] = (time.perf_counter() - t0) * 1000.0

def cpu_time(name):
    """Get the last recorded time for a CPU section (ms)."""
    return _cpu_times.get(name, 0.0)


# Rolling history — (timestamp, frame_time_ms) pairs
_samples = []      # list of (perf_counter, dt_ms)
_prev_time = 0.0
_fps = 0.0
_stats = {
    'avg': 0.0,       # windowed average (5s)
    'min': 0.0,        # session absolute min frame time
    'max': 0.0,        # session absolute max frame time
    'p1': 0.0,         # windowed 99th percentile (1% low)
}
_session_min_ms = 999999.0   # absolute min across entire session
_session_max_ms = 0.0        # absolute max across entire session
_session_sum_ms = 0.0        # sum for session average
_session_count = 0           # frame count for session average


def reset():
    """Clear all history (call on renderer destroy)."""
    global _samples, _prev_time, _fps
    global _session_min_ms, _session_max_ms, _session_sum_ms, _session_count
    _samples.clear()
    _prev_time = 0.0
    _fps = 0.0
    _stats.update(avg=0.0, min=0.0, max=0.0, p1=0.0)
    _session_min_ms = 999999.0
    _session_max_ms = 0.0
    _session_sum_ms = 0.0
    _session_count = 0


def get_frame_times():
    """Return list of frame times in ms (for graph)."""
    return [s[1] for s in _samples]


def get_fps():
    """Return current smoothed FPS."""
    return _fps


def get_frame_ms():
    """Return current frame time in ms."""
    return 1000.0 / max(_fps, 0.001) if _fps > 0 else 0.0


def update():
    """Record a frame and recompute stats. Call once per frame."""
    global _prev_time, _fps
    global _session_min_ms, _session_max_ms, _session_sum_ms, _session_count

    now = time.perf_counter()
    if _prev_time > 0:
        dt = (now - _prev_time) * 1000.0
        _samples.append((now, dt))

        # Session-absolute tracking
        _session_min_ms = min(_session_min_ms, dt)
        _session_max_ms = max(_session_max_ms, dt)
        _session_sum_ms += dt
        _session_count += 1
    _prev_time = now

    # Trim to time window
    cutoff = now - WINDOW_SEC
    while _samples and _samples[0][0] < cutoff:
        _samples.pop(0)

    n = len(_samples)
    if n >= 2:
        # EMA-smoothed FPS
        last_dt = _samples[-1][1]
        instant_fps = 1000.0 / last_dt if last_dt > 0 else 0
        alpha = 0.1
        _fps = _fps + alpha * (instant_fps - _fps) if _fps > 0 else instant_fps

        # Windowed avg (5s) + session absolute min/max
        times = [s[1] for s in _samples]
        _stats['avg'] = sum(times) / n
        _stats['min'] = _session_min_ms if _session_min_ms < 999999.0 else 0.0
        _stats['max'] = _session_max_ms
        sorted_ft = sorted(times)
        _stats['p1'] = sorted_ft[min(int(n * 0.99), n - 1)]


# ── Drawing ──────────────────────────────────────────────────────────

_COLOR_BG = (0.05, 0.05, 0.05, 0.82)
_COLOR_GRAPH_BG = (0.1, 0.1, 0.1, 0.9)
_COLOR_GREEN = (0.2, 0.8, 0.3, 0.9)
_COLOR_YELLOW = (0.9, 0.8, 0.2, 0.9)
_COLOR_RED = (0.9, 0.2, 0.2, 0.9)
_LINE_60 = (0.2, 0.5, 0.2, 0.5)
_LINE_30 = (0.5, 0.5, 0.2, 0.4)

_PAD = 10
_GRAPH_W = 280
_GRAPH_H = 60
_TEXT_H = 90
_TOTAL_W = _GRAPH_W + _PAD * 2
_TOTAL_H = _TEXT_H + _GRAPH_H + _PAD * 3


def _bar_color(ms):
    if ms < 16.67:
        return _COLOR_GREEN
    elif ms < 33.33:
        return _COLOR_YELLOW
    return _COLOR_RED


def _draw_rect(shader, x, y, w, h, color):
    verts = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_graph(shader, gx, gy):
    """Draw the frame time bar graph."""
    ft = get_frame_times()
    n = len(ft)
    if n < 2:
        return

    _draw_rect(shader, gx, gy, _GRAPH_W, _GRAPH_H, _COLOR_GRAPH_BG)

    # Y-axis scale
    visible = ft[-min(n, _GRAPH_W):]
    max_ms = max(max(visible), 33.3, _stats['max']) * 1.1

    # Target lines (60fps, 30fps)
    for target_ms, col in [(16.67, _LINE_60), (33.33, _LINE_30)]:
        ty = gy + (target_ms / max_ms) * _GRAPH_H
        if ty < gy + _GRAPH_H:
            lv = ((gx, ty), (gx + _GRAPH_W, ty))
            b = batch_for_shader(shader, 'LINES', {"pos": lv})
            shader.bind()
            shader.uniform_float("color", col)
            b.draw(shader)

    # Bars
    bar_w = max(_GRAPH_W / len(visible), 1.0)
    for i, ms in enumerate(visible):
        bx = gx + i * bar_w
        bh = min((ms / max_ms) * _GRAPH_H, _GRAPH_H)
        _draw_rect(shader, bx, gy, bar_w, bh, _bar_color(ms))


def _overlay_height(mode):
    """Return overlay height for a given mode."""
    if mode == 'FPS':
        return 28
    elif mode == 'MS':
        return 28
    elif mode == 'STATS':
        return 54
    else:  # FULL
        return _TEXT_H + _GRAPH_H + _PAD * 3


def _anchor(w, h, bw, bh, pos, y_offset=0):
    """Compute x0, y0 for a box of size bw x bh at the given anchor position."""
    margin = 8
    # Horizontal
    if pos[1] == 'L':
        x0 = margin
    elif pos[1] == 'C':
        x0 = (w - bw) // 2
    else:  # R
        x0 = w - bw - margin
    # Vertical
    if pos[0] == 'T':
        y0 = h - bh - margin - y_offset
    else:  # B
        y0 = margin + y_offset
    return x0, y0


def draw_fps(w, h, mode, pos='TR'):
    """Draw FPS overlay. mode: 'FPS', 'MS', 'STATS', 'FULL'. pos: 'TL','TC','TR','BL','BC','BR'."""
    fid = 0
    avg = _stats['avg']
    lo = _stats['min']
    hi = _stats['max']
    p1 = _stats['p1']
    p1_fps = 1000.0 / p1 if p1 > 0 else 0

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    if mode == 'FPS':
        # Compact: "60 FPS"
        bw, bh = 90, 28
        x0, y0 = _anchor(w, h, bw, bh, pos)
        _draw_rect(shader, x0, y0, bw, bh, _COLOR_BG)
        blf.size(fid, 16)
        blf.color(fid, 1.0, 1.0, 1.0, 1.0)
        blf.position(fid, x0 + 8, y0 + 7, 0)
        blf.draw(fid, f"{_fps:.0f} FPS")

    elif mode == 'MS':
        # Compact: "60 FPS  16.7 ms"
        bw, bh = 180, 28
        x0, y0 = _anchor(w, h, bw, bh, pos)
        _draw_rect(shader, x0, y0, bw, bh, _COLOR_BG)
        blf.size(fid, 16)
        blf.color(fid, 1.0, 1.0, 1.0, 1.0)
        blf.position(fid, x0 + 8, y0 + 7, 0)
        blf.draw(fid, f"{_fps:.0f} FPS")
        blf.size(fid, 13)
        blf.color(fid, 0.7, 0.7, 0.7, 1.0)
        blf.position(fid, x0 + 80, y0 + 8, 0)
        blf.draw(fid, f"{avg:.1f} ms")

    elif mode == 'STATS':
        # Two rows: FPS + ms, then Min/Max/1%
        bw, bh = 280, 54
        x0, y0 = _anchor(w, h, bw, bh, pos)
        _draw_rect(shader, x0, y0, bw, bh, _COLOR_BG)
        tx = x0 + 8

        # Row 1: FPS + ms + 1% low
        blf.size(fid, 16)
        blf.color(fid, 1.0, 1.0, 1.0, 1.0)
        blf.position(fid, tx, y0 + 30, 0)
        blf.draw(fid, f"{_fps:.0f} FPS")
        blf.size(fid, 13)
        blf.color(fid, 0.7, 0.7, 0.7, 1.0)
        blf.position(fid, tx + 72, y0 + 31, 0)
        blf.draw(fid, f"{avg:.1f} ms")
        blf.color(fid, 0.9, 0.5, 0.3, 1.0)
        blf.position(fid, tx + 148, y0 + 31, 0)
        blf.draw(fid, f"1%Low {p1_fps:.0f}")

        # Row 2: Min / Max / 1%
        blf.size(fid, 12)
        cw = 65

        blf.color(fid, 0.3, 0.9, 0.4, 1.0)
        blf.position(fid, tx, y0 + 8, 0)
        blf.draw(fid, f"Min {lo:.1f}")

        blf.color(fid, 0.9, 0.9, 0.3, 1.0)
        blf.position(fid, tx + cw, y0 + 8, 0)
        blf.draw(fid, f"Max {hi:.1f}")

        blf.color(fid, 0.9, 0.4, 0.3, 1.0)
        blf.position(fid, tx + cw * 2, y0 + 8, 0)
        blf.draw(fid, f"1% {p1:.1f}")

        blf.color(fid, 0.5, 0.5, 0.5, 1.0)
        blf.position(fid, tx + cw * 3, y0 + 8, 0)
        blf.draw(fid, "ms")

    else:  # FULL
        panel_w = _GRAPH_W + _PAD * 2
        panel_h = _TEXT_H + _GRAPH_H + _PAD * 3
        x0, y0 = _anchor(w, h, panel_w, panel_h, pos)

        _draw_rect(shader, x0, y0, panel_w, panel_h, _COLOR_BG)

        gx = x0 + _PAD
        gy = y0 + _PAD
        _draw_graph(shader, gx, gy)

        tx = x0 + _PAD
        ty = gy + _GRAPH_H + _PAD + 4

        # FPS + avg
        blf.size(fid, 22)
        blf.color(fid, 1.0, 1.0, 1.0, 1.0)
        blf.position(fid, tx, ty + 56, 0)
        blf.draw(fid, f"{_fps:.0f} FPS")
        blf.size(fid, 14)
        blf.color(fid, 0.7, 0.7, 0.7, 1.0)
        blf.position(fid, tx + 90, ty + 60, 0)
        blf.draw(fid, f"{avg:.1f} ms")

        # Min / Max / 1%
        blf.size(fid, 13)
        cw = 68
        blf.color(fid, 0.3, 0.9, 0.4, 1.0)
        blf.position(fid, tx, ty + 30, 0)
        blf.draw(fid, f"Min {lo:.1f}")
        blf.color(fid, 0.9, 0.9, 0.3, 1.0)
        blf.position(fid, tx + cw, ty + 30, 0)
        blf.draw(fid, f"Max {hi:.1f}")
        blf.color(fid, 0.9, 0.4, 0.3, 1.0)
        blf.position(fid, tx + cw * 2, ty + 30, 0)
        blf.draw(fid, f"1% {p1:.1f}")
        blf.color(fid, 0.6, 0.6, 0.6, 1.0)
        blf.position(fid, tx + cw * 3, ty + 30, 0)
        blf.draw(fid, "ms")

        # 1% low FPS + frame count
        blf.color(fid, 0.9, 0.5, 0.3, 1.0)
        blf.position(fid, tx, ty + 8, 0)
        blf.draw(fid, f"1% Low: {p1_fps:.0f} FPS")
        blf.color(fid, 0.5, 0.5, 0.5, 1.0)
        blf.position(fid, tx + 140, ty + 8, 0)
        blf.draw(fid, f"({len(_samples)} frames)")

    gpu.state.blend_set('NONE')


# ── GPU Profiler Overlay ─────────────────────────────────────────────

_GPU_BAR_W = 200
_GPU_ROW_H = 18
_GPU_PAD = 8
_GPU_LABEL_W = 100

def draw_gpu_profiler(w, h, fps_visible, fps_mode='FULL', pos='TR'):
    """Draw GPU + CPU timing breakdown at top-right, below FPS overlay if visible."""
    from . import dll_wrapper

    hybrid_ms = dll_wrapper.get_float("gpu_time_hybrid")
    rt_ms = dll_wrapper.get_float("gpu_time_rt")
    hair_ms = dll_wrapper.get_float("gpu_time_hair")
    denoise_ms = dll_wrapper.get_float("gpu_time_denoise")
    composite_ms = dll_wrapper.get_float("gpu_time_composite")
    tonemap_ms = dll_wrapper.get_float("gpu_time_tonemap")
    total_gpu = dll_wrapper.get_float("gpu_time_total")

    sync_ms = cpu_time("sync")
    render_ms = cpu_time("render")
    gpu_call_ms = cpu_time("gpu")
    interop_ms = cpu_time("interop")

    sections = [
        # (label, ms, color, is_header)
        ("GPU", None, None, True),
        ("Hybrid Raster", hybrid_ms, (0.6, 0.9, 0.3, 0.9), False),
        ("RT Trace", rt_ms, (0.3, 0.7, 1.0, 0.9), False),
        ("Hair+Resolve", hair_ms, (0.8, 0.6, 0.3, 0.9), False),
        ("Denoise", denoise_ms, (0.9, 0.4, 0.7, 0.9), False),
        ("Composite", composite_ms, (0.9, 0.6, 0.2, 0.9), False),
        ("Tonemap", tonemap_ms, (0.7, 0.7, 0.3, 0.9), False),
        ("GPU Total", total_gpu, (0.8, 0.8, 0.8, 0.9), False),
        ("CPU", None, None, True),
        ("Scene Sync", sync_ms, (0.4, 0.9, 0.5, 0.9), False),
        ("Camera+Prep", render_ms, (0.5, 0.8, 0.9, 0.9), False),
        ("Render Call", gpu_call_ms, (0.3, 0.7, 1.0, 0.9), False),
        ("GL Interop", interop_ms, (0.9, 0.5, 0.8, 0.9), False),
        ("Blender Gap", cpu_time("blender"), (0.6, 0.4, 0.4, 0.9), False),
    ]

    rows = len(sections)
    panel_w = _GPU_LABEL_W + _GPU_BAR_W + 70 + _GPU_PAD * 2
    panel_h = rows * _GPU_ROW_H + _GPU_PAD * 2 + 8

    y_offset = (_overlay_height(fps_mode) + 10) if fps_visible else 0
    x0, y0 = _anchor(w, h, panel_w, panel_h, pos, y_offset)

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    _draw_rect(shader, x0, y0, panel_w, panel_h, _COLOR_BG)

    fid = 0
    tx = x0 + _GPU_PAD
    ty = y0 + panel_h - _GPU_PAD - 4

    # Scale: max of total GPU or 33ms
    max_ms = max(total_gpu, gpu_call_ms, 16.67, 1.0)

    for label, ms, color, is_header in sections:
        if is_header:
            blf.size(fid, 13)
            blf.color(fid, 0.8, 0.8, 0.8, 1.0)
            blf.position(fid, tx, ty, 0)
            blf.draw(fid, label)
            ty -= _GPU_ROW_H
            continue

        blf.size(fid, 12)
        # Label
        blf.color(fid, *color)
        blf.position(fid, tx + 8, ty, 0)
        blf.draw(fid, label)

        # Bar
        bar_x = tx + _GPU_LABEL_W
        bar_frac = min(ms / max_ms, 1.0) if max_ms > 0 else 0
        bar_px = max(bar_frac * _GPU_BAR_W, 1.0)
        _draw_rect(shader, bar_x, ty - 2, bar_px, 12, color)
        _draw_rect(shader, bar_x + bar_px, ty - 2, _GPU_BAR_W - bar_px, 12, (0.15, 0.15, 0.15, 0.6))

        # Value
        blf.color(fid, 1.0, 1.0, 1.0, 1.0)
        blf.position(fid, bar_x + _GPU_BAR_W + 6, ty, 0)
        blf.draw(fid, f"{ms:.1f} ms")

        ty -= _GPU_ROW_H

    gpu.state.blend_set('NONE')
