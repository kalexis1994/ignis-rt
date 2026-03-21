"""Ignis RT loading screen — logo, spinner, progress bar, status text."""

import math
import os
import time

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

# ── Ignis Palette (from icons/palette.json) ──
COL_BG        = (0.055, 0.055, 0.063, 1.0)    # darkness-700 #1C1C1E
COL_SUNSET    = (0.878, 0.376, 0.188)          # sunset-500 #E06030
COL_GOLDEN    = (0.831, 0.722, 0.478)          # golden #D4B87A
COL_SUNSET_DK = (0.478, 0.196, 0.094)          # sunset-900 #7A3218
COL_TEXT_RGB  = (0.863, 0.890, 0.929)           # sky-50 #DCE4ED
COL_DIM_RGB   = (0.561, 0.647, 0.769)           # sky-500 #8FA7C4
COL_BAR_BG_RGB = (0.180, 0.180, 0.190)          # darkness-500 #2E2E30

# ── Cached resources ──
_logo_texture = None
_font_id = None
_start_time = 0.0


def _get_logo_texture():
    """Load the icon texture for the loading screen (cached)."""
    global _logo_texture
    if _logo_texture is not None:
        return _logo_texture
    try:
        icons_dir = os.path.join(os.path.dirname(__file__), "icons")
        logo_path = os.path.join(icons_dir, "ignis_icon.png")
        if not os.path.isfile(logo_path):
            return None
        img = bpy.data.images.load(logo_path, check_existing=True)
        img.gl_load()
        _logo_texture = gpu.texture.from_image(img)
        return _logo_texture
    except Exception:
        return None


def _get_font_id():
    """Load Nova Round font (cached). Returns blf font id."""
    global _font_id
    if _font_id is not None:
        return _font_id
    try:
        font_path = os.path.join(os.path.dirname(__file__), "icons", "NovaRound-Regular.ttf")
        if os.path.isfile(font_path):
            _font_id = blf.load(font_path)
        else:
            _font_id = 0
    except Exception:
        _font_id = 0
    return _font_id


def _draw_spinner(shader, cx, cy, t, radius=22, segments=48, ring_width=4.0):
    """Draw a sunset-colored spinning ring using the Ignis palette."""
    for i in range(segments):
        frac = i / segments
        angle = 2 * math.pi * frac - t * 4.0

        sweep = (frac + t * 4.0 / (2 * math.pi)) % 1.0
        intensity = pow(1.0 - sweep, 2.0)

        # Palette gradient: golden → sunset-500 → sunset-900
        if intensity > 0.5:
            t2 = (intensity - 0.5) * 2.0
            r_col = COL_SUNSET[0] + t2 * (COL_GOLDEN[0] - COL_SUNSET[0])
            g_col = COL_SUNSET[1] + t2 * (COL_GOLDEN[1] - COL_SUNSET[1])
            b_col = COL_SUNSET[2] + t2 * (COL_GOLDEN[2] - COL_SUNSET[2])
        else:
            t2 = intensity * 2.0
            r_col = COL_SUNSET_DK[0] + t2 * (COL_SUNSET[0] - COL_SUNSET_DK[0])
            g_col = COL_SUNSET_DK[1] + t2 * (COL_SUNSET[1] - COL_SUNSET_DK[1])
            b_col = COL_SUNSET_DK[2] + t2 * (COL_SUNSET[2] - COL_SUNSET_DK[2])
        alpha = max(0.0, min(1.0, intensity * 1.5))

        dot_r = ring_width * (0.4 + 0.6 * intensity)
        dx = cx + math.cos(angle) * radius
        dy = cy + math.sin(angle) * radius

        verts = [(dx, dy)]
        for s in range(13):
            a = 2 * math.pi * s / 12
            verts.append((dx + math.cos(a) * dot_r, dy + math.sin(a) * dot_r))
        dot_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
        shader.bind()
        shader.uniform_float("color", (r_col, g_col, b_col, alpha))
        dot_batch.draw(shader)


def draw(w, h, status="", progress=0.0):
    """Draw the Ignis RT loading screen."""
    global _start_time

    now = time.perf_counter()
    if _start_time <= 0.0:
        _start_time = now
    fade = min((now - _start_time) / 0.6, 1.0)

    col_flame = (*COL_SUNSET, fade)
    col_text = (*COL_TEXT_RGB, fade)
    col_text_dim = (*COL_DIM_RGB, fade)
    col_bar_bg = (*COL_BAR_BG_RGB, fade)

    gpu.state.blend_set('NONE')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)

    # Background
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    bg = ((0, 0), (w, 0), (w, h), (0, h))
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": bg})
    shader.bind()
    shader.uniform_float("color", COL_BG)
    batch.draw(shader)

    cx, cy = w / 2, h / 2
    gpu.state.blend_set('ALPHA')

    # Logo
    logo_tex = _get_logo_texture()
    if logo_tex and fade > 0.01:
        logo_size = 100
        lx = cx - logo_size / 2
        ly = cy + 25
        img_shader = gpu.shader.from_builtin('IMAGE')
        img_batch = batch_for_shader(img_shader, 'TRI_FAN', {
            "pos": ((lx, ly), (lx + logo_size, ly), (lx + logo_size, ly + logo_size), (lx, ly + logo_size)),
            "texCoord": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        })
        img_shader.bind()
        img_shader.uniform_sampler("image", logo_tex)
        img_batch.draw(img_shader)

        if fade < 1.0:
            shader.bind()
            shader.uniform_float("color", (COL_BG[0], COL_BG[1], COL_BG[2], 1.0 - fade))
            cover = ((lx, ly), (lx + logo_size, ly), (lx + logo_size, ly + logo_size), (lx, ly + logo_size))
            batch_for_shader(shader, 'TRI_FAN', {"pos": cover}).draw(shader)

        title_y = ly - 8
    else:
        title_y = cy + 30

    # Title
    font_id = _get_font_id()
    blf.size(font_id, 48)
    blf.color(font_id, *col_text)
    title = "Ignis RT"
    tw, _ = blf.dimensions(font_id, title)
    blf.position(font_id, cx - tw / 2, title_y - 48, 0)
    blf.draw(font_id, title)

    # Spinner
    spinner_y = title_y - 100
    _draw_spinner(shader, cx, spinner_y, time.perf_counter())

    # Status text
    if status:
        blf.size(font_id, 18)
        blf.color(font_id, *col_text_dim)
        sw = blf.dimensions(font_id, status)[0]
        blf.position(font_id, cx - sw / 2, spinner_y - 55, 0)
        blf.draw(font_id, status)

    # Progress bar
    if progress > 0.0:
        bar_w = 240
        bar_h = 4
        bx = cx - bar_w / 2
        by = spinner_y - 75
        bg_verts = ((bx, by), (bx + bar_w, by), (bx + bar_w, by + bar_h), (bx, by + bar_h))
        bg_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": bg_verts})
        shader.bind()
        shader.uniform_float("color", col_bar_bg)
        bg_batch.draw(shader)
        fill_w = bar_w * min(progress, 1.0)
        fill_verts = ((bx, by), (bx + fill_w, by), (bx + fill_w, by + bar_h), (bx, by + bar_h))
        fill_batch = batch_for_shader(shader, 'TRI_FAN', {"pos": fill_verts})
        shader.bind()
        shader.uniform_float("color", col_flame)
        fill_batch.draw(shader)

    gpu.state.blend_set('NONE')


def reset():
    """Reset the loading screen timer (call before starting a new load)."""
    global _start_time
    _start_time = 0.0
