"""Ignis RT loading screen — logo, spinner, progress bar, status text."""

import math
import os
import time

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

# ── Ignis Palette (from icons/palette.json) ──
COL_BG        = (0.003, 0.003, 0.004, 1.0)    # darkness-900 #0E0E10 (sRGB→linear)
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
CROSSFADE_DURATION = 0.6  # seconds for fade-in of loading screen elements

# ── Custom progress bar shader (rounded pill with specular highlight) ──
_fire_shader = None
_fire_shader_failed = False

_FIRE_VERT_SRC = (
    "void main() {"
    "  vUV = uv;"
    "  gl_Position = mvp * vec4(pos, 0.0, 1.0);"
    "}"
)

_FIRE_FRAG_SRC = """
// Ignis diamond SDF (asymmetric: taller top, shorter bottom, like the logo)
float sdDiamond(vec2 p, float hw, float hTop, float hBot) {
    float ax = abs(p.x);
    float ry = (p.y > 0.0) ? hTop : hBot;  // y+ = top in screen
    return ax / hw + abs(p.y) / ry - 1.0;
}

vec2 hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec2 p) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;
    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    vec2 o = (a.x > a.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hash(i)), dot(b, hash(i + o)), dot(c, hash(i + 1.0)));
    return dot(n, vec3(70.0));
}

float fbm(vec2 uv) {
    float f;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    f  = 0.5000 * noise(uv); uv = m * uv;
    f += 0.2500 * noise(uv); uv = m * uv;
    f += 0.1250 * noise(uv); uv = m * uv;
    f += 0.0625 * noise(uv);
    return 0.5 + 0.5 * f;
}

void main() {
    // Map UV to centered [-1, 1] space
    vec2 p = vUV * 2.0 - 1.0;
    float dist = length(p);

    // Ring parameters
    float ringRadius = 0.55;
    float ringWidth = 0.22;

    // Head position (clockwise)
    float headAngle = time * 4.0;
    vec2 headPos = vec2(cos(headAngle), -sin(headAngle)) * ringRadius;

    // Distance from pixel to head
    float headDist = length(p - headPos);

    // Angular distance behind the head (clockwise trail)
    float PI2 = 6.28318;
    float angle = atan(-p.y, p.x);  // negate Y to match clockwise rotation
    float angDist = mod(headAngle - angle, PI2);  // 0 at head, increases behind (clockwise)

    // ── Diamond head ──
    float moveAngle = -headAngle + 3.14159;
    vec2 localHead = p - headPos;
    float cs = cos(moveAngle), sn = sin(moveAngle);
    vec2 rotHead = vec2(localHead.x * cs + localHead.y * sn,
                       -localHead.x * sn + localHead.y * cs);
    float dSize = 0.13;
    float dd = sdDiamond(rotHead, dSize * 0.39, dSize * 1.0, dSize * 0.48);
    float ball = smoothstep(0.02, -0.02, dd);

    // ── Flame trail with turbulent rotation (reference shader technique) ──
    float radial = dist - ringRadius;

    float flame = 0.0;
    float trailMask = smoothstep(2.8, 0.0, angDist);
    if (trailMask > 0.001) {
        // Map to flame-local coords:
        // fx = radial (perpendicular to arc), fy = arc distance from head
        vec2 fuv = vec2(radial * 4.0, angDist * 0.4);

        // Height variation
        float variationH = fbm(vec2(time * 0.3)) * 1.1;

        // Flame turbulence: rotate UVs based on noise (the key technique)
        float f = fbm(fuv * 0.8 + vec2(0.0, -time * 0.6));
        float l = max(0.1, length(fuv));
        float rotAmt = ((f - 0.5) / l) * smoothstep(-0.1, 0.5, fuv.y) * 0.6;
        float cr = cos(rotAmt), sr = sin(rotAmt);
        fuv = vec2(fuv.x * cr - fuv.y * sr, fuv.x * sr + fuv.y * cr);

        // Flame thickness (wider near head)
        flame = 1.3 - abs(fuv.x) * 3.5;

        // Flame height (fades along trail)
        flame *= smoothstep(1.0, variationH * 0.4, fuv.y);
        flame = clamp(flame, 0.0, 1.0);
        flame = pow(flame, 2.5);
        flame *= trailMask;
    }

    // ── Integration: diamond merges into flame via soft glow around diamond ──
    float diamondGlow = smoothstep(0.08, -0.03, dd);  // soft halo around diamond
    float c = max(diamondGlow, flame);
    c = clamp(c, 0.0, 1.0);

    // Fire color: use Ignis icon palette
    // Head: sunset warm (E06030) → trail: sky cool (8FA7C4)
    // Transition based on angular distance from head
    float colorT = clamp(angDist / 1.2, 0.0, 1.0);  // 0=head, 1=tail — faster transition to blue
    vec3 warmCol = fillCol;                           // sunset-500
    vec3 coolCol = vec3(0.561, 0.655, 0.769);         // sky-500 #8FA7C4
    vec3 darkCol = vec3(0.180, 0.180, 0.188);          // darkness-500 #2E2E30
    vec3 fireCol = mix(warmCol, coolCol, colorT);
    // Intensity gradient: bright core → dim tail
    vec3 col = fireCol * c * (1.5 - colorT * 0.8);

    // Hot diamond core (white-hot center)
    float hotCore = smoothstep(0.01, -0.04, dd);
    col += vec3(1.0, 0.92, 0.65) * hotCore * 0.9;

    // ── Bloom flash: diamond-shaped pulse radiating from center ──
    // Use the same rotated local coords as the diamond head
    float flashDD = sdDiamond(rotHead, dSize * 0.39, dSize * 1.0, dSize * 0.48);

    // Expanding diamond glow that pulses
    float pulseT = fract(time * 1.3);
    float pulseI = (1.0 - pulseT) * (1.0 - pulseT);
    float glowR = 0.02 + pulseT * 0.15;
    float dBloom = smoothstep(glowR + 0.03, glowR - 0.01, flashDD);
    dBloom -= ball;
    dBloom = max(0.0, dBloom) * pulseI;

    col += vec3(1.0, 0.85, 0.5) * dBloom * 2.0;

    // Outer glow around head
    float outerGlow = exp(-headDist * headDist / 0.06) * 0.25;
    col += fillCol * outerGlow;

    float a = max(c, hotCore * 0.8 + outerGlow + dBloom * 0.4);
    a *= alpha;
    if (a < 0.002) discard;
    fragColor = vec4(col, a);
}
"""


def _get_fire_shader():
    global _fire_shader, _fire_shader_failed
    if _fire_shader is not None:
        return _fire_shader
    if _fire_shader_failed:
        return None
    try:
        iface = gpu.types.GPUStageInterfaceInfo("fire_iface")
        iface.smooth('VEC2', "vUV")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', "pos")
        info.vertex_in(1, 'VEC2', "uv")
        info.push_constant('MAT4', "mvp")
        info.push_constant('FLOAT', "time")
        info.push_constant('FLOAT', "alpha")
        info.push_constant('VEC3', "fillCol")
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source(_FIRE_VERT_SRC)
        info.fragment_source(_FIRE_FRAG_SRC)

        _fire_shader = gpu.shader.create_from_info(info)
        del iface, info
    except Exception as e:
        import traceback
        err_path = os.path.join(os.path.expanduser("~"), "ignis-shader-error.txt")
        with open(err_path, "w") as f:
            f.write(f"Fire shader error: {e}\n\n")
            traceback.print_exc(file=f)
        _fire_shader_failed = True
        _fire_shader = None
    return _fire_shader


_bar_shader = None
_bar_shader_failed = False

_BAR_VERT_SRC = (
    "void main()"
    "{"
    "  vUV = uv;"
    "  gl_Position = mvp * vec4(pos, 0.0, 1.0);"
    "}"
)

_BAR_FRAG_SRC = """
float sdBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) - r + min(max(q.x, q.y), 0.0);
}

void main() {
    vec2 p = vUV - 0.5;
    p.x *= aspect;
    float hw = aspect * 0.45;
    float hh = 0.45;
    float rad = hh;

    // Outer pill SDF
    float d = sdBox(p, vec2(hw, hh), rad);
    float fw = fwidth(d) * 1.5;

    // Outline: thin glowing line (sky-blue, subtle)
    vec3 outlineCol = vec3(0.3, 0.85, 1.0);
    float outlineGlow = pow(1.0 / max(abs(d) * 80.0, 0.01), 0.7);
    outlineGlow = clamp(outlineGlow, 0.0, 1.0) * 0.4;

    // Fill pill: inset with padding, rounded ends
    float fillPad = 0.25;
    float fillHH = hh - fillPad;
    float fillHWMax = hw - fillPad;
    float fillRad = fillHH;  // fully rounded ends

    // Fill extent: left edge fixed, right edge grows with progress
    float fillLeft = -fillHWMax;
    float fillRight = fillLeft + 2.0 * fillHWMax * progress;
    float fillCX = (fillLeft + fillRight) * 0.5;
    float fillEX = max((fillRight - fillLeft) * 0.5, 0.001);

    // Fill pill SDF (rounded rectangle = rounded ends)
    float fillD = sdBox(p - vec2(fillCX, 0.0), vec2(fillEX, fillHH), fillRad);

    // Fill line: glowing outline of the fill pill (sunset)
    float fillLine = 0.0;
    float fillInside = 0.0;
    if (progress > 0.005 && fillEX > 0.02) {
        fillLine = pow(1.0 / max(abs(fillD) * 60.0, 0.01), 0.7);
        fillLine = clamp(fillLine, 0.0, 1.0) * 0.6;
        // Soft fill inside
        fillInside = smoothstep(0.0, -0.05, fillD) * 0.2;
    }

    // Compose
    vec3 col = vec3(0.0);
    col += outlineCol * outlineGlow;
    col += fillCol * fillLine * 1.5;
    col += fillCol * fillInside;

    float a = max(outlineGlow, max(fillLine, fillInside));
    a = clamp(a * barAlpha, 0.0, 1.0);
    if (a < 0.002) discard;
    fragColor = vec4(col, a);
}
"""


def _get_bar_shader():
    global _bar_shader, _bar_shader_failed
    if _bar_shader is not None:
        return _bar_shader
    if _bar_shader_failed:
        return None
    try:
        iface = gpu.types.GPUStageInterfaceInfo("bar_iface")
        iface.smooth('VEC2', "vUV")

        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC2', "pos")
        info.vertex_in(1, 'VEC2', "uv")
        info.push_constant('MAT4', "mvp")
        info.push_constant('FLOAT', "progress")
        info.push_constant('FLOAT', "barAlpha")
        info.push_constant('FLOAT', "aspect")
        info.push_constant('VEC3', "fillCol")
        info.vertex_out(iface)
        info.fragment_out(0, 'VEC4', "fragColor")
        info.vertex_source(_BAR_VERT_SRC)
        info.fragment_source(_BAR_FRAG_SRC)

        _bar_shader = gpu.shader.create_from_info(info)
        del iface, info
    except Exception as e:
        import traceback
        err_path = os.path.join(os.path.expanduser("~"), "ignis-shader-error.txt")
        with open(err_path, "w") as f:
            f.write(f"Bar shader error: {e}\n\n")
            traceback.print_exc(file=f)
        _bar_shader_failed = True
        _bar_shader = None
    return _bar_shader

def _draw_pill(shader, x, y, w, h, color, segments=8):
    """Draw a pill-shaped (fully rounded) rectangle."""
    r = h / 2
    verts = []
    for i in range(segments + 1):
        angle = math.pi / 2 + math.pi * i / segments
        verts.append((x + r + math.cos(angle) * r, y + r + math.sin(angle) * r))
    for i in range(segments + 1):
        angle = -math.pi / 2 + math.pi * i / segments
        verts.append((x + w - r + math.cos(angle) * r, y + r + math.sin(angle) * r))
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_fill_bar(shader, x, y, w, h, color, segments=8):
    """Draw the fill portion: flat left edge, rounded right cap."""
    r = h / 2
    if w <= h:
        # Too small for round cap — just draw a rect
        verts = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)
        return
    verts = []
    # Flat left edge
    verts.append((x, y))
    verts.append((x, y + h))
    # Right semicircle cap
    for i in range(segments + 1):
        angle = math.pi / 2 - math.pi * i / segments
        verts.append((x + w - r + math.cos(angle) * r, y + r + math.sin(angle) * r))
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_progress_bar(uniform_shader, cx, spinner_y, progress, fade):
    """Draw a glass-style progress bar container with fill."""
    bar_w = 280
    bar_h = 16
    bx = cx - bar_w / 2
    by = spinner_y - 115

    bar_sh = _get_bar_shader()
    if bar_sh:
        # Add padding around the quad so SDF has room for AA on rounded edges
        # Full-screen quad — glow can extend freely, SDF handles the shape
        # Calculate UV mapping so (0,0)-(1,1) maps to the bar area
        import bpy
        sw = bpy.context.region.width if hasattr(bpy.context, 'region') and bpy.context.region else 1920
        sh = bpy.context.region.height if hasattr(bpy.context, 'region') and bpy.context.region else 1080
        u0 = -bx / bar_w
        v0 = -by / bar_h
        u1 = (sw - bx) / bar_w
        v1 = (sh - by) / bar_h
        bar_batch = batch_for_shader(bar_sh, 'TRI_FAN', {
            "pos": ((0, 0), (sw, 0), (sw, sh), (0, sh)),
            "uv": ((u0, v0), (u1, v0), (u1, v1), (u0, v1)),
        })
        gpu.state.blend_set('ALPHA')
        bar_sh.bind()
        bar_sh.uniform_float("mvp", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
        bar_sh.uniform_float("progress", min(progress, 1.0))
        bar_sh.uniform_float("barAlpha", fade)
        bar_sh.uniform_float("aspect", bar_w / bar_h)
        bar_sh.uniform_float("fillCol", COL_SUNSET)
        bar_batch.draw(bar_sh)
    else:
        # Fallback: simple pill shapes
        _draw_pill(uniform_shader, bx, by, bar_w, bar_h,
                   (*COL_BAR_BG_RGB, fade))
        fill_w = bar_w * min(progress, 1.0)
        if fill_w > bar_h:
            _draw_fill_bar(uniform_shader, bx, by, fill_w, bar_h,
                           (*COL_SUNSET, fade))


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


_fadeout_time = 0.0    # when the loading→render fade-out started
FADEOUT_DURATION = 0.8  # seconds for loading screen to dissolve into render


def begin_fadeout():
    """Call when loading is complete to start fading out the loading screen."""
    global _fadeout_time
    _fadeout_time = time.perf_counter()


def draw_fadeout(w, h):
    """Draw the loading screen fading out over the render. Returns True while fading."""
    global _fadeout_time
    if _fadeout_time <= 0.0:
        return False
    elapsed = time.perf_counter() - _fadeout_time
    if elapsed >= FADEOUT_DURATION:
        _fadeout_time = 0.0
        return False

    alpha = 1.0 - (elapsed / FADEOUT_DURATION)

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)

    # Just the dark background fading out — clean dissolve into the render
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", (COL_BG[0], COL_BG[1], COL_BG[2], alpha))
    bg = ((0, 0), (w, 0), (w, h), (0, h))
    batch_for_shader(shader, 'TRI_FAN', {"pos": bg}).draw(shader)

    gpu.state.blend_set('NONE')
    return True



def draw(w, h, status="", progress=0.0):
    """Draw the Ignis RT loading screen."""
    global _start_time

    now = time.perf_counter()
    if _start_time <= 0.0:
        _start_time = now
    fade = min((now - _start_time) / CROSSFADE_DURATION, 1.0)

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
    spinner_y = title_y - 130
    # Fire spinner (custom shader with FBM noise)
    fire_sh = None
    try:
        fire_sh = _get_fire_shader()
    except Exception:
        pass
    if fire_sh:
        fire_size = 60
        fx = cx - fire_size
        fy = spinner_y - fire_size
        fire_batch = batch_for_shader(fire_sh, 'TRI_FAN', {
            "pos": ((fx, fy), (fx + fire_size * 2, fy),
                    (fx + fire_size * 2, fy + fire_size * 2), (fx, fy + fire_size * 2)),
            "uv": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        })
        gpu.state.blend_set('ALPHA')
        fire_sh.bind()
        fire_sh.uniform_float("mvp", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
        fire_sh.uniform_float("time", now - _start_time)
        fire_sh.uniform_float("alpha", fade)
        fire_sh.uniform_float("fillCol", COL_SUNSET)
        fire_batch.draw(fire_sh)
    else:
        _draw_spinner(shader, cx, spinner_y, now - _start_time)

    # Status text
    if status:
        blf.size(font_id, 18)
        blf.color(font_id, *col_text_dim)
        sw = blf.dimensions(font_id, status)[0]
        blf.position(font_id, cx - sw / 2, spinner_y - 80, 0)
        blf.draw(font_id, status)

    # Progress bar
    if progress > 0.0:
        try:
            _draw_progress_bar(shader, cx, spinner_y, progress, fade)
        except Exception as _e:
            import traceback
            _ep = os.path.join(os.path.expanduser("~"), "ignis-bar-draw-error.txt")
            with open(_ep, "w") as _ef:
                _ef.write(f"Bar draw error: {_e}\n\n")
                traceback.print_exc(file=_ef)

    gpu.state.blend_set('NONE')


def reset():
    """Reset the loading screen timer and shader cache."""
    global _start_time, _logo_texture
    global _fire_shader, _fire_shader_failed, _bar_shader, _bar_shader_failed
    _start_time = 0.0
    _logo_texture = None
    _fire_shader = None
    _fire_shader_failed = False
    _bar_shader = None
    _bar_shader_failed = False
