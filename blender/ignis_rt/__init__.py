"""Ignis RT — Vulkan ray tracing render engine for Blender."""

bl_info = {
    "name": "Ignis RT",
    "author": "Alexis",
    "version": (0, 0, 1),
    "blender": (4, 0, 0),
    "category": "Render",
    "description": "Vulkan ray tracing render engine (viewport preview)",
}

import os
import bpy
import bpy.utils.previews
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty,
)
from . import engine

# Custom icon collection
_icon_previews = None


def get_icon(name="ignis"):
    """Get icon_id for a custom icon. Returns 0 if not loaded."""
    if _icon_previews and name in _icon_previews:
        return _icon_previews[name].icon_id
    return 0


class IgnisRTPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__  # "ignis_rt"

    log_directory: StringProperty(
        name="Log Directory",
        description="Directory for ignis-rt.log output (leave empty for user home)",
        subtype='DIR_PATH',
        default="",
    )

    ignis_root: StringProperty(
        name="Ignis RT Root",
        description="Path to ignis-rt repo root (for shaders). Leave empty to auto-detect from DLL location",
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "log_directory")
        layout.prop(self, "ignis_root")
        # Show resolved paths
        log_path = get_log_path()
        base_path = get_base_path()
        layout.label(text=f"Log file: {log_path}")
        layout.label(text=f"Shader root: {base_path}")


def get_log_path():
    """Resolve the log file path from preferences."""
    prefs = bpy.context.preferences.addons.get(__package__)
    log_dir = ""
    if prefs:
        log_dir = prefs.preferences.log_directory
    if not log_dir:
        log_dir = os.path.expanduser("~")
    return os.path.join(bpy.path.abspath(log_dir), "ignis-rt.log")


def get_base_path():
    """Resolve the ignis-rt root path for shader loading."""
    prefs = bpy.context.preferences.addons.get(__package__)
    root = ""
    if prefs:
        try:
            root = prefs.preferences.ignis_root
        except Exception:
            pass
    if root:
        return bpy.path.abspath(root)

    addon_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Check _deploy_root.txt written by deploy_blender.ps1
    breadcrumb = os.path.join(addon_dir, "_deploy_root.txt")
    if os.path.isfile(breadcrumb):
        with open(breadcrumb, "r") as f:
            candidate = f.read().strip()
        if candidate and os.path.isdir(os.path.join(candidate, "shaders")):
            return candidate

    # 2. Standalone addon: shaders/ is inside addon directory (zip install)
    if os.path.isdir(os.path.join(addon_dir, "shaders")):
        return addon_dir

    # 3. Heuristic: addon is in <repo>/blender/ignis_rt/ -> repo root = ../../
    candidate = os.path.normpath(os.path.join(addon_dir, "..", ".."))
    if os.path.isdir(os.path.join(candidate, "shaders")):
        return candidate

    # 4. Fallback: use DLL directory (lib/ is sibling to shaders/)
    lib_dir = os.path.join(addon_dir, "lib")
    if os.path.isdir(lib_dir):
        return addon_dir

    # 5. Fallback: empty (CWD)
    return ""


def _tag_redraw(self, context):
    """Tag all 3D viewports for redraw when a property changes."""
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


class IgnisRTSceneProperties(bpy.types.PropertyGroup):
    """Scene-level settings shown in Render properties."""

    # -- Sky / Lighting --
    auto_sky_colors: BoolProperty(
        name="Auto Sky Colors",
        description="Compute sun/ambient colors automatically from sun elevation",
        default=False, update=_tag_redraw,
    )
    sun_intensity: FloatProperty(
        name="Sun Intensity", default=1.29, min=0.0, max=20.0, step=10,
        update=_tag_redraw,
    )
    sun_color: FloatVectorProperty(
        name="Sun Color", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(1.0, 0.95, 0.9), update=_tag_redraw,
    )
    ambient_intensity: FloatProperty(
        name="Ambient Intensity", default=0.5, min=0.0, max=5.0, step=10,
        update=_tag_redraw,
    )
    ambient_color: FloatVectorProperty(
        name="Ambient Color", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.5, 0.6, 0.8), update=_tag_redraw,
    )
    sky_refl_intensity: FloatProperty(
        name="Sky Reflection", default=0.5, min=0.0, max=5.0, step=10,
        description="Intensity of sky reflections on surfaces",
        update=_tag_redraw,
    )
    sky_bounce_intensity: FloatProperty(
        name="Sky Bounce", default=0.25, min=0.0, max=2.0, step=10,
        description="Intensity of sky indirect lighting on bounce hits",
        update=_tag_redraw,
    )
    cloud_visibility: FloatProperty(
        name="Visibility (km)", default=50.0, min=0.1, max=200.0, step=100,
        description="Atmospheric visibility / fog distance (Koschmieder)",
        update=_tag_redraw,
    )

    # -- Quality --
    max_bounces: IntProperty(
        name="Max Bounces", default=2, min=1, max=8,
        description="Path tracing bounces (1=direct only, higher=better GI, slower)",
        update=_tag_redraw,
    )
    samples_per_pixel: IntProperty(
        name="Samples/Pixel", default=1, min=1, max=128,
        description="Samples per pixel per frame (1-4=realtime, 8-16=quality, 32+=offline)",
        update=_tag_redraw,
    )
    backface_culling: BoolProperty(
        name="Backface Culling",
        default=False,
        description="Cull back-facing triangles (skip geometry facing away from camera)",
        update=_tag_redraw,
    )
    debug_view: IntProperty(
        name="Debug View",
        default=0, min=0, max=99,
        description="Debug visualization (0=off, 99=material index)",
        update=_tag_redraw,
    )

    # -- DLSS --
    dlss_enabled: BoolProperty(
        name="DLSS",
        description="Enable NVIDIA DLSS upscaling (requires RTX GPU, restarts renderer)",
        default=False,
    )
    dlss_quality: EnumProperty(
        name="DLSS Quality",
        description="Upscaling quality preset (higher = sharper but slower)",
        items=[
            ('1', "Ultra Performance", "3.0x upscaling — maximum FPS"),
            ('2', "Performance", "2.0x upscaling — high FPS"),
            ('3', "Balanced", "1.7x upscaling — balanced quality/performance"),
            ('4', "Quality", "1.5x upscaling — high quality (recommended)"),
            ('5', "Ultra Quality", "1.3x upscaling — may not be available on all GPUs"),
            ('6', "DLAA", "Native resolution — anti-aliasing only, no upscaling"),
        ],
        default='4',
    )
    dlss_rr_enabled: BoolProperty(
        name="Ray Reconstruction",
        description="Use DLSS Ray Reconstruction (replaces NRD denoiser, requires RTX GPU + driver 535+)",
        default=False,
    )

    # -- Experimental --
    use_wavefront: BoolProperty(
        name="Wavefront Path Tracing",
        description="Experimental: compute-based multi-kernel path tracing for better GPU occupancy",
        default=False,
    )

    # -- Performance --
    fps_limit: IntProperty(
        name="FPS Limit", default=0, min=0, max=240,
        description="Limit viewport FPS (0 = unlimited)",
        update=_tag_redraw,
    )

    # -- Tonemap --
    exposure: FloatProperty(
        name="Exposure", default=1.0, min=0.01, max=20.0, step=10,
        update=_tag_redraw,
    )
    tonemap_mode: IntProperty(
        name="Tonemap", default=0, min=0, max=4,
        description="0=AgX, 1=ACES, 2=Reinhard, 3=Hable, 4=Neutral",
        update=_tag_redraw,
    )
    saturation: FloatProperty(
        name="Saturation", default=1.0, min=0.0, max=3.0, step=10,
        update=_tag_redraw,
    )
    contrast: FloatProperty(
        name="Contrast", default=1.0, min=0.0, max=3.0, step=10,
        update=_tag_redraw,
    )

    # -- Overlay --
    show_fps: BoolProperty(
        name="Show FPS",
        description="Display frames per second in the viewport",
        default=False, update=_tag_redraw,
    )


class IGNIS_OT_reload_scene(bpy.types.Operator):
    bl_idname = "ignis.reload_scene"
    bl_label = "Reload Scene"
    bl_description = "Force full scene reload (geometry + materials + textures)"

    def execute(self, context):
        from . import engine
        engine._ignis_full_dirty = True
        # Tag viewport for redraw so it picks up the flag
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


class IGNIS_PT_sampling(bpy.types.Panel):
    bl_label = "Sampling"
    bl_idname = "IGNIS_PT_sampling"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'IGNIS_RT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        props = context.scene.ignis_rt
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(props, "max_bounces")
        layout.prop(props, "samples_per_pixel")
        layout.prop(props, "backface_culling")
        layout.separator()
        layout.prop(props, "dlss_enabled")
        col = layout.column()
        col.active = props.dlss_enabled
        col.prop(props, "dlss_quality")
        col.prop(props, "dlss_rr_enabled")
        layout.separator()
        layout.prop(props, "use_wavefront")
        layout.separator()
        layout.prop(props, "fps_limit")
        layout.prop(props, "show_fps")
        layout.separator()
        layout.operator("ignis.reload_scene", icon='FILE_REFRESH')
        layout.separator()
        layout.prop(props, "debug_view")


class IGNIS_PT_status(bpy.types.Panel):
    bl_label = "Status"
    bl_idname = "IGNIS_PT_status"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'IGNIS_RT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        from . import dll_wrapper

        if not dll_wrapper.load():
            layout.label(text="DLL not loaded", icon='ERROR')
            return

        dlss = dll_wrapper.get_int("dlss_active")
        rr = dll_wrapper.get_int("dlss_rr_active")
        nrd = dll_wrapper.get_int("nrd_active")

        col = layout.column(align=True)
        if dlss:
            # Show actual quality mode (may differ from selected if GPU doesn't support it)
            actual_q = dll_wrapper.get_int("dlss_quality_actual")
            quality_names = {1: "Ultra Performance", 2: "Performance", 3: "Balanced",
                             4: "Quality", 5: "Ultra Quality", 6: "DLAA"}
            q_name = quality_names.get(actual_q, f"Mode {actual_q}")
            props = context.scene.ignis_rt
            requested_q = int(props.dlss_quality)
            if actual_q != requested_q and actual_q > 0:
                col.label(text=f"DLSS: {q_name} (fallback)", icon='INFO')
            else:
                col.label(text=f"DLSS: {q_name}", icon='CHECKMARK')
        else:
            col.label(text="DLSS: Off", icon='X')

        if rr:
            col.label(text="Denoiser: Ray Reconstruction", icon='CHECKMARK')
        elif nrd:
            col.label(text="Denoiser: NRD (RELAX + SIGMA)", icon='CHECKMARK')
        else:
            col.label(text="Denoiser: None", icon='X')


class IGNIS_PT_sky(bpy.types.Panel):
    bl_label = "Sky & Lighting"
    bl_idname = "IGNIS_PT_sky"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'IGNIS_RT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        props = context.scene.ignis_rt

        layout.use_property_split = True
        layout.use_property_decorate = False

        # Sky
        layout.prop(props, "auto_sky_colors")

        col = layout.column()
        col.active = not props.auto_sky_colors
        col.prop(props, "sun_color")
        col.prop(props, "sun_intensity")
        col.prop(props, "ambient_color")
        col.prop(props, "ambient_intensity")

        layout.separator()
        layout.prop(props, "sky_refl_intensity")
        layout.prop(props, "sky_bounce_intensity")
        layout.prop(props, "cloud_visibility")


class IGNIS_PT_tonemap(bpy.types.Panel):
    bl_label = "Tonemap"
    bl_idname = "IGNIS_PT_tonemap"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'IGNIS_RT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        props = context.scene.ignis_rt

        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(props, "exposure")
        layout.prop(props, "tonemap_mode")
        layout.prop(props, "saturation")
        layout.prop(props, "contrast")


def _get_compatible_panels():
    """Find Blender panels to make visible when Ignis RT is the active engine."""
    exclude = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }
    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and panel.__name__ not in exclude:
            if ('BLENDER_EEVEE' in panel.COMPAT_ENGINES
                    or 'BLENDER_EEVEE_NEXT' in panel.COMPAT_ENGINES):
                panels.append(panel)
    return panels


def register():
    global _icon_previews
    _icon_previews = bpy.utils.previews.new()
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    icon_file = os.path.join(icons_dir, "ignis_32.png")
    if os.path.isfile(icon_file):
        _icon_previews.load("ignis", icon_file, 'IMAGE')

    bpy.utils.register_class(IgnisRTPreferences)
    bpy.utils.register_class(IgnisRTSceneProperties)
    bpy.utils.register_class(IGNIS_OT_reload_scene)
    bpy.utils.register_class(IGNIS_PT_sampling)
    bpy.utils.register_class(IGNIS_PT_status)
    bpy.utils.register_class(IGNIS_PT_sky)
    bpy.utils.register_class(IGNIS_PT_tonemap)
    bpy.utils.register_class(engine.IgnisRenderEngine)
    bpy.types.Scene.ignis_rt = bpy.props.PointerProperty(type=IgnisRTSceneProperties)
    for panel in _get_compatible_panels():
        panel.COMPAT_ENGINES.add('IGNIS_RT')



def unregister():
    global _icon_previews
    try:
        engine._ignis_shutdown()
    except Exception:
        pass
    for panel in _get_compatible_panels():
        panel.COMPAT_ENGINES.discard('IGNIS_RT')
    bpy.utils.unregister_class(engine.IgnisRenderEngine)
    bpy.utils.unregister_class(IGNIS_PT_tonemap)
    bpy.utils.unregister_class(IGNIS_PT_sky)
    bpy.utils.unregister_class(IGNIS_PT_status)
    bpy.utils.unregister_class(IGNIS_PT_sampling)
    bpy.utils.unregister_class(IGNIS_OT_reload_scene)
    del bpy.types.Scene.ignis_rt
    bpy.utils.unregister_class(IgnisRTSceneProperties)
    bpy.utils.unregister_class(IgnisRTPreferences)
    if _icon_previews:
        bpy.utils.previews.remove(_icon_previews)
        _icon_previews = None
