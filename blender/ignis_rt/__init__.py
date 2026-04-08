"""Ignis RT — Vulkan ray tracing render engine for Blender.

Metadata is in blender_manifest.toml (Blender Extension System).
"""

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

    # (Sky & Lighting removed — sun/ambient/sky now come from Blender World settings)

    # -- Quality --
    max_bounces: IntProperty(
        name="Max Bounces", default=2, min=2, max=8,
        description="Path tracing bounces (2=one GI bounce, higher=better indirect lighting)",
        update=_tag_redraw,
    )
    samples_per_pixel: IntProperty(
        name="Samples/Pixel", default=1, min=1, max=10,
        description="Samples per pixel per frame (1-3=realtime, 4-6=quality, 8-10=offline)",
        update=_tag_redraw,
    )
    backface_culling: BoolProperty(
        name="Backface Culling",
        default=False,
        description="Cull back-facing triangles (skip geometry facing away from camera)",
        update=_tag_redraw,
    )
    debug_view: bpy.props.EnumProperty(
        name="View Mode",
        items=[
            ('0', "Final", "Full rendered output"),
            ('1', "Lighting", "White surfaces — shows lighting, shadows, GI only"),
            ('2', "Unlit", "Albedo textures only — no lighting"),
            ('3', "Normals", "World-space normals as RGB"),
            ('4', "Depth", "Linear depth as grayscale gradient"),
            ('5', "UV", "UV coordinates as RG color"),
            ('7', "UV Checker", "Checkerboard pattern on UV coordinates"),
            ('6', "Complexity", "Shader cost heatmap (green=cheap, red=expensive)"),
        ],
        default='0',
        description="Viewport visualization mode",
        update=_tag_redraw,
    )

    # -- DLSS --
    dlss_enabled: BoolProperty(
        name="DLSS",
        description="Enable NVIDIA DLSS upscaling (requires RTX GPU, restarts renderer)",
        default=True,
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
            ('6', "DLAA", "Native resolution — no upscaling, AI anti-aliasing only"),
        ],
        default='4',
    )
    dlss_rr_enabled: BoolProperty(
        name="Ray Reconstruction",
        description="Use DLSS Ray Reconstruction (replaces NRD denoiser, requires RTX GPU + driver 535+)",
        default=True,
    )

    # -- Performance --
    hybrid_rasterization: BoolProperty(
        name="Hybrid Rasterization",
        description="Rasterize primary visibility for faster first bounce (uncheck for pure path tracing)",
        default=True,
        update=_tag_redraw,
    )

    # -- Experimental --
    restir_di: BoolProperty(
        name="ReSTIR DI",
        description="Reservoir-based light sampling. Better for many lights (16+), may cause noise with few lights",
        default=False,
        update=_tag_redraw,
    )
    material_sort: BoolProperty(
        name="Material Sort",
        description="GPU material sorting for wavefront. Improves FPS in scenes with mixed materials (glass+hair+volume). Slight overhead for uniform scenes",
        default=False,
        update=_tag_redraw,
    )
    use_wavefront: BoolProperty(
        name="Wavefront Path Tracing",
        description="Experimental: compute-based multi-kernel path tracing for better GPU occupancy",
        default=False,
    )

    # -- Performance --
    vsync: BoolProperty(
        name="V-Sync",
        description="Limit FPS to monitor refresh rate to reduce screen tearing",
        default=False,
        update=_tag_redraw,
    )
    fps_limit: IntProperty(
        name="FPS Limit", default=0, min=0, max=240,
        description="Limit viewport FPS (0 = unlimited, overridden by V-Sync)",
        update=_tag_redraw,
    )

    # -- Color --
    # (Exposure, tonemap, contrast now come from Blender Color Management)
    saturation: FloatProperty(
        name="Saturation", default=1.0, min=0.0, max=3.0, step=10,
        update=_tag_redraw,
    )

    # -- Overlay --
    fps_overlay: EnumProperty(
        name="FPS Overlay",
        items=[
            ('OFF', "Off", "No FPS overlay"),
            ('FPS', "FPS", "FPS counter only"),
            ('MS', "FPS + ms", "FPS and frame time"),
            ('STATS', "Stats", "Min, Max, Avg, 1% low"),
            ('FULL', "Full", "Stats + frame time graph"),
        ],
        default='OFF',
        description="Performance overlay detail level",
        update=_tag_redraw,
    )
    fps_position: EnumProperty(
        name="Position",
        items=[
            ('TL', "Top Left", ""),
            ('TC', "Top Center", ""),
            ('TR', "Top Right", ""),
            ('BL', "Bottom Left", ""),
            ('BC', "Bottom Center", ""),
            ('BR', "Bottom Right", ""),
        ],
        default='TR',
        description="Overlay position in viewport",
        update=_tag_redraw,
    )
    show_gpu_profiler: BoolProperty(
        name="GPU Profiler",
        description="Display GPU timing breakdown (RT, PostProcess, Total)",
        default=False, update=_tag_redraw,
    )


class IGNIS_OT_set_fps_position(bpy.types.Operator):
    bl_idname = "ignis.set_fps_position"
    bl_label = "Set Overlay Position"
    bl_options = {'INTERNAL'}

    position: StringProperty()

    def execute(self, context):
        context.scene.ignis_rt.fps_position = self.position
        return {'FINISHED'}


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


class IGNIS_OT_reload_shaders(bpy.types.Operator):
    bl_idname = "ignis.reload_shaders"
    bl_label = "Reload Shaders"
    bl_description = "Hot-reload shaders from disk (recompile + recreate pipeline, keep geometry)"

    def execute(self, context):
        from . import dll_wrapper
        try:
            ok = dll_wrapper.reload_shaders()
            self.report({'INFO'} if ok else {'ERROR'},
                        "Shaders reloaded" if ok else "Shader reload failed — check log")
        except Exception as e:
            self.report({'ERROR'}, f"Shader reload error: {e}")
        # Tag viewport for redraw
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


class IGNIS_PT_dlss(bpy.types.Panel):
    bl_label = "DLSS"
    bl_idname = "IGNIS_PT_dlss"
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
        layout.prop(props, "dlss_quality")


class IGNIS_PT_color(bpy.types.Panel):
    bl_label = "Color"
    bl_idname = "IGNIS_PT_color"
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
        layout.prop(props, "saturation")


class IGNIS_PT_performance(bpy.types.Panel):
    bl_label = "Performance"
    bl_idname = "IGNIS_PT_performance"
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
        layout.prop(props, "vsync")
        layout.prop(props, "fps_limit")
        layout.prop(props, "fps_overlay")
        layout.prop(props, "show_gpu_profiler")

        # Position grid (3x2) — visual selector
        if props.fps_overlay != 'OFF' or props.show_gpu_profiler:
            layout.label(text="Position:")
            positions = [
                ('TL', 'TC', 'TR'),
                ('BL', 'BC', 'BR'),
            ]
            labels = {
                'TL': '', 'TC': '', 'TR': '',
                'BL': '', 'BC': '', 'BR': '',
            }
            current = props.fps_position
            for row_positions in positions:
                row = layout.row(align=True)
                for pos in row_positions:
                    op = row.operator(
                        "ignis.set_fps_position",
                        text=labels[pos],
                        depress=(current == pos),
                    )
                    op.position = pos


class IGNIS_PT_advanced(bpy.types.Panel):
    bl_label = "Advanced"
    bl_idname = "IGNIS_PT_advanced"
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
        layout.prop(props, "hybrid_rasterization")
        layout.prop(props, "restir_di")
        layout.prop(props, "material_sort")
        # layout.prop(props, "use_wavefront")  # TODO: not ready yet
        layout.separator()
        layout.prop(props, "debug_view")
        layout.separator()
        layout.operator("ignis.reload_scene", icon='FILE_REFRESH')
        layout.operator("ignis.reload_shaders", icon='SHADING_RENDERED')



def _get_compatible_panels():
    """Find Blender panels that Ignis RT should inherit.

    Only inherit panels for editing scene data (materials, lights, camera,
    world, objects, etc.). The Render Properties tab is handled entirely by
    our own IGNIS_PT_* panels.
    """
    # Contexts where we want Blender's default panels to appear
    include_contexts = {
        'data', 'material', 'world', 'object', 'scene',
        'constraint', 'modifier', 'particle', 'physics',
        'bone', 'bone_constraint', 'output',
    }
    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if not hasattr(panel, 'COMPAT_ENGINES'):
            continue
        has_render = 'BLENDER_RENDER' in panel.COMPAT_ENGINES
        has_eevee = ('BLENDER_EEVEE' in panel.COMPAT_ENGINES
                     or 'BLENDER_EEVEE_NEXT' in panel.COMPAT_ENGINES)
        if not has_render and not has_eevee:
            continue
        ctx = getattr(panel, 'bl_context', '')
        if ctx in include_contexts:
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
    bpy.utils.register_class(IGNIS_OT_set_fps_position)
    bpy.utils.register_class(IGNIS_OT_reload_scene)
    bpy.utils.register_class(IGNIS_OT_reload_shaders)
    bpy.utils.register_class(IGNIS_PT_sampling)
    bpy.utils.register_class(IGNIS_PT_dlss)
    bpy.utils.register_class(IGNIS_PT_color)
    bpy.utils.register_class(IGNIS_PT_performance)
    bpy.utils.register_class(IGNIS_PT_advanced)
    bpy.utils.register_class(engine.IgnisRenderEngine)
    bpy.types.Scene.ignis_rt = bpy.props.PointerProperty(type=IgnisRTSceneProperties)

    # Add IGNIS_RT to generic panels (materials, lights, world, etc.)
    for panel in _get_compatible_panels():
        panel.COMPAT_ENGINES.add('IGNIS_RT')

    # Explicitly add IGNIS_RT to Color Management panels (render context)
    # so users can control exposure, view transform, and look from Blender's UI
    for panel in bpy.types.Panel.__subclasses__():
        if panel.__name__.startswith('RENDER_PT_color_management'):
            compat = getattr(panel, 'COMPAT_ENGINES', None)
            if compat is not None:
                compat.add('IGNIS_RT')

    # Undo/redo handlers: force full transform sync after undo/redo
    # Guard against duplicate registration (re-enable without restart)
    if engine._on_undo_redo not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(engine._on_undo_redo)
    if engine._on_undo_redo not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(engine._on_undo_redo)
    # Also hook load_post to catch file loads
    if engine._on_undo_redo not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(engine._on_undo_redo)
    # Depsgraph handler: catches modifier/deformation changes that view_update misses
    if engine._on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(engine._on_depsgraph_update)

    # Clean up: remove IGNIS_RT from render-context panels that aren't ours,
    # EXCEPT Blender's Color Management panel (we read exposure/tonemap from it)
    _own_panels = {
        'IGNIS_PT_sampling', 'IGNIS_PT_dlss', 'IGNIS_PT_color',
        'IGNIS_PT_performance', 'IGNIS_PT_advanced',
    }
    for panel in bpy.types.Panel.__subclasses__():
        if getattr(panel, 'bl_context', '') == 'render' and panel.__name__ not in _own_panels:
            if panel.__name__.startswith('RENDER_PT_color_management'):
                continue  # keep Blender's full Color Management stack visible
            compat = getattr(panel, 'COMPAT_ENGINES', None)
            if compat and 'IGNIS_RT' in compat:
                compat.discard('IGNIS_RT')



def unregister():
    global _icon_previews
    # Remove undo/redo/load handlers
    for handler_list in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post, bpy.app.handlers.load_post):
        if engine._on_undo_redo in handler_list:
            handler_list.remove(engine._on_undo_redo)
    if engine._on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(engine._on_depsgraph_update)
    try:
        engine._ignis_shutdown()
    except Exception:
        pass
    for panel in _get_compatible_panels():
        panel.COMPAT_ENGINES.discard('IGNIS_RT')
    bpy.utils.unregister_class(engine.IgnisRenderEngine)
    bpy.utils.unregister_class(IGNIS_PT_advanced)
    bpy.utils.unregister_class(IGNIS_PT_performance)
    bpy.utils.unregister_class(IGNIS_PT_color)
    bpy.utils.unregister_class(IGNIS_PT_dlss)
    bpy.utils.unregister_class(IGNIS_PT_sampling)
    bpy.utils.unregister_class(IGNIS_OT_reload_shaders)
    bpy.utils.unregister_class(IGNIS_OT_reload_scene)
    bpy.utils.unregister_class(IGNIS_OT_set_fps_position)
    del bpy.types.Scene.ignis_rt
    bpy.utils.unregister_class(IgnisRTSceneProperties)
    bpy.utils.unregister_class(IgnisRTPreferences)
    if _icon_previews:
        bpy.utils.previews.remove(_icon_previews)
        _icon_previews = None
