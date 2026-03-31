"""Debug World/Environment settings for ignis-rt.

Run from Blender:
  blender --background file.blend --python debug_world.py
Or paste into Blender's scripting editor and run.

Outputs all World shader nodes, environment settings, sky configuration,
and color management settings that affect how ignis-rt renders the scene.
"""

import bpy
from io import StringIO


def _write(buf, line=""):
    buf.write(line + "\n")


def _dump_node_inputs(buf, node, indent="    "):
    for inp in node.inputs:
        linked = ""
        if inp.is_linked:
            from_node = inp.links[0].from_node
            from_socket = inp.links[0].from_socket.name
            linked = f" <- [{from_node.type}] {from_node.name}.{from_socket}"
        try:
            val = inp.default_value
            if hasattr(val, '__len__'):
                val_str = "({})".format(", ".join(f"{v:.4f}" for v in val))
            else:
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        except Exception:
            val_str = "?"
        _write(buf, f"{indent}in: {inp.name}: = {val_str}{linked}")
    for out in node.outputs:
        linked = ""
        if out.is_linked:
            targets = []
            for link in out.links:
                targets.append(f"[{link.to_node.type}] {link.to_node.name}.{link.to_socket.name}")
            linked = " -> " + ", ".join(targets)
        _write(buf, f"{indent}out: {out.name}{linked}")


def _dump_image(buf, image, indent="    "):
    if not image:
        _write(buf, f"{indent}image: None")
        return
    _write(buf, f"{indent}image: name='{image.name}' size=({image.size[0]}, {image.size[1]})"
           f" source={image.source} filepath='{image.filepath}'"
           f" colorspace='{image.colorspace_settings.name}'"
           f" is_data={image.colorspace_settings.is_data}")


def main():
    buf = StringIO()
    scene = bpy.context.scene

    _write(buf, "=" * 80)
    _write(buf, "IGNIS WORLD / ENVIRONMENT DEBUG")
    _write(buf, "=" * 80)

    # ── World Settings ──
    world = scene.world
    _write(buf)
    _write(buf, "World")
    _write(buf, "-" * 60)
    if not world:
        _write(buf, "  No World assigned to scene")
    else:
        _write(buf, f"  Name: '{world.name}'")
        _use_nodes = getattr(world, 'use_nodes', True)  # deprecated in Blender 6.0, default True
        _write(buf, f"  Use nodes: {_use_nodes}")
        _write(buf, f"  Color: ({world.color[0]:.4f}, {world.color[1]:.4f}, {world.color[2]:.4f})")

        if _use_nodes and world.node_tree:
            _write(buf)
            _write(buf, f"  Node Tree: '{world.node_tree.name}' ({len(world.node_tree.nodes)} nodes)")
            for node in world.node_tree.nodes:
                _write(buf, f"  - [{node.type}] '{node.name}' label='{node.label}' mute={node.mute}")

                # Special handling for specific node types
                if node.type == 'TEX_ENVIRONMENT':
                    _dump_image(buf, node.image)
                    if hasattr(node, 'projection'):
                        _write(buf, f"      projection: {node.projection}")
                    if hasattr(node, 'interpolation'):
                        _write(buf, f"      interpolation: {node.interpolation}")

                elif node.type == 'TEX_SKY':
                    _write(buf, f"      sky_type: {node.sky_type}")
                    for attr in ('sun_direction', 'sun_elevation', 'sun_rotation',
                                 'sun_intensity', 'sun_size', 'altitude',
                                 'air_density', 'dust_density', 'ozone_density',
                                 'sun_disc', 'ground_albedo'):
                        if hasattr(node, attr):
                            val = getattr(node, attr)
                            if hasattr(val, '__len__'):
                                val_str = "({})".format(", ".join(f"{v:.4f}" for v in val))
                            elif isinstance(val, float):
                                val_str = f"{val:.4f}"
                            else:
                                val_str = str(val)
                            _write(buf, f"      {attr}: {val_str}")

                elif node.type == 'BACKGROUND':
                    pass  # inputs shown below

                elif node.type == 'MAPPING':
                    _write(buf, f"      vector_type: {node.vector_type}")

                _dump_node_inputs(buf, node, indent="      ")
                _write(buf)

            # Show links
            _write(buf, "  Links:")
            for link in world.node_tree.links:
                _write(buf, f"    [{link.from_node.name}].{link.from_socket.name}"
                       f" -> [{link.to_node.name}].{link.to_socket.name}")
        else:
            _write(buf, "  (No node tree)")

    # ── Lights in Scene ──
    _write(buf)
    _write(buf, "Scene Lights")
    _write(buf, "-" * 60)
    light_count = 0
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            light = obj.data
            light_count += 1
            mat = obj.matrix_world
            pos = mat.translation
            _write(buf, f"  [{light.type}] '{obj.name}'")
            _write(buf, f"    energy: {light.energy:.2f} W")
            _write(buf, f"    color: ({light.color[0]:.4f}, {light.color[1]:.4f}, {light.color[2]:.4f})")
            _write(buf, f"    position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            if light.type == 'SUN':
                direction = -mat.col[2].xyz.normalized()
                _write(buf, f"    direction: ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})")
                if hasattr(light, 'angle'):
                    _write(buf, f"    angle: {light.angle:.4f} rad ({light.angle * 57.2958:.2f} deg)")
            elif light.type == 'SPOT':
                _write(buf, f"    spot_size: {light.spot_size:.4f} rad ({light.spot_size * 57.2958:.1f} deg)")
                _write(buf, f"    spot_blend: {light.spot_blend:.4f}")
                if hasattr(light, 'shadow_soft_size'):
                    _write(buf, f"    shadow_soft_size: {light.shadow_soft_size:.4f}")
            elif light.type == 'AREA':
                _write(buf, f"    shape: {light.shape}")
                _write(buf, f"    size: {light.size:.4f}")
                if light.shape in ('RECTANGLE', 'ELLIPSE'):
                    _write(buf, f"    size_y: {light.size_y:.4f}")
            elif light.type == 'POINT':
                if hasattr(light, 'shadow_soft_size'):
                    _write(buf, f"    shadow_soft_size: {light.shadow_soft_size:.4f}")
            # Check for nodes (emission color/texture)
            if light.use_nodes and light.node_tree:
                for node in light.node_tree.nodes:
                    if node.type == 'EMISSION':
                        _write(buf, f"    emission node: strength={node.inputs['Strength'].default_value:.2f}")
                    elif node.type == 'BLACKBODY':
                        _write(buf, f"    blackbody: temp={node.inputs['Temperature'].default_value:.0f}K")
            _write(buf)
    if light_count == 0:
        _write(buf, "  (No lights in scene)")

    # ── Color Management ──
    _write(buf)
    _write(buf, "Color Management")
    _write(buf, "-" * 60)
    ds = scene.display_settings
    vs = scene.view_settings
    _write(buf, f"  display_device: '{ds.display_device}'")
    _write(buf, f"  view_transform: '{vs.view_transform}'")
    _write(buf, f"  look: '{getattr(vs, 'look', '')}'")
    _write(buf, f"  exposure: {vs.exposure:.4f}")
    _write(buf, f"  gamma: {getattr(vs, 'gamma', 1.0):.4f}")
    _write(buf, f"  use_curve_mapping: {getattr(vs, 'use_curve_mapping', False)}")
    _write(buf, f"  use_white_balance: {getattr(vs, 'use_white_balance', False)}")
    if getattr(vs, 'use_white_balance', False):
        wp = getattr(vs, 'white_balance_whitepoint', (1, 1, 1))
        _write(buf, f"  white_balance_whitepoint: ({wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f})")

    # Working space
    blend_cs = getattr(bpy.data, "colorspace", None)
    if blend_cs:
        _write(buf, f"  working_space: '{getattr(blend_cs, 'working_space', '')}'")
        _write(buf, f"  working_space_interop_id: '{getattr(blend_cs, 'working_space_interop_id', '')}'")

    # ── Scene Render Settings ──
    _write(buf)
    _write(buf, "Render Settings (relevant)")
    _write(buf, "-" * 60)
    _write(buf, f"  engine: '{scene.render.engine}'")
    _write(buf, f"  resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    _write(buf, f"  resolution_percentage: {scene.render.resolution_percentage}%")
    _write(buf, f"  film_transparent: {scene.render.film_transparent}")

    # ── Ignis RT Properties ──
    _write(buf)
    _write(buf, "Ignis RT Properties")
    _write(buf, "-" * 60)
    try:
        props = scene.ignis_rt
        for prop_name in dir(props):
            if prop_name.startswith('_') or prop_name.startswith('bl_'):
                continue
            try:
                val = getattr(props, prop_name)
                if callable(val):
                    continue
                _write(buf, f"  {prop_name}: {val}")
            except Exception:
                pass
    except Exception:
        _write(buf, "  (ignis_rt properties not available)")

    # ── Output ──
    output = buf.getvalue()
    print(output)

    # Also write to file next to blend
    import os
    try:
        out_path = os.path.join(os.path.expanduser("~"), "ignis-world-debug.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n[Saved to {out_path}]")
    except Exception as e:
        print(f"\n[Could not save: {e}]")


if __name__ == "__main__":
    main()
