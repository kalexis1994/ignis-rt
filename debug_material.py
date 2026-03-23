"""Trace the FULL UV chain for laminate_floor_02 Base Color."""
import bpy

mat = bpy.data.materials.get("laminate_floor_02")
if not mat or not mat.node_tree:
    print("Material not found")
else:
    # Find Principled BSDF
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bc = node.inputs.get('Base Color')
            if bc and bc.is_linked:
                tex_node = bc.links[0].from_node
                print(f"Base Color <- {tex_node.name} (type={tex_node.type})")

                # Follow Vector input chain
                current = tex_node
                depth = 0
                while current and depth < 10:
                    vec_inp = current.inputs.get('Vector') or current.inputs.get('Input')
                    if vec_inp is None:
                        # Try first input
                        for inp in current.inputs:
                            if inp.is_linked:
                                vec_inp = inp
                                break

                    if vec_inp and vec_inp.is_linked:
                        prev = vec_inp.links[0].from_node
                        from_socket = vec_inp.links[0].from_socket.name
                        print(f"  {'  ' * depth}Vector <- {prev.name} (type={prev.type}).{from_socket}")

                        # Print node properties
                        if prev.type == 'MAPPING':
                            vt = getattr(prev, 'vector_type', '?')
                            s = prev.inputs.get('Scale')
                            r = prev.inputs.get('Rotation')
                            l = prev.inputs.get('Location')
                            print(f"  {'  ' * depth}  vector_type={vt}")
                            if s: print(f"  {'  ' * depth}  Scale=({s.default_value[0]:.3f}, {s.default_value[1]:.3f}, {s.default_value[2]:.3f})")
                            if r: print(f"  {'  ' * depth}  Rotation=({r.default_value[0]:.4f}, {r.default_value[1]:.4f}, {r.default_value[2]:.4f})")
                            if l: print(f"  {'  ' * depth}  Location=({l.default_value[0]:.3f}, {l.default_value[1]:.3f}, {l.default_value[2]:.3f})")
                        elif prev.type == 'TEX_COORD':
                            print(f"  {'  ' * depth}  output: {from_socket}")

                        current = prev
                    else:
                        break
                    depth += 1

    # Also check: what does the VM compile?
    print("\n--- Now checking what VM compiles ---")
    # Import and run the compiler
    import sys, os
    addon_dir = os.path.dirname(os.path.abspath(bpy.data.filepath or __file__))

    from ignis_rt import scene_export

    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            # Check _has_nontrivial_mapping
            bc = node.inputs.get('Base Color')
            if bc and bc.is_linked:
                from_node = bc.links[0].from_node
                print(f"\nBase Color from_node: {from_node.name} (type={from_node.type})")
                if from_node.type == 'TEX_IMAGE':
                    vec_inp = from_node.inputs.get('Vector')
                    if vec_inp and vec_inp.is_linked:
                        n = vec_inp.links[0].from_node
                        while n.type == 'REROUTE' and n.inputs[0].is_linked:
                            print(f"  Following REROUTE -> {n.inputs[0].links[0].from_node.name}")
                            n = n.inputs[0].links[0].from_node
                        print(f"  Final node: {n.name} (type={n.type})")
                        if n.type == 'MAPPING':
                            vt = getattr(n, 'vector_type', '?')
                            print(f"  vector_type = {vt}")
