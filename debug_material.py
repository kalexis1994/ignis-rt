"""Full analysis of ALL materials on Floor object — complete node chains."""
import bpy

obj = bpy.data.objects.get("Floor")
if not obj:
    print("Floor not found")
else:
    for slot_idx, slot in enumerate(obj.material_slots):
        mat = slot.material
        if not mat or not mat.node_tree:
            continue
        print(f"\n{'='*60}")
        print(f"Material [{slot_idx}]: {mat.name}")
        print(f"{'='*60}")

        # Find ALL Mapping nodes and their connections
        for node in mat.node_tree.nodes:
            if node.type == 'MAPPING':
                vt = getattr(node, 'vector_type', '?')
                s = node.inputs.get('Scale')
                r = node.inputs.get('Rotation')
                l = node.inputs.get('Location')
                print(f"\n  MAPPING '{node.name}': vector_type={vt}")
                if s: print(f"    Scale=({s.default_value[0]:.3f}, {s.default_value[1]:.3f}, {s.default_value[2]:.3f})")
                if r: print(f"    Rotation=({r.default_value[0]:.4f}, {r.default_value[1]:.4f}, {r.default_value[2]:.4f})")
                if l: print(f"    Location=({l.default_value[0]:.3f}, {l.default_value[1]:.3f}, {l.default_value[2]:.3f})")

                # What feeds into this Mapping?
                vec_inp = node.inputs.get('Vector')
                if vec_inp and vec_inp.is_linked:
                    src = vec_inp.links[0].from_node
                    print(f"    Input: {src.name} (type={src.type}).{vec_inp.links[0].from_socket.name}")

                # What does this Mapping feed into?
                vec_out = node.outputs.get('Vector')
                if vec_out and vec_out.is_linked:
                    for link in vec_out.links:
                        print(f"    Output → {link.to_node.name} (type={link.to_node.type}).{link.to_socket.name}")

        # Find Principled BSDF and trace Base Color chain
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                print(f"\n  PRINCIPLED BSDF '{node.name}':")
                for inp_name in ['Base Color', 'Roughness', 'Normal', 'Metallic']:
                    inp = node.inputs.get(inp_name)
                    if inp and inp.is_linked:
                        chain = []
                        current_socket = inp
                        depth = 0
                        while current_socket and current_socket.is_linked and depth < 10:
                            from_node = current_socket.links[0].from_node
                            from_sock = current_socket.links[0].from_socket.name
                            chain.append(f"{from_node.name}({from_node.type}).{from_sock}")
                            # Follow first color/vector input
                            next_socket = None
                            for fi in from_node.inputs:
                                if fi.is_linked and fi.type in ('RGBA', 'VECTOR', 'VALUE'):
                                    next_socket = fi
                                    break
                            current_socket = next_socket
                            depth += 1
                        print(f"    {inp_name}: {' → '.join(chain)}")
                    elif inp:
                        try:
                            v = inp.default_value
                            if hasattr(v, '__len__'):
                                print(f"    {inp_name}: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
                            else:
                                print(f"    {inp_name}: {v:.3f}")
                        except:
                            pass

        # Check for Mix Shader or other shader nodes
        for node in mat.node_tree.nodes:
            if node.type in ('MIX_SHADER', 'ADD_SHADER'):
                print(f"\n  {node.type} '{node.name}':")
                for inp in node.inputs:
                    if inp.is_linked:
                        print(f"    {inp.name} ← {inp.links[0].from_node.name}")

        # List ALL Image Texture nodes
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE':
                img_name = node.image.name if node.image else "None"
                vec_inp = node.inputs.get('Vector')
                vec_src = ""
                if vec_inp and vec_inp.is_linked:
                    vec_src = f" ← {vec_inp.links[0].from_node.name}.{vec_inp.links[0].from_socket.name}"
                print(f"\n  TEX_IMAGE '{node.name}': image='{img_name}'{vec_src}")
