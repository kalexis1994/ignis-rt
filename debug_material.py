"""Inspect Fabric.002 material."""
import bpy
mat = bpy.data.materials.get("Fabric Sofa")
if mat and mat.node_tree:
    print(f"\n=== Material: {mat.name} ===")
    for node in mat.node_tree.nodes:
        print(f"\n  Node: {node.name} (type={node.type})")
        for inp in node.inputs:
            linked = "LINKED" if inp.is_linked else "unlinked"
            val = ""
            try:
                v = inp.default_value
                if hasattr(v, '__len__'):
                    val = f" = ({', '.join(f'{x:.3f}' for x in v)})"
                else:
                    val = f" = {v:.4f}"
            except:
                pass
            from_info = ""
            if inp.is_linked:
                from_info = f" <- {inp.links[0].from_node.name}.{inp.links[0].from_socket.name}"
            print(f"    IN  {inp.name}: {linked}{val}{from_info}")
