"""Check Mapping node vector_type for laminate_floor_02"""
import bpy
mat = bpy.data.materials.get("laminate_floor_02")
if mat and mat.node_tree:
    for node in mat.node_tree.nodes:
        if node.type == 'MAPPING':
            print(f"Mapping: vector_type = '{node.vector_type}'")
            s = node.inputs.get('Scale')
            r = node.inputs.get('Rotation')
            l = node.inputs.get('Location')
            if s: print(f"  Scale = ({s.default_value[0]:.4f}, {s.default_value[1]:.4f}, {s.default_value[2]:.4f})")
            if r: print(f"  Rotation = ({r.default_value[0]:.4f}, {r.default_value[1]:.4f}, {r.default_value[2]:.4f})")
            if l: print(f"  Location = ({l.default_value[0]:.4f}, {l.default_value[1]:.4f}, {l.default_value[2]:.4f})")
