"""Dependency-free mesh writers (numpy only): OBJ, ASCII FBX 7.4, GLB.

Input convention (same as ComfyUI MESH / glTF): Y-up, right-handed, counter-clockwise
front faces. Triangles only. Units are set by the caller (FBX: unit_cm; OBJ: none; GLB: meters).
"""

import io
import json
import struct

import numpy as np


def _fmt(arr, spec="%.6f"):
    return ",".join(np.char.mod(spec, np.asarray(arr).ravel()))


def write_obj(path, name, vertices, faces, normals=None):
    f1 = faces.astype(np.int64) + 1
    buf = io.StringIO()
    buf.write("# s_tools\n")
    buf.write(f"o {name}\n")
    np.savetxt(buf, vertices, fmt="v %.6f %.6f %.6f")
    if normals is not None:
        np.savetxt(buf, normals, fmt="vn %.6f %.6f %.6f")
        corners = np.stack([f1, f1], axis=-1).reshape(len(f1), -1)
        np.savetxt(buf, corners, fmt="f %d//%d %d//%d %d//%d")
    else:
        np.savetxt(buf, f1, fmt="f %d %d %d")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(buf.getvalue())


def write_fbx(path, name, vertices, faces, normals=None, unit_cm=100.0):
    """ASCII FBX 7.4: one mesh, Y-up, one grey material. unit_cm: length of 1 unit in cm (100 = meters)."""
    idx = faces.astype(np.int64).copy()
    idx[:, 2] = -idx[:, 2] - 1  # last index of each polygon is stored as ~i
    geo_id, model_id, mat_id = 1000001, 1000002, 1000003
    out = []
    w = out.append
    w("; FBX 7.4.0 project file\n; Created by s_tools\n; ----------------------------------------------------\n\n")
    w("FBXHeaderExtension:  {\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n\tCreator: \"s_tools\"\n}\n")
    w("GlobalSettings:  {\n\tVersion: 1000\n\tProperties70:  {\n")
    for key, val in (("UpAxis", 1), ("UpAxisSign", 1), ("FrontAxis", 2), ("FrontAxisSign", 1),
                     ("CoordAxis", 0), ("CoordAxisSign", 1), ("OriginalUpAxis", 1), ("OriginalUpAxisSign", 1)):
        w(f"\t\tP: \"{key}\", \"int\", \"Integer\", \"\",{val}\n")
    w(f"\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",{unit_cm:.10g}\n")
    w(f"\t\tP: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",{unit_cm:.10g}\n")
    w("\t}\n}\n")
    w("References:  {\n}\n")
    w("Definitions:  {\n\tVersion: 100\n\tCount: 4\n")
    for t in ("GlobalSettings", "Model", "Geometry", "Material"):
        w(f"\tObjectType: \"{t}\" {{\n\t\tCount: 1\n\t}}\n")
    w("}\n")
    w("Objects:  {\n")
    w(f"\tGeometry: {geo_id}, \"Geometry::{name}\", \"Mesh\" {{\n")
    w(f"\t\tVertices: *{vertices.size} {{\n\t\t\ta: {_fmt(vertices)}\n\t\t}} \n")
    w(f"\t\tPolygonVertexIndex: *{idx.size} {{\n\t\t\ta: {_fmt(idx, '%d')}\n\t\t}} \n")
    w("\t\tGeometryVersion: 124\n")
    if normals is not None:
        w("\t\tLayerElementNormal: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n"
          "\t\t\tMappingInformationType: \"ByVertice\"\n\t\t\tReferenceInformationType: \"Direct\"\n")
        w(f"\t\t\tNormals: *{normals.size} {{\n\t\t\t\ta: {_fmt(normals)}\n\t\t\t}} \n\t\t}}\n")
    w("\t\tLayerElementMaterial: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n"
      "\t\t\tMappingInformationType: \"AllSame\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n"
      "\t\t\tMaterials: *1 {\n\t\t\t\ta: 0\n\t\t\t} \n\t\t}\n")
    w("\t\tLayer: 0 {\n\t\t\tVersion: 100\n")
    if normals is not None:
        w("\t\t\tLayerElement:  {\n\t\t\t\tType: \"LayerElementNormal\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n")
    w("\t\t\tLayerElement:  {\n\t\t\t\tType: \"LayerElementMaterial\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n")
    w("\t\t}\n\t}\n")
    w(f"\tModel: {model_id}, \"Model::{name}\", \"Mesh\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n"
      "\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n\t\t}\n"
      "\t\tShading: T\n\t\tCulling: \"CullingOff\"\n\t}\n")
    w(f"\tMaterial: {mat_id}, \"Material::{name}_mat\", \"\" {{\n\t\tVersion: 102\n\t\tShadingModel: \"phong\"\n"
      "\t\tMultiLayer: 0\n\t\tProperties70:  {\n"
      "\t\t\tP: \"DiffuseColor\", \"Color\", \"\", \"A\",0.8,0.8,0.8\n\t\t}\n\t}\n")
    w("}\n")
    w(f"Connections:  {{\n\tC: \"OO\",{model_id},0\n\tC: \"OO\",{geo_id},{model_id}\n\tC: \"OO\",{mat_id},{model_id}\n}}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(out))


def write_glb(path, name, vertices, faces, normals=None):
    pos = np.ascontiguousarray(vertices, dtype=np.float32)
    ind = np.ascontiguousarray(faces, dtype=np.uint32).ravel()
    blobs, views, accessors = [], [], []
    offset = 0

    def add(data, target, comp, count, typ, minmax=None):
        nonlocal offset
        raw = data.tobytes()
        raw += b"\x00" * ((4 - len(raw) % 4) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data.tobytes()), "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": count, "type": typ}
        if minmax:
            acc["min"], acc["max"] = minmax
        accessors.append(acc)
        blobs.append(raw)
        offset += len(raw)
        return len(accessors) - 1

    attrs = {"POSITION": add(pos, 34962, 5126, len(pos), "VEC3",
                             (pos.min(axis=0).tolist(), pos.max(axis=0).tolist()))}
    if normals is not None:
        attrs["NORMAL"] = add(np.ascontiguousarray(normals, dtype=np.float32), 34962, 5126, len(pos), "VEC3")
    ind_acc = add(ind, 34963, 5125, ind.size, "SCALAR")
    gltf = {
        "asset": {"version": "2.0", "generator": "s_tools"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": ind_acc, "mode": 4}]}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": offset}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    bin_ = b"".join(blobs)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(bin_), 0x004E4942) + bin_)


WRITERS = {"fbx": write_fbx, "obj": write_obj, "glb": write_glb}
