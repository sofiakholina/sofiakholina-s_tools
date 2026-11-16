"""Dependency-free writers for textured meshes (numpy + PIL): OBJ + MTL, ASCII FBX 7.4, GLB.

Input convention: Y-up, right-handed, CCW front faces, triangles only. Units are set by the caller
(FBX: unit_cm; OBJ: none; GLB: meters).
uvs: per-vertex (N, 2), OBJ/FBX convention (V up, origin bottom-left).
texture: uint8 (H, W, 3). OBJ and FBX reference a PNG next to the model; GLB embeds it.
Positions and UVs are welded separately, so UV seams do not split the geometry.
"""

import io
import json
import os
import struct

import numpy as np
from PIL import Image


def _weld(a):
    uniq, inv = np.unique(np.ascontiguousarray(a), axis=0, return_inverse=True)
    return uniq, np.asarray(inv).reshape(-1)


def _fmt(arr, spec="%.6f"):
    return ",".join(np.char.mod(spec, np.asarray(arr).ravel()))


def save_png(path, texture):
    Image.fromarray(texture).save(path, compress_level=4)


def write_obj(path, name, vertices, faces, normals, uvs, png_name):
    vu, vi = _weld(vertices.astype(np.float32))
    tu, ti = _weld(uvs.astype(np.float32))
    corners = [vi[faces] + 1, ti[faces] + 1]
    fmt = "f %d/%d %d/%d %d/%d"
    nu = None
    if normals is not None:
        nu, ni = _weld(normals.astype(np.float32))
        corners.append(ni[faces] + 1)
        fmt = "f %d/%d/%d %d/%d/%d %d/%d/%d"
    fc = np.stack(corners, axis=-1).reshape(len(faces), -1)
    mtl = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    buf = io.StringIO()
    buf.write(f"# s_tools\nmtllib {mtl}\no {name}\n")
    np.savetxt(buf, vu, fmt="v %.6f %.6f %.6f")
    np.savetxt(buf, tu, fmt="vt %.6f %.6f")
    if nu is not None:
        np.savetxt(buf, nu, fmt="vn %.6f %.6f %.6f")
    buf.write(f"usemtl {name}_mat\ns 1\n")
    np.savetxt(buf, fc, fmt=fmt)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(buf.getvalue())
    with open(os.path.join(os.path.dirname(path), mtl), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"newmtl {name}_mat\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nNs 0\nd 1\nillum 1\nmap_Kd {png_name}\n")


def write_fbx(path, name, vertices, faces, normals, uvs, png_name, unit_cm=100.0):
    """ASCII FBX 7.4: one mesh, Y-up, one material with a diffuse texture. unit_cm: 1 unit in cm (100 = meters)."""
    vu, vi = _weld(vertices.astype(np.float32))
    tu, ti = _weld(uvs.astype(np.float32))
    idx = vi[faces].astype(np.int64)
    idx[:, 2] = -idx[:, 2] - 1  # last index of each polygon is stored as ~i
    uv_idx = ti[faces].reshape(-1)
    geo_id, model_id, mat_id, tex_id, vid_id = 1000001, 1000002, 1000003, 1000004, 1000005
    png_abs = os.path.join(os.path.dirname(os.path.abspath(path)), png_name).replace("\\", "/")
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
    w("Definitions:  {\n\tVersion: 100\n\tCount: 6\n")
    for t in ("GlobalSettings", "Model", "Geometry", "Material", "Texture", "Video"):
        w(f"\tObjectType: \"{t}\" {{\n\t\tCount: 1\n\t}}\n")
    w("}\n")
    w("Objects:  {\n")
    w(f"\tGeometry: {geo_id}, \"Geometry::{name}\", \"Mesh\" {{\n")
    w(f"\t\tVertices: *{vu.size} {{\n\t\t\ta: {_fmt(vu)}\n\t\t}} \n")
    w(f"\t\tPolygonVertexIndex: *{idx.size} {{\n\t\t\ta: {_fmt(idx, '%d')}\n\t\t}} \n")
    w("\t\tGeometryVersion: 124\n")
    if normals is not None:
        pn = normals.astype(np.float32)[faces].reshape(-1, 3)
        w("\t\tLayerElementNormal: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n"
          "\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"Direct\"\n")
        w(f"\t\t\tNormals: *{pn.size} {{\n\t\t\t\ta: {_fmt(pn)}\n\t\t\t}} \n\t\t}}\n")
    w("\t\tLayerElementUV: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"UVChannel_1\"\n"
      "\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n")
    w(f"\t\t\tUV: *{tu.size} {{\n\t\t\t\ta: {_fmt(tu)}\n\t\t\t}} \n")
    w(f"\t\t\tUVIndex: *{uv_idx.size} {{\n\t\t\t\ta: {_fmt(uv_idx, '%d')}\n\t\t\t}} \n\t\t}}\n")
    w("\t\tLayerElementMaterial: 0 {\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n"
      "\t\t\tMappingInformationType: \"AllSame\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n"
      "\t\t\tMaterials: *1 {\n\t\t\t\ta: 0\n\t\t\t} \n\t\t}\n")
    w("\t\tLayer: 0 {\n\t\t\tVersion: 100\n")
    if normals is not None:
        w("\t\t\tLayerElement:  {\n\t\t\t\tType: \"LayerElementNormal\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n")
    w("\t\t\tLayerElement:  {\n\t\t\t\tType: \"LayerElementMaterial\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n")
    w("\t\t\tLayerElement:  {\n\t\t\t\tType: \"LayerElementUV\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n")
    w("\t\t}\n\t}\n")
    w(f"\tModel: {model_id}, \"Model::{name}\", \"Mesh\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n"
      "\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n\t\t}\n"
      "\t\tShading: T\n\t\tCulling: \"CullingOff\"\n\t}\n")
    w(f"\tMaterial: {mat_id}, \"Material::{name}_mat\", \"\" {{\n\t\tVersion: 102\n\t\tShadingModel: \"phong\"\n"
      "\t\tMultiLayer: 0\n\t\tProperties70:  {\n"
      "\t\t\tP: \"DiffuseColor\", \"Color\", \"\", \"A\",1,1,1\n\t\t}\n\t}\n")
    w(f"\tVideo: {vid_id}, \"Video::{name}_tex\", \"Clip\" {{\n\t\tType: \"Clip\"\n\t\tProperties70:  {{\n"
      f"\t\t\tP: \"Path\", \"KString\", \"XRefUrl\", \"\", \"{png_abs}\"\n\t\t}}\n"
      f"\t\tFileName: \"{png_abs}\"\n\t\tRelativeFilename: \"{png_name}\"\n\t}}\n")
    w(f"\tTexture: {tex_id}, \"Texture::{name}_tex\", \"\" {{\n\t\tType: \"TextureVideoClip\"\n\t\tVersion: 202\n"
      f"\t\tTextureName: \"Texture::{name}_tex\"\n\t\tMedia: \"Video::{name}_tex\"\n"
      f"\t\tFileName: \"{png_abs}\"\n\t\tRelativeFilename: \"{png_name}\"\n"
      "\t\tModelUVTranslation: 0,0\n\t\tModelUVScaling: 1,1\n\t\tTexture_Alpha_Source: \"None\"\n"
      "\t\tCropping: 0,0,0,0\n\t}\n")
    w("}\n")
    w(f"Connections:  {{\n\tC: \"OO\",{model_id},0\n\tC: \"OO\",{geo_id},{model_id}\n\tC: \"OO\",{mat_id},{model_id}\n"
      f"\tC: \"OP\",{tex_id},{mat_id}, \"DiffuseColor\"\n\tC: \"OO\",{vid_id},{tex_id}\n}}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(out))


def write_glb(path, name, vertices, faces, normals, uvs, texture):
    pos = np.ascontiguousarray(vertices, dtype=np.float32)
    ind = np.ascontiguousarray(faces, dtype=np.uint32).ravel()
    uv = np.ascontiguousarray(uvs, dtype=np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # glTF: V down
    png = io.BytesIO()
    Image.fromarray(texture).save(png, format="PNG", compress_level=4)
    png = png.getvalue()
    blobs, views, accessors = [], [], []
    offset = 0

    def add_view(raw, target=None):
        nonlocal offset
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target:
            view["target"] = target
        views.append(view)
        blobs.append(raw + b"\x00" * ((4 - len(raw) % 4) % 4))
        offset += len(blobs[-1])
        return len(views) - 1

    def add(data, target, comp, count, typ, minmax=None):
        acc = {"bufferView": add_view(data.tobytes(), target), "componentType": comp, "count": count, "type": typ}
        if minmax:
            acc["min"], acc["max"] = minmax
        accessors.append(acc)
        return len(accessors) - 1

    attrs = {"POSITION": add(pos, 34962, 5126, len(pos), "VEC3", (pos.min(axis=0).tolist(), pos.max(axis=0).tolist()))}
    if normals is not None:
        attrs["NORMAL"] = add(np.ascontiguousarray(normals, dtype=np.float32), 34962, 5126, len(pos), "VEC3")
    attrs["TEXCOORD_0"] = add(uv, 34962, 5126, len(uv), "VEC2")
    ind_acc = add(ind, 34963, 5125, ind.size, "SCALAR")
    img_view = add_view(png)
    gltf = {
        "asset": {"version": "2.0", "generator": "s_tools"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": ind_acc, "mode": 4, "material": 0}]}],
        "materials": [{"name": f"{name}_mat", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.8}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [{"bufferView": img_view, "mimeType": "image/png"}],
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
