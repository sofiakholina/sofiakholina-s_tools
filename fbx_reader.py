"""Dependency-free FBX mesh reader (numpy + zlib): binary 7.x and ASCII.

Reads every mesh with its UVs (first UV set) and normals, applies the full FBX node transform
(translation, pivots, pre/post rotation, rotation order, scaling, geometric transform, parents),
converts the file's axis system to Y-up and its units to meters.
Materials, textures, animation and skinning are ignored.
"""

import re
import struct
import zlib

import numpy as np


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name, self.props, self.children = name, props, children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def all(self, name):
        return [c for c in self.children if c.name == name]


# ----------------------------------------------------------------------------- binary
_ARRAY = {"f": ("<f4", 4), "d": ("<f8", 8), "l": ("<i8", 8), "i": ("<i4", 4), "b": ("<u1", 1)}


def _read_binary(data):
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    head = struct.Struct("<QQQ" if wide else "<III")
    null_len = head.size + 1

    def read_props(pos, count):
        props = []
        for _ in range(count):
            t = chr(data[pos])
            pos += 1
            if t == "Y":
                props.append(struct.unpack_from("<h", data, pos)[0]); pos += 2
            elif t == "C":
                props.append(bool(data[pos])); pos += 1
            elif t == "I":
                props.append(struct.unpack_from("<i", data, pos)[0]); pos += 4
            elif t == "F":
                props.append(struct.unpack_from("<f", data, pos)[0]); pos += 4
            elif t == "D":
                props.append(struct.unpack_from("<d", data, pos)[0]); pos += 8
            elif t == "L":
                props.append(struct.unpack_from("<q", data, pos)[0]); pos += 8
            elif t in _ARRAY:
                n, enc, clen = struct.unpack_from("<III", data, pos); pos += 12
                raw = data[pos:pos + clen]; pos += clen
                if enc == 1:
                    raw = zlib.decompress(raw)
                dt, _ = _ARRAY[t]
                props.append(np.frombuffer(raw, dtype=dt, count=n).copy())
            elif t in "SR":
                n = struct.unpack_from("<I", data, pos)[0]; pos += 4
                raw = data[pos:pos + n]; pos += n
                props.append(raw.decode("utf-8", "replace") if t == "S" else raw)
            else:
                raise ValueError(f"FBX: unknown property type {t!r}")
        return props, pos

    def read_node(pos):
        end, nprops, _plen = head.unpack_from(data, pos)
        pos += head.size
        if end == 0:
            return None, pos + 1
        nlen = data[pos]; pos += 1
        name = data[pos:pos + nlen].decode("utf-8", "replace"); pos += nlen
        props, pos = read_props(pos, nprops)
        children = []
        while pos < end - null_len + 1 and pos < end:
            child, pos = read_node(pos)
            if child is None:
                break
            children.append(child)
        return Node(name, props, children), end

    pos, top = 27, []
    while pos < len(data) - null_len:
        node, pos = read_node(pos)
        if node is None:
            break
        top.append(node)
    return Node("", [], top)


# ----------------------------------------------------------------------------- ASCII
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|;[^\n]*|[A-Za-z_][\w|]*:|\*\d+|[{},]|[^\s{},"]+')


def _read_ascii(text):
    toks = [t for t in _TOKEN.findall(text) if not t.startswith(";")]
    i = 0

    def value(t):
        if t.startswith('"'):
            return t[1:-1]
        try:
            return int(t)
        except ValueError:
            try:
                return float(t)
            except ValueError:
                return t

    def read_body():
        nonlocal i
        children = []
        while i < len(toks) and toks[i] != "}":
            children.append(read_node())
        i += 1  # "}"
        return children

    def read_node():
        nonlocal i
        name = toks[i][:-1]; i += 1
        props, children = [], []
        if i < len(toks) and toks[i].startswith("*"):  # array: *N { a: v,v,... }
            i += 1
            assert toks[i] == "{"; i += 1
            vals = []
            if toks[i] == "a:":
                i += 1
                while toks[i] != "}":
                    if toks[i] != ",":
                        vals.append(toks[i])
                    i += 1
            i += 1
            arr = np.array(vals, dtype=np.float64)
            if arr.size and np.all(arr == np.round(arr)) and not any("." in v or "e" in v.lower() for v in vals[:64]):
                arr = arr.astype(np.int64)
            return Node(name, [arr], [])
        while i < len(toks) and not toks[i].endswith(":") and toks[i] not in ("{", "}"):
            if toks[i] != ",":
                props.append(value(toks[i]))
            i += 1
        if i < len(toks) and toks[i] == "{":
            i += 1
            children = read_body()
        return Node(name, props, children)

    top = []
    while i < len(toks):
        if toks[i] == "}":
            i += 1
            continue
        top.append(read_node())
    return Node("", [], top)


def parse(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:20] == b"Kaydara FBX Binary  ":
        return _read_binary(data)
    return _read_ascii(data.decode("utf-8", "replace"))


# ----------------------------------------------------------------------------- scene
def _props70(node):
    out = {}
    p70 = node.find("Properties70") if node is not None else None
    for p in (p70.children if p70 is not None else []):
        if p.name == "P" and p.props:
            out[p.props[0]] = p.props[4:] if len(p.props) > 4 else []
    return out


def _arr(node, name):
    c = node.find(name)
    if c is None or not c.props:
        return None
    a = c.props[0]
    return np.asarray(a) if isinstance(a, np.ndarray) else np.asarray(c.props)


def _str(node, name, default=""):
    c = node.find(name)
    return str(c.props[0]) if c is not None and c.props else default


def _vec(p, key, default):
    v = p.get(key)
    if v and len(v) >= 3:
        return np.array([float(x) for x in v[:3]])
    return np.full(3, float(default))


def _rot(deg, order=0):
    x, y, z = np.radians(deg)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    seq = {0: (rx, ry, rz), 1: (rx, rz, ry), 2: (ry, rz, rx), 3: (ry, rx, rz), 4: (rz, rx, ry), 5: (rz, ry, rx)}
    a, b, c = seq.get(int(order), seq[0])  # eEulerXYZ: X applied first
    m = np.eye(4)
    m[:3, :3] = c @ b @ a
    return m


def _t(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _s(v):
    return np.diag([v[0], v[1], v[2], 1.0])


def _local(p):
    order = int(p.get("RotationOrder", [0])[0]) if p.get("RotationOrder") else 0
    active = bool(p.get("RotationActive", [0])[0]) if p.get("RotationActive") else False
    T, R, S = _vec(p, "Lcl Translation", 0), _vec(p, "Lcl Rotation", 0), _vec(p, "Lcl Scaling", 1)
    roff, rp = _vec(p, "RotationOffset", 0), _vec(p, "RotationPivot", 0)
    soff, sp = _vec(p, "ScalingOffset", 0), _vec(p, "ScalingPivot", 0)
    pre = _rot(_vec(p, "PreRotation", 0)) if active else np.eye(4)
    post = _rot(_vec(p, "PostRotation", 0)) if active else np.eye(4)
    return (_t(T) @ _t(roff) @ _t(rp) @ pre @ _rot(R, order) @ np.linalg.inv(post) @ _t(-rp)
            @ _t(soff) @ _t(sp) @ _s(S) @ _t(-sp))


def _geometric(p):
    return _t(_vec(p, "GeometricTranslation", 0)) @ _rot(_vec(p, "GeometricRotation", 0)) @ _s(_vec(p, "GeometricScaling", 1))


def _axis_matrix(gs):
    """File axes -> Y-up, right = +X, front = +Z (the convention of the s_tools writers)."""
    def ax(key, default):
        v = gs.get(key)
        return int(v[0]) if v else default
    up, ups = ax("UpAxis", 1), ax("UpAxisSign", 1)
    front, fs = ax("FrontAxis", 2), ax("FrontAxisSign", 1)
    coord, cs = ax("CoordAxis", 0), ax("CoordAxisSign", 1)
    m = np.zeros((3, 3))
    m[0, coord] = cs
    m[1, up] = ups
    m[2, front] = fs
    return m


def _layer(geo, name, data_key, index_key, n_ctrl, poly_ctrl):
    """Per polygon-vertex values of the first layer element `name` (None if missing)."""
    le = geo.find(name)
    if le is None:
        return None
    vals = _arr(le, data_key)
    if vals is None:
        return None
    dim = 2 if name == "LayerElementUV" else 3
    vals = vals.astype(np.float64).reshape(-1, dim)
    mapping = _str(le, "MappingInformationType", "ByPolygonVertex")
    ref = _str(le, "ReferenceInformationType", "Direct")
    idx = _arr(le, index_key) if ref in ("IndexToDirect", "Index") else None
    if mapping in ("ByPolygonVertex",):
        sel = idx if idx is not None else np.arange(len(poly_ctrl))
    elif mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
        sel = (idx[poly_ctrl] if idx is not None else poly_ctrl)
    elif mapping == "AllSame":
        sel = np.zeros(len(poly_ctrl), dtype=np.int64)
    else:
        return None
    sel = np.asarray(sel, dtype=np.int64)
    sel = np.where(sel < 0, 0, sel)  # -1 = no value
    return vals[sel]


def load_meshes(path):
    """-> dict(vertices (N,3) meters Y-up, faces (F,3), uvs (N,2) or None, normals (N,3) or None)."""
    root = parse(path)
    gs = _props70(root.find("GlobalSettings"))
    unit_m = float(gs.get("UnitScaleFactor", [1.0])[0]) / 100.0 if gs.get("UnitScaleFactor") else 0.01
    axes = _axis_matrix(gs)
    objs = root.find("Objects")
    if objs is None:
        raise ValueError("FBX: no objects")
    models, geos = {}, {}
    for n in objs.children:
        if n.name == "Model" and n.props:
            models[n.props[0]] = n
        elif n.name == "Geometry" and len(n.props) >= 3 and n.props[2] == "Mesh":
            geos[n.props[0]] = n
    parent, geo_model = {}, {}
    conns = root.find("Connections")
    for c in (conns.children if conns is not None else []):
        if c.name != "C" or len(c.props) < 3 or c.props[0] != "OO":
            continue
        child, par = c.props[1], c.props[2]
        if child in geos and par in models:
            geo_model[child] = par
        elif child in models:
            parent[child] = par
    world_cache = {}

    def world(mid):
        if mid in world_cache:
            return world_cache[mid]
        m = _local(_props70(models[mid]))
        p = parent.get(mid)
        if p in models:
            m = world(p) @ m
        world_cache[mid] = m
        return m

    all_v, all_f, all_uv, all_n, base = [], [], [], [], 0
    has_uv = has_n = True
    for gid, geo in geos.items():
        verts = _arr(geo, "Vertices")
        pvi = _arr(geo, "PolygonVertexIndex")
        if verts is None or pvi is None or not len(pvi):
            continue
        verts = verts.astype(np.float64).reshape(-1, 3)
        pvi = pvi.astype(np.int64)
        ends = pvi < 0
        ctrl = np.where(ends, ~pvi, pvi)
        mid = geo_model.get(gid)
        mat = np.eye(4)
        if mid in models:
            mat = world(mid) @ _geometric(_props70(models[mid]))
        wv = (np.c_[verts, np.ones(len(verts))] @ mat.T)[:, :3] @ axes.T * unit_m
        uv = _layer(geo, "LayerElementUV", "UV", "UVIndex", len(verts), ctrl)
        if uv is not None and np.ptp(uv[:, 0]) * np.ptp(uv[:, 1]) < 1e-10:
            uv = None  # degenerate set (e.g. 3ds Max writes one (0, 0) UV for a mesh without mapping)
        nrm = _layer(geo, "LayerElementNormal", "Normals", "NormalsIndex", len(verts), ctrl)
        if nrm is not None:
            nm = np.linalg.inv(mat[:3, :3]).T
            nrm = nrm @ nm.T @ axes.T
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        # fan triangulation of each polygon, per polygon-vertex corners
        starts = np.r_[0, np.nonzero(ends)[0][:-1] + 1]
        stops = np.nonzero(ends)[0]
        tris = [(s, s + k, s + k + 1) for s, e in zip(starts, stops) for k in range(1, e - s)]
        if not tris:
            continue
        tris = np.asarray(tris, dtype=np.int64)
        # corners -> unique (control point, uv, normal) vertices
        key = [ctrl[:, None].astype(np.float64)]
        if uv is not None:
            key.append(np.round(uv, 7))
        if nrm is not None:
            key.append(np.round(nrm, 5))
        uniq, inv = np.unique(np.hstack(key), axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        all_v.append(wv[ctrl[first]])
        all_uv.append(uv[first] if uv is not None else None)
        all_n.append(nrm[first] if nrm is not None else None)
        has_uv &= uv is not None
        has_n &= nrm is not None
        if np.linalg.det(mat[:3, :3]) * np.linalg.det(axes) < 0:  # mirrored: keep faces outward
            tris = tris[:, ::-1]
        all_f.append(inv[tris] + base)
        base += len(uniq)
    if not all_v:
        raise ValueError(f"FBX: no meshes in {path}")
    return {
        "vertices": np.vstack(all_v).astype(np.float32),
        "faces": np.vstack(all_f).astype(np.int64),
        "uvs": np.vstack(all_uv).astype(np.float32) if has_uv else None,
        "normals": np.vstack(all_n).astype(np.float32) if has_n else None,
    }


def to_obj(path, out_path):
    """Writes the FBX meshes as one OBJ (meters, Y-up) for mesh loaders that do not read FBX."""
    m = load_meshes(path)
    v, f, uv, n = m["vertices"], m["faces"] + 1, m["uvs"], m["normals"]
    lines = ["# s_tools: from FBX, meters, Y-up\no mesh\n"]
    lines.append("".join("v %.7f %.7f %.7f\n" % tuple(p) for p in v))
    if uv is not None:
        lines.append("".join("vt %.7f %.7f\n" % tuple(t) for t in uv))
    if n is not None:
        lines.append("".join("vn %.6f %.6f %.6f\n" % tuple(t) for t in n))
    if uv is not None and n is not None:
        lines.append("".join("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (a, a, a, b, b, b, c, c, c) for a, b, c in f))
    elif uv is not None:
        lines.append("".join("f %d/%d %d/%d %d/%d\n" % (a, a, b, b, c, c) for a, b, c in f))
    elif n is not None:
        lines.append("".join("f %d//%d %d//%d %d//%d\n" % (a, a, b, b, c, c) for a, b, c in f))
    else:
        lines.append("".join("f %d %d %d\n" % (a, b, c) for a, b, c in f))
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))
    return m
