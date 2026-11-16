"""s_tools: own ComfyUI nodes and UI helpers.

- Save 3D: saves a MESH (or a textured TRIMESH from UltraTex) as FBX / OBJ / GLB to the ComfyUI output folder
  or any folder; a texture is written as PNG next to OBJ / FBX and embedded in GLB.
- UltraTex glue nodes (ultratex.py).
- Load Image (optional): image upload that may stay empty; outputs the image and a "use" flag.
- "Clear" button on s_tools nodes with a standard image upload widget (js/s_tools.js); other nodes are not touched.

Uses only the classic node API (NODE_CLASS_MAPPINGS) and numpy, no imports of ComfyUI internals,
so the pack keeps loading across ComfyUI updates.
"""

import asyncio
import copy
import hashlib
import logging
import os
import re
import subprocess
import sys

import numpy as np

import folder_paths

from .mesh_writers import WRITERS
from . import textured_writers as TW
from .ultratex import NODE_CLASS_MAPPINGS as _UT_NODES, NODE_DISPLAY_NAME_MAPPINGS as _UT_NAMES

log = logging.getLogger("s_tools")

WEB_DIRECTORY = "./js"
FORMATS = ["fbx", "obj", "glb"]
# Length of one unit in meters. OBJ has no units: it is written in the scene units of the target 3D app.
UNIT_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254}
SCENE_UNITS = list(UNIT_M)


def _source_m(input_units, save_units):
    """Meters per coordinate unit of the mesh. "m" (default): ComfyUI meshes / glTF. "scene": a unitless
    file (OBJ, PLY, STL) made in the scene units. A unit name ("mm" ...) is taken as is."""
    su = str(input_units or "m").strip().lower()
    if su == "scene":
        return UNIT_M[save_units]
    return UNIT_M.get(su, 1.0)


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
        if x.is_floating_point():
            x = x.float()
        x = x.cpu()
    return np.asarray(x)


def _mesh_items(mesh):
    v = _to_np(getattr(mesh, "vertices", None))
    f = _to_np(getattr(mesh, "faces", None))
    if v is None or f is None:
        raise ValueError("Save 3D: the input is not a mesh (no vertices/faces).")
    n = _to_np(getattr(mesh, "normals", None))
    uvs = _to_np(getattr(mesh, "uvs", None))
    tex = _to_np(getattr(mesh, "texture", None))
    vc = _to_np(getattr(mesh, "vertex_counts", None))
    fc = _to_np(getattr(mesh, "face_counts", None))
    if v.ndim == 2:
        v, f = v[None], f[None]
        n = n[None] if n is not None and n.ndim == 2 else n
        uvs = uvs[None] if uvs is not None and uvs.ndim == 2 else uvs
    if tex is not None and tex.ndim == 3:
        tex = tex[None]
    for i in range(v.shape[0]):
        nv = int(vc[i]) if vc is not None else v.shape[1]
        nf = int(fc[i]) if fc is not None else f.shape[1]
        normals = n[i, :nv].astype(np.float32) if n is not None else None
        uv = img = None
        if uvs is not None and tex is not None:
            uv = uvs[i, :nv].astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # MESH / glTF V down -> OBJ / FBX V up
            img = _tex_u8(tex[min(i, tex.shape[0] - 1)])
        yield v[i, :nv].astype(np.float32), f[i, :nf].astype(np.int64), normals, uv, img


def _tex_u8(t):
    t = np.asarray(t)
    if t.dtype != np.uint8:
        t = np.clip(t.astype(np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(t[..., :3])


def _trimesh_items(tm):
    v = np.asarray(tm.vertices, dtype=np.float32)
    f = np.asarray(tm.faces, dtype=np.int64)
    n = np.asarray(tm.vertex_normals, dtype=np.float32) if len(v) else None
    visual = getattr(tm, "visual", None)
    uv = getattr(visual, "uv", None)
    mat = getattr(visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if uv is not None and img is not None and len(uv) == len(v):
        yield v, f, n, np.asarray(uv, dtype=np.float32), _tex_u8(np.asarray(img.convert("RGB")))
    else:
        yield v, f, n, None, None


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _target(folder, filename_prefix):
    """Saves straight into the folder (no subfolders). Empty folder = ComfyUI output folder."""
    folder = (folder or "").strip().strip('"').strip()
    out_dir = os.path.normpath(os.path.expanduser(folder)) if folder else folder_paths.get_output_directory()
    os.makedirs(out_dir, exist_ok=True)
    name = (filename_prefix or "").replace("\\", "/").strip().strip("/").split("/")[-1]
    name = _BAD_CHARS.sub("_", name).strip(" .") or "ComfyUI"
    return out_dir, name


def _next_counter(out_dir, name):
    pat = re.compile(re.escape(name) + r"_(\d+)\.(?:fbx|obj|glb)$", re.IGNORECASE)
    nums = [int(m.group(1)) for m in (pat.match(f) for f in os.listdir(out_dir)) if m]
    return max(nums, default=0) + 1


# Last mesh of every Save 3D node (by node id), for the "Save" button. In memory only.
_LAST = {}
_LAST_MAX = 8


def _remember(node_id, items):
    if node_id is None:
        return
    _LAST.pop(node_id, None)
    _LAST[node_id] = items
    while len(_LAST) > _LAST_MAX:
        _LAST.pop(next(iter(_LAST)))


def _find_last(node_id):
    node_id = str(node_id or "")
    if node_id in _LAST:
        return _LAST[node_id]
    tail = [k for k in _LAST if k.endswith(":" + node_id)]
    if len(tail) == 1:  # node inside a subgraph: execution id is "<parent id>:<node id>"
        return _LAST[tail[0]]
    return None


def _write_items(items, format, filename_prefix, folder, save_units="mm", input_units="m"):
    """Keeps the real size in every format: FBX stores its unit, GLB is in meters, OBJ in the scene units."""
    if format not in WRITERS:
        raise ValueError(f"Save 3D: unknown format {format!r}")
    if save_units not in UNIT_M:
        raise ValueError(f"Save 3D: unknown save_units {save_units!r}")
    src_m = _source_m(input_units, save_units)
    writer = WRITERS[format]
    out_dir, name = _target(folder, filename_prefix)
    counter = _next_counter(out_dir, name)
    saved = []
    for vertices, faces, normals, uvs, texture in items:
        path = os.path.join(out_dir, f"{name}_{counter:02d}.{format}")
        unit = {}
        if format == "fbx":
            unit = {"unit_cm": src_m * 100.0}  # coordinates as they are, unit written into the file
        else:
            k = src_m if format == "glb" else src_m / UNIT_M[save_units]
            if k != 1.0:
                vertices = (vertices.astype(np.float64) * k).astype(np.float32)
        if texture is None:
            writer(path, name, vertices, faces, normals, **unit)
        elif format == "glb":
            TW.write_glb(path, name, vertices, faces, normals, uvs, texture)
        else:
            png = f"{name}_{counter:02d}.png"
            TW.save_png(os.path.join(out_dir, png), texture)
            {"obj": TW.write_obj, "fbx": TW.write_fbx}[format](path, name, vertices, faces, normals, uvs, png, **unit)
        log.info("Save 3D: %s (%d vertices, %d faces)", path, len(vertices), len(faces))
        saved.append(path)
        counter += 1
    return saved


class SSave3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "autosave": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off",
                                        "tooltip": "Save the model on every run."}),
                "format": (FORMATS, {"default": "fbx"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI",
                                               "tooltip": "File name. A counter is added: name_01.fbx, name_02.fbx ..."}),
                "folder": ("STRING", {"default": "",
                                      "tooltip": "Target folder (files go straight into it). Empty = ComfyUI output folder."}),
                "save_units": (SCENE_UNITS, {"default": "mm", "advanced": True,
                                              "tooltip": "Units of your 3D scene (3ds Max: Units Setup > System Unit). "
                                                         "OBJ has no units: it is written in these units, and an OBJ / PLY / STL "
                                                         "loaded into UltraTex is read in them. FBX and GLB keep the real size by themselves."}),
                "input_units": ("STRING", {"default": "m", "advanced": True,
                                            "tooltip": "Units of the mesh coordinates: m (ComfyUI 3D models) or the units output "
                                                       "of UltraTex Mesh connected here."}),
            },
            "optional": {
                "mesh": ("MESH", {"tooltip": "Mesh to save (e.g. after MeshSmoothNormals / DecimateMesh). A texture on it is saved too."}),
                "trimesh": ("TRIMESH", {"tooltip": "Textured mesh from UltraTex Bake (used when mesh is not connected)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "s_tools"
    DESCRIPTION = "Saves a mesh (with its texture, if any) as FBX, OBJ or GLB to the ComfyUI output folder or any folder."

    def save(self, autosave, format, filename_prefix, folder, save_units="mm", input_units="m", mesh=None,
             trimesh=None, unique_id=None):
        if mesh is not None:
            items = list(_mesh_items(mesh))
        elif trimesh is not None:
            items = list(_trimesh_items(trimesh))
        else:
            raise ValueError("Save 3D: connect mesh or trimesh.")
        input_units = input_units or "m"
        _remember(str(unique_id) if unique_id is not None else None, (items, input_units))
        if not autosave:
            return {"ui": {"text": ["autosave off"]}}
        return {"ui": {"text": _write_items(items, format, filename_prefix, folder, save_units, input_units)}}


class SMeshClay:
    """Plain grey clay look for previews: drops vertex colours / textures and the unlit flag."""

    # same look as the viewer's own "Clay" mode: #888888, roughness 0.9, metalness 0
    CLAY = {"base_color_factor": [0.2462, 0.2462, 0.2462, 1.0], "metallic_factor": 0.0,
            "roughness_factor": 0.9, "double_sided": True}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mesh": ("MESH", {"tooltip": "Mesh to show as grey clay (geometry is not changed)."})}}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "clay"
    CATEGORY = "s_tools"
    DESCRIPTION = "Grey clay look for the 3D preview: removes colours and textures, keeps the geometry."

    def clay(self, mesh):
        out = copy.copy(mesh)
        for attr in ("vertex_colors", "texture", "metallic_roughness", "normal_map", "emissive"):
            if hasattr(out, attr):
                setattr(out, attr, None)
        if hasattr(out, "unlit"):
            out.unlit = False
        if hasattr(out, "occlusion_in_mr"):
            out.occlusion_in_mr = False
        out.material = dict(self.CLAY)
        return (out,)


class SLoadImageOptional:
    """Image upload that may stay empty. Empty -> no image and use = False,
    so a lazy switch downstream skips this branch."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")
        try:
            files = sorted(f for f in os.listdir(input_dir)
                           if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(exts))
        except Exception:
            files = []
        return {
            "required": {
                "image": ([""] + files, {"image_upload": True,
                                         "tooltip": "May stay empty: then this image is not used."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "use")
    FUNCTION = "load"
    CATEGORY = "s_tools"
    DESCRIPTION = ("Load Image that may stay empty. 'use' is True only when an image is loaded; "
                   "connect it to a switch so an empty slot is skipped.")

    @staticmethod
    def _active(image):
        return bool((image or "").strip())

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not cls._active(image):
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    @classmethod
    def IS_CHANGED(cls, image):
        if not cls._active(image):
            return "unused"
        m = hashlib.sha256()
        with open(folder_paths.get_annotated_filepath(image), "rb") as f:
            m.update(f.read())
        return m.hexdigest()

    def load(self, image):
        if not self._active(image):
            return (None, False)
        import torch
        from PIL import Image, ImageOps
        with Image.open(folder_paths.get_annotated_filepath(image)) as img:
            img.seek(0)
            img = ImageOps.exif_transpose(img).convert("RGB")
            arr = np.asarray(img).astype(np.float32) / 255.0
        return (torch.from_numpy(arr)[None,], True)


# ---------- folder picker (native OS dialog, local server only) ----------

_PS_PICKER = r"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.ShowNewFolderButton = $true
$d.Description = 'Select folder'
if ($env:S_TOOLS_START -and (Test-Path -LiteralPath $env:S_TOOLS_START)) { $d.SelectedPath = $env:S_TOOLS_START }
if ($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.SelectedPath }
$owner.Dispose()
"""


def _run(cmd, env=None):
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    r = subprocess.run(cmd, capture_output=True, env=env, **kw)
    return r.stdout.decode("utf-8", "replace").lstrip("﻿").strip()


def _pick_folder_dialog(start):
    if sys.platform == "win32":
        env = dict(os.environ, S_TOOLS_START=start or "")
        return _run(["powershell", "-NoProfile", "-NonInteractive", "-STA", "-ExecutionPolicy", "Bypass",
                     "-Command", _PS_PICKER], env=env)
    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "Select folder")'
        if start and os.path.isdir(start):
            esc = start.replace("\\", "\\\\").replace('"', '\\"')
            script = f'POSIX path of (choose folder with prompt "Select folder" default location (POSIX file "{esc}"))'
        return _run(["osascript", "-e", script])
    for cmd in (["zenity", "--file-selection", "--directory", f"--filename={(start or '').rstrip('/')}/"],
                ["kdialog", "--getexistingdirectory", start or "."]):
        try:
            return _run(cmd)
        except FileNotFoundError:
            continue
    raise RuntimeError("no folder dialog available (zenity/kdialog)")


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/s_tools/pick_folder")
    async def _s_tools_pick_folder(request):
        if request.remote not in ("127.0.0.1", "::1", "localhost"):
            return web.json_response({"error": "The folder dialog works only when ComfyUI runs on this computer."}, status=403)
        try:
            data = await request.json()
        except Exception:
            data = {}
        start = (data.get("start") or "").strip() or folder_paths.get_output_directory()
        try:
            path = await asyncio.to_thread(_pick_folder_dialog, start)
        except Exception as e:
            log.warning("s_tools: folder dialog failed: %s", e)
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"path": path})

    @PromptServer.instance.routes.post("/s_tools/save_now")
    async def _s_tools_save_now(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        last = _find_last(data.get("node_id"))
        if last is None:
            return web.json_response({"error": "No model yet: run the workflow once, then press Save."}, status=404)
        try:
            items, input_units = last
            paths = await asyncio.to_thread(_write_items, items, data.get("format") or "fbx",
                                            data.get("filename_prefix") or "", data.get("folder") or "",
                                            data.get("save_units") or "mm", input_units)
        except Exception as e:
            log.warning("s_tools: save failed: %s", e)
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response({"paths": paths})

    @PromptServer.instance.routes.get("/s_tools/output_dir")
    async def _s_tools_output_dir(request):
        return web.json_response({"path": os.path.normpath(folder_paths.get_output_directory())})
except Exception as e:  # the nodes still work without the dialog
    log.warning("s_tools: folder dialog route not registered: %s", e)


NODE_CLASS_MAPPINGS = {"SSave3D": SSave3D, "SMeshClay": SMeshClay, "SLoadImageOptional": SLoadImageOptional, **_UT_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {"SSave3D": "Save 3D", "SMeshClay": "Mesh Clay",
                              "SLoadImageOptional": "Load Image (optional)", **_UT_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
