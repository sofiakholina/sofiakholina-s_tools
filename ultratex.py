"""s_tools glue nodes for UltraTex workflows.

- UltraTex Mesh: loads the mesh for UltraTex (visualbruno) Prep and picks the UV mode in one node.
  FBX is read by fbx_reader.py (both UltraTex Mesh nodes) and handed on as a temporary OBJ in meters.
- UltraTex Bake 6/10: UltraTex (visualbruno) bake with a 6/10 view selector, no auto-save; the second pass is lazy.
- UltraTex Mesh (GGUF): UTX Load Mesh v2 (MostAadTech) + UV mode in one node.
- UltraTex View Count (GGUF): 6 / 10 views for UTX Generate Texture Views.
- UltraTex Cache -> temp: UltraTex Modular (MostAadTech) disk cache in the ComfyUI temp folder.
- Multi-view (your photos): UltraTex View Gray (Klein input), UltraTex Side From Front/Back (side guide),
  UltraTex Photo Views (photo views -> 6 slots, lazy: unused branches are not computed); Bake keeps photo views first.

Only routes data: the work is done by the original node packs, found through their loaded modules
(no imports of ComfyUI internals). Missing pack -> clear error message.
"""

import hashlib
import io
import logging
import os
import sys
import types

import numpy as np

import folder_paths

log = logging.getLogger("s_tools")
CATEGORY = "s_tools/UltraTex"
UV_MODES = ["keep_existing", "comfy_gpu", "xatlas"]


def _pack_module(*attrs, pack=""):
    """Loaded module of a node pack that defines all `attrs`."""
    for mod in list(sys.modules.values()):
        # real modules only, attributes defined in the module itself (proxy objects such as torch.ops answer any name)
        d = getattr(mod, "__dict__", None) if isinstance(mod, types.ModuleType) else None
        if isinstance(d, dict) and all(a in d for a in attrs):
            return mod
    raise RuntimeError(f"s_tools: node pack '{pack}' is not installed or not loaded")


def _resolve(path):
    path = str(path or "").strip().strip("\"'\u00ab\u00bb\u201c\u201d\u201e\u2018\u2019").strip()
    if not path:
        raise ValueError("s_tools: mesh path is empty")
    for p in (path, os.path.join(folder_paths.get_input_directory(), path)):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"s_tools: mesh file not found: {path!r} (absolute, or relative to the input folder)")


def _stem(path):
    """File name without extension: the base name for Save 3D."""
    return os.path.splitext(os.path.basename(path))[0] or "model"


# glTF is in meters by definition, FBX is converted to meters; OBJ / PLY / STL ... have no units (scene units).
_METER_EXT = (".glb", ".gltf", ".fbx")


def _units(path):
    """Units of the file's coordinates for Save 3D (input_units)."""
    return "m" if os.path.splitext(path)[1].lower() in _METER_EXT else "scene"


def _readable(path):
    """FBX -> temporary OBJ (meters, Y-up, UVs and normals kept); other formats unchanged."""
    if os.path.splitext(path)[1].lower() != ".fbx":
        return path
    from . import fbx_reader
    st = os.stat(path)
    tag = hashlib.md5(f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")).hexdigest()[:10]
    out_dir = os.path.join(folder_paths.get_temp_directory(), "s_tools_fbx")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{_stem(path)}_{tag}.obj")
    if not os.path.isfile(out):
        m = fbx_reader.to_obj(path, out)
        log.info("s_tools: FBX %s -> %d vertices, %d faces, UVs: %s", path, len(m["vertices"]), len(m["faces"]),
                 m["uvs"] is not None)
    return out


def _has_uv(mesh):
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    return uv is not None and len(uv) == len(mesh.vertices)


def _view_list(text):
    if text is None or not str(text).strip():
        return None
    return {int(v) for v in str(text).replace(";", ",").split(",") if v.strip()}


def _priority_baker(bake_core, prep, texture_size, settings, users, gen_bias):
    """Baker in which first-pass views from your photos keep priority 1.0 and the other ones get gen_bias.

    UltraTex reads BakeSettings.side_view_bias right after sampling each first-pass non-front view
    (second-pass views use extra_view_bias). A subclass remembers the view being sampled; the settings
    answer side_view_bias for that view. UltraTex code itself is not changed."""

    class _Baker(bake_core.Baker):
        s_view = 0

        def _sample_view(self, img, mask, rig, view, s, data, dev):
            self.s_view = int(view)
            return super()._sample_view(img, mask, rig, view, s, data, dev)

    baker = _Baker(prep, texture_size)

    class _Settings(type(settings)):
        side_view_bias = property(lambda self: 1.0 if baker.s_view in users else gen_bias)

    settings.__class__ = _Settings
    log.info("UltraTex Bake (s_tools): photo views %s keep priority 1.0, other views %.2f", sorted(users), gen_bias)
    return baker


# ======================================================================= visualbruno
class SUltraTexMesh:
    """Mesh for UltraTex Prep. keep_existing: the file's own UVs (a file without UVs falls back to comfy_gpu).
    comfy_gpu / xatlas: geometry only, Prep unwraps it (GPU / CPU)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh_path": ("STRING", {"default": "", "tooltip": "Mesh file (OBJ/FBX/GLB/PLY...), absolute or relative to the input folder."}),
            "uv_mode": (UV_MODES, {"default": "keep_existing", "tooltip": "keep_existing: your own UVs. comfy_gpu: fast new unwrap. xatlas: new unwrap on the CPU (slow on dense meshes)."}),
        }}

    # "*": UltraTex Prep (classic node) checks a list type against its own option list exactly
    RETURN_TYPES = ("TRIMESH", "*", "STRING", "STRING")
    RETURN_NAMES = ("trimesh", "uv_mode", "name", "units")
    OUTPUT_TOOLTIPS = ("Mesh for UltraTex Prep.", "UV mode for UltraTex Prep.", "File name for Save 3D (filename_prefix).",
                       "Units of the file for Save 3D (input_units): FBX / GLB = their own units, OBJ / PLY / STL = scene units.")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, mesh_path, uv_mode):
        import trimesh
        path = _resolve(mesh_path)
        mesh = trimesh.load(_readable(path), force="mesh")
        if uv_mode == "keep_existing" and not _has_uv(mesh):
            log.warning("UltraTex Mesh: %s has no UVs, using comfy_gpu", path)
            uv_mode = "comfy_gpu"
        if uv_mode != "keep_existing":
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return (mesh, uv_mode, _stem(path), _units(path))


class SUltraTexBake:
    """UltraTex Bake Texture (visualbruno) with a 6/10 view selector. Writes no files (save with Save 3D)."""

    VIEW_COUNTS = ["6", "10"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": ("ULTRATEX_PREP",),
                "albedo_views": ("IMAGE",),
                "view_count": (cls.VIEW_COUNTS, {"default": "6", "tooltip": "6: first pass only (the second pass is not computed). 10: + 4 side views at 45 deg."}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512}),
                "view_weight_power": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 16.0, "step": 0.5}),
                "best_view_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "edge_feather_px": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 1.0}),
                "fusion": (["two_band", "legacy"], {"default": "two_band"}),
                "detail_sharpness": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 128.0, "step": 1.0}),
                "exposure_match": ("BOOLEAN", {"default": True}),
                "hole_fill": (["mesh", "nearest"], {"default": "mesh"}),
                "views_2_weight": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "prep_2": ("ULTRATEX_PREP", {"lazy": True, "tooltip": "UltraTex Rotate Rig output (10 views only)."}),
                "albedo_views_2": ("IMAGE", {"lazy": True, "tooltip": "Second-pass sampler views (10 views only)."}),
                "user_views": ("STRING", {"forceInput": True, "tooltip": "First-pass views drawn from your photos (0 front, 1 left, 2 back, 3 right), comma separated: from UltraTex Photo Views. They keep priority 1.0."}),
                "generated_view_bias": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Priority of the other first-pass views when user_views is connected. 1.0 = original UltraTex. Lower = your photo views win a texel up to steeper angles."}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("trimesh", "albedo_texture", "model_3d")
    FUNCTION = "bake"
    CATEGORY = CATEGORY
    DESCRIPTION = "Bakes with the original UltraTex (visualbruno) code. Writes no files: save with Save 3D."

    def check_lazy_status(self, view_count, prep_2=None, albedo_views_2=None, **kwargs):
        if str(view_count) == "6":
            return []
        return [n for n, v in (("prep_2", prep_2), ("albedo_views_2", albedo_views_2)) if v is None]

    def bake(self, prep, albedo_views, view_count, texture_size, view_weight_power, best_view_mix, edge_feather_px,
             fusion, detail_sharpness, exposure_match, hole_fill, views_2_weight, prep_2=None, albedo_views_2=None,
             user_views=None, generated_view_bias=1.0):
        import torch
        ut = _pack_module("UltraTexBake", "bake_core", "VIEWS_2_CHOICES", pack="ComfyUI-UltraTex (visualbruno)")
        bake_core, Bake = ut.bake_core, ut.UltraTexBake
        settings = bake_core.BakeSettings(
            power=view_weight_power, best_view_mix=best_view_mix, edge_px=edge_feather_px, fusion=fusion,
            detail_power=float(detail_sharpness), exposure_match=bool(exposure_match), hole_fill=hole_fill,
            extra_view_bias=float(views_2_weight),
        )
        sets = [Bake._views(albedo_views, prep)]
        if str(view_count) == "10":
            if prep_2 is None or albedo_views_2 is None:
                raise ValueError("UltraTex Bake: 10 views need prep_2 and albedo_views_2 connected")
            if prep_2.vertices.shape != prep.vertices.shape or not np.array_equal(prep_2.uvs, prep.uvs):
                raise ValueError("UltraTex Bake: prep_2 must come from UltraTex Rotate Rig on the same prep")
            sets.append((*Bake._views(albedo_views_2, prep_2), ut.VIEWS_2_CHOICES["no top/bottom"]))
        log.info("UltraTex Bake (s_tools): %s views", view_count)
        users = _view_list(user_views)
        if users is not None and float(generated_view_bias) < 1.0:
            baker = _priority_baker(bake_core, prep, texture_size, settings, users, float(generated_view_bias))
        else:
            baker = bake_core.Baker(prep, texture_size)
        try:
            albedo_np, _ = baker.bake(sets, settings)
        finally:
            baker.release()
        albedo = bake_core.to_pil(albedo_np)
        mesh = bake_core.textured_mesh(prep, albedo, None)
        model_3d = None
        File3D = getattr(ut, "File3D", None)
        if File3D is not None:
            model_3d = File3D(io.BytesIO(mesh.export(file_type="glb")), file_format="glb")
        else:
            log.warning("UltraTex Bake: this ComfyUI has no 3D file type, the 3D preview stays empty")
        albedo_t = torch.from_numpy(np.asarray(albedo).astype(np.float32) / 255.0)[None]
        return (mesh, albedo_t, model_3d)


# ======================================================================= multi-view (your photos)
VIEW_NAMES = ["front", "left", "back", "right", "top", "bottom"]  # UltraTex view order


def _gray(normal_views, idx):
    import torch
    from PIL import Image, ImageOps
    arr = (normal_views[idx, ..., :3].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    g = ImageOps.autocontrast(Image.fromarray(arr).convert("L"), cutoff=1).convert("RGB")
    return np.asarray(g).astype(np.float32) / 255.0


class SUltraTexViewGray:
    """One UltraTex normal view (UltraTex Prep normal_views) in gray: Klein input without colour leaking into the paint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "normal_views": ("IMAGE", {"tooltip": "UltraTex Prep output normal_views (6 views)."}),
            "view": (VIEW_NAMES, {"default": "front"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "gray"
    CATEGORY = CATEGORY

    def gray(self, normal_views, view):
        import torch
        idx = VIEW_NAMES.index(view)
        if normal_views.shape[0] <= idx:
            raise ValueError(f"UltraTex View Gray: expected 6 normal views, got {normal_views.shape[0]}")
        return (torch.from_numpy(_gray(normal_views, idx))[None],)


class SUltraTexSideFromFrontBack:
    """Side guide from the finished front and back views: bakes them, renders the side camera and fills the rest
    with gray normals.

    guide: what Klein completes (painted where front/back see the side at least keep_above head-on, gray elsewhere).
    projection + paste_mask: paste the projection back onto Klein's side view wherever front/back saw the surface
    at least paste_above head-on, so the side matches them exactly there."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prep": ("ULTRATEX_PREP",),
            "normal_views": ("IMAGE", {"tooltip": "UltraTex Prep output normal_views (6 views)."}),
            "front": ("IMAGE", {"tooltip": "Finished front view."}),
            "back": ("IMAGE", {"tooltip": "Finished back view."}),
            "view": (["left", "right"], {"default": "left"}),
            "keep_above": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "guide: keep the projected colour where front/back saw the surface at least this head-on (cosine)."}),
            "keep_feather": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
            "paste_above": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "paste_mask: where front/back saw the surface at least this head-on (cosine)."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("guide", "projection", "paste_mask")
    FUNCTION = "project"
    CATEGORY = CATEGORY

    def project(self, prep, normal_views, front, back, view, keep_above, keep_feather, paste_above):
        import torch
        from PIL import Image
        ut = _pack_module("UltraTexBake", "bake_core", "VIEWS_2_CHOICES", pack="ComfyUI-UltraTex (visualbruno)")
        bake_core = ut.bake_core
        size = int(normal_views.shape[1])
        views = np.ones((6, size, size, 3), np.float32)
        for i, img in ((0, front), (2, back)):
            im = img[:1, ..., :3].movedim(-1, 1).float()
            if im.shape[2] != size or im.shape[3] != size:
                im = torch.nn.functional.interpolate(im, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
            views[i] = im[0].movedim(0, -1).cpu().numpy()
        masks = np.stack([np.asarray(Image.fromarray(prep.normal_rgba[v, ..., 3]).resize((size, size), Image.Resampling.BILINEAR)) > 0
                          for v in range(6)])
        rig = None if np.allclose(prep.rig, np.eye(3)) else prep.rig
        baker = bake_core.Baker(prep, 2048)
        try:
            tex, conf_tex = baker.bake([(views, masks, rig, [0, 2])], bake_core.BakeSettings())
            rviews, covers, conf = baker.render_views(tex, conf_tex, size, rig)
        finally:
            baker.release()
        idx = VIEW_NAMES.index(view)
        lo = keep_above - keep_feather / 2
        keep = (np.clip((conf[idx] - lo) / keep_feather, 0.0, 1.0) * covers[idx]).astype(np.float32)
        plo = paste_above - keep_feather / 2
        paste = (np.clip((conf[idx] - plo) / keep_feather, 0.0, 1.0) * covers[idx]).astype(np.float32)
        guide = keep[..., None] * rviews[idx] + (1 - keep[..., None]) * _gray(normal_views, idx)
        log.info("UltraTex Side From Front/Back: %s, %.0f%% of its surface comes from front/back", view,
                 100.0 * keep.sum() / max(float(covers[idx].sum()), 1.0))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]
        return (t(guide), t(rviews[idx]), t(paste))


class SUltraTexPhotoViews:
    """Puts the views drawn from your photos into the 6 UltraTex slots (front, left, back, right, top, bottom).

    front: always (from the front photo). back / left / right: from their photo when it is loaded. A side without
    a photo is completed from front + back when the back photo is loaded; otherwise the slot keeps base_views
    (UltraTex). Only the branches needed are computed (lazy inputs).
    Outputs the 6 views, keep_mask (1 = keep, 0 = UltraTex may repaint) and user_views for UltraTex Bake."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_views": ("IMAGE", {"tooltip": "6 views of an UltraTex Sampler: used where there is no view of yours."}),
                "front": ("IMAGE", {"tooltip": "Front view drawn from the front photo."}),
                "use_back": ("BOOLEAN", {"forceInput": True}),
                "use_left": ("BOOLEAN", {"forceInput": True}),
                "use_right": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
                "back": ("IMAGE", {"lazy": True, "tooltip": "Back view drawn from the back photo."}),
                "left_photo": ("IMAGE", {"lazy": True, "tooltip": "Left view drawn from the left photo."}),
                "left_completed": ("IMAGE", {"lazy": True, "tooltip": "Left view completed from front + back (no left photo)."}),
                "right_photo": ("IMAGE", {"lazy": True, "tooltip": "Right view drawn from the right photo."}),
                "right_completed": ("IMAGE", {"lazy": True, "tooltip": "Right view completed from front + back (no right photo)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("views", "keep_mask", "user_views")
    FUNCTION = "compose"
    CATEGORY = CATEGORY

    @staticmethod
    def _plan(use_back, use_left, use_right):
        """slot -> input name, for the slots that get a view of yours."""
        plan = {0: "front"}
        if use_back:
            plan[2] = "back"
        for idx, side, use in ((1, "left", use_left), (3, "right", use_right)):
            if use:
                plan[idx] = f"{side}_photo"
            elif use_back:
                plan[idx] = f"{side}_completed"
        return plan

    def check_lazy_status(self, use_back, use_left, use_right, **kw):
        return [n for n in self._plan(use_back, use_left, use_right).values() if n != "front" and kw.get(n) is None]

    def compose(self, base_views, front, use_back, use_left, use_right, **kw):
        import torch
        if base_views.shape[0] != 6:
            raise ValueError(f"UltraTex Photo Views: base_views must hold 6 views, got {base_views.shape[0]}")
        views = base_views[..., :3].clone()
        n, h, w, _ = views.shape
        keep = torch.zeros((n, h, w), dtype=torch.float32)
        plan = self._plan(use_back, use_left, use_right)
        imgs = dict(kw, front=front)
        for idx, name in plan.items():
            im = imgs.get(name)
            if im is None:
                raise ValueError(f"UltraTex Photo Views: {name} is not connected")
            im = im[:1, ..., :3].movedim(-1, 1).float()
            if im.shape[2] != h or im.shape[3] != w:
                im = torch.nn.functional.interpolate(im, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
            views[idx] = im[0].movedim(0, -1).to(views.dtype)
            keep[idx] = 1.0
        user = sorted(i for i, name in plan.items() if not name.endswith("_completed"))
        log.info("UltraTex Photo Views: %s; from photos: %s; UltraTex: %s",
                 {VIEW_NAMES[i]: name for i, name in sorted(plan.items())}, [VIEW_NAMES[i] for i in user],
                 [VIEW_NAMES[i] for i in range(6) if i not in plan])
        return (views, keep, ",".join(str(i) for i in user))


# ======================================================================= MostAadTech (GGUF)
class SUltraTexMeshGGUF:
    """UTX Load Mesh (v2) by MostAadTech + UV mode. keep_existing keeps the file's UVs (a file without UVs is
    loaded without them and the unwrap node falls back to comfy_gpu)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh_path": ("STRING", {"default": "", "tooltip": "Mesh file (OBJ/FBX/GLB...), absolute or relative to the input folder."}),
            "uv_mode": (UV_MODES, {"default": "keep_existing"}),
        }}

    RETURN_TYPES = ("MESH", UV_MODES, "STRING", "STRING")
    RETURN_NAMES = ("mesh", "uv_mode", "name", "units")
    OUTPUT_TOOLTIPS = ("Mesh for the UltraTex Modular nodes.", "UV mode.", "File name for Save 3D (filename_prefix).",
                       "Units of the file for Save 3D (input_units): FBX / GLB = their own units, OBJ / PLY / STL = scene units.")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, mesh_path, uv_mode):
        mod = _pack_module("LoadMeshV2", "_CUSTOM", pack="UltraTex Modular (MostAadTech)")
        path = _resolve(mesh_path)
        src = _readable(path)
        copy_uvs = uv_mode == "keep_existing"
        try:
            out = mod.LoadMeshV2.execute(mod._CUSTOM, src, copy_uvs, "")
        except ValueError as exc:
            if not copy_uvs or "no valid per-vertex UVs" not in str(exc):
                raise
            log.warning("UltraTex Mesh (GGUF): %s has no UVs, using comfy_gpu", path)
            out = mod.LoadMeshV2.execute(mod._CUSTOM, src, False, "")
            uv_mode = "comfy_gpu"
        return (out.args[0], uv_mode, _stem(path), _units(path))


class SUltraTexViewCountGGUF:
    """View count for UTX Generate Texture Views limited to 6 / 10."""

    VIEW_COUNTS = [6, 10]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "view_count": (cls.VIEW_COUNTS, {"default": 6, "tooltip": "6: one rig. 10: + 4 side views at 45 deg."}),
        }}

    RETURN_TYPES = (VIEW_COUNTS,)
    RETURN_NAMES = ("view_count",)
    FUNCTION = "get"
    CATEGORY = CATEGORY

    def get(self, view_count):
        return (int(view_count),)


class SUltraTexCacheTemp:
    """Runtime policy for UltraTex Modular with its disk cache in ComfyUI's temp folder (cleared on start/exit).
    Other settings = the pack's defaults."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("UTX_RUNTIME_POLICY",)
    RETURN_NAMES = ("runtime",)
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self):
        mod = _pack_module("PrepareTexture", "RuntimePolicy", pack="UltraTex Modular (MostAadTech)")
        path = os.path.abspath(os.path.join(folder_paths.get_temp_directory(), "ultratex_cache"))
        return (mod.RuntimePolicy(path),)


NODE_CLASS_MAPPINGS = {
    "SUltraTexMesh": SUltraTexMesh,
    "SUltraTexBake": SUltraTexBake,
    "SUltraTexMeshGGUF": SUltraTexMeshGGUF,
    "SUltraTexViewCountGGUF": SUltraTexViewCountGGUF,
    "SUltraTexCacheTemp": SUltraTexCacheTemp,
    "SUltraTexViewGray": SUltraTexViewGray,
    "SUltraTexSideFromFrontBack": SUltraTexSideFromFrontBack,
    "SUltraTexPhotoViews": SUltraTexPhotoViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SUltraTexMesh": "UltraTex Mesh",
    "SUltraTexBake": "UltraTex Bake 6/10",
    "SUltraTexMeshGGUF": "UltraTex Mesh (GGUF)",
    "SUltraTexViewCountGGUF": "UltraTex View Count (GGUF)",
    "SUltraTexCacheTemp": "UltraTex Cache -> temp",
    "SUltraTexViewGray": "UltraTex View Gray (Klein input)",
    "SUltraTexSideFromFrontBack": "UltraTex Side From Front/Back",
    "SUltraTexPhotoViews": "UltraTex Photo Views (-> 6 slots)",
}
