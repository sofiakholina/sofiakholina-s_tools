// s_tools UI helpers. Only official extension hooks are used (no prototype patching),
// buttons are not saved into workflows, so workflows open fine without s_tools.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const imageInputs = new Map(); // s_tools node type -> names of standard image-upload inputs

// s_tools changes only its own nodes (category "s_tools" / "s_tools/...").
function isOwnNode(nodeData) {
  const c = String(nodeData?.category || "");
  return c === "s_tools" || c.startsWith("s_tools/");
}

function collectImageInputs(nodeData) {
  const names = new Set();
  for (const group of ["required", "optional"]) {
    const inputs = nodeData?.input?.[group];
    if (!inputs) continue;
    for (const [name, spec] of Object.entries(inputs)) {
      if (!Array.isArray(spec)) continue;
      const [type, opts] = spec;
      if (!opts || opts.image_upload !== true || opts.widgetType) continue; // custom widgets (e.g. Painter) skipped
      if (type === "IMAGEUPLOAD") {
        if (opts.imageInputName) names.add(opts.imageInputName); // upload button -> its file list
      } else if (Array.isArray(type) || type === "COMBO") {
        names.add(name); // the file list itself
      }
    }
  }
  return [...names];
}

function addButton(node, label, onClick) {
  const w = node.addWidget("button", label, null, onClick, { serialize: false });
  if (w) w.serialize = false;
  return w;
}

function redraw(node) {
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
}

// Drops the node's cached preview through the public app.nodeOutputs object
// (root graph key: node id; inside a subgraph: "<subgraph id>:<node id>").
function dropPreview(node) {
  const outputs = app.nodeOutputs;
  if (!outputs) return;
  const keys = [String(node.id)];
  const root = app.rootGraph ?? app.graph;
  if (node.graph && root && node.graph !== root && node.graph.id) keys.push(`${node.graph.id}:${node.id}`);
  for (const k of keys) if (k in outputs) delete outputs[k];
}

function clearImage(node, name) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return;
  try { dropPreview(node); } catch (e) { console.warn("[s_tools] clear preview:", e); }
  w.value = "";
  try { w.callback?.(""); } catch (e) { console.warn("[s_tools] clear callback:", e); }
  node.imgs = undefined;
  node.imageIndex = null;
  redraw(node);
}

// The empty "folder" field (= ComfyUI output folder) shows that folder's real path.
// The workflow file keeps "" for it, so the workflow stays valid on other computers.
let defaultDirRequest = null;
function getDefaultDir() {
  defaultDirRequest ??= api
    .fetchApi("/s_tools/output_dir")
    .then((r) => (r.ok ? r.json() : {}))
    .then((d) => d?.path || "")
    .catch(() => "");
  return defaultDirRequest;
}

async function saveNow(node) {
  const val = (n) => node.widgets?.find((x) => x.name === n)?.value;
  try {
    const res = await api.fetchApi("/s_tools/save_now", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id: String(node.id),
        format: val("format"),
        filename_prefix: val("filename_prefix"),
        folder: val("folder") || node.__s_tools_default_dir || "",
        save_units: val("save_units") || "mm",
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || res.statusText);
    notify("success", "Saved: " + (data.paths || []).join(", "));
  } catch (e) {
    notify("warn", e.message);
  }
}

function setupSave3D(node) {
  // "Save" goes above "autosave"; buttons are not serialized, so widget values keep their order
  const save = addButton(node, "Save", () => saveNow(node));
  if (save && node.widgets) {
    node.widgets.splice(node.widgets.indexOf(save), 1);
    node.widgets.unshift(save);
  }
  addButton(node, "Choose folder", () => pickFolder(node));
  const w = node.widgets?.find((x) => x.name === "folder");
  if (!w) return;
  const fill = async () => {
    const dir = await getDefaultDir();
    if (!dir) return;
    node.__s_tools_default_dir = dir;
    if (!w.value) {
      w.value = dir;
      redraw(node);
    }
  };
  fill();
  const prevConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const r = prevConfigure?.apply(this, args);
    fill();
    return r;
  };
  const prevSerialize = node.onSerialize;
  node.onSerialize = function (o, ...args) {
    const r = prevSerialize?.apply(this, [o, ...args]);
    try {
      const dir = node.__s_tools_default_dir;
      const vals = o?.widgets_values;
      if (dir && Array.isArray(vals)) {
        const i = vals.indexOf(dir);
        if (i >= 0) vals[i] = "";
      }
      const named = o?.widgets_values_named;
      if (dir && named && named.folder === dir) named.folder = "";
    } catch (e) {
      console.warn("[s_tools] serialize:", e);
    }
    return r;
  };
}

async function pickFolder(node) {
  const w = node.widgets?.find((x) => x.name === "folder");
  if (!w) return;
  try {
    const res = await api.fetchApi("/s_tools/pick_folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start: w.value || node.__s_tools_default_dir || "" }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.error || res.statusText);
    if (data.path) {
      w.value = data.path;
      w.callback?.(data.path);
      redraw(node);
    }
  } catch (e) {
    console.warn("[s_tools] folder dialog:", e);
    notify("warn", "Folder dialog is not available here. Type the folder path into the 'folder' field. " + e.message);
  }
}

function notify(severity, text) {
  try {
    if (app.extensionManager?.toast?.add) {
      app.extensionManager.toast.add({ severity, summary: "Save 3D", detail: text, life: 6000 });
      return;
    }
  } catch (e) { /* fall through */ }
  console.warn("[s_tools]", text);
}

app.registerExtension({
  name: "s_tools.ui",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    // Only s_tools nodes get the Clear button; other packs' and core nodes stay untouched.
    if (!isOwnNode(nodeData)) return;
    const names = collectImageInputs(nodeData);
    if (names.length) imageInputs.set(nodeData.name, names);
  },

  nodeCreated(node) {
    try {
      if (node.__s_tools_ui) return; // the hook can fire more than once per node
      node.__s_tools_ui = true;
      const type = node.comfyClass || node.type;
      for (const name of imageInputs.get(type) || []) {
        addButton(node, "Clear", () => clearImage(node, name));
      }
      if (type === "SSave3D") setupSave3D(node);
    } catch (e) {
      console.warn("[s_tools] nodeCreated:", e);
    }
  },
});
