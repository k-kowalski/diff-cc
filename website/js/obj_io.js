export function parseOBJ(text) {
  const verts = [];
  const faces = [];

  const lines = text.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "v" && parts.length >= 4) {
      verts.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      continue;
    }
    if (parts[0] === "f" && parts.length >= 4) {
      const idxs = [];
      for (let i = 1; i < parts.length; i++) {
        const token = parts[i];
        if (!token) continue;
        const vStr = token.split("/")[0];
        let v = parseInt(vStr, 10);
        if (Number.isNaN(v)) continue;
        if (v < 0) {
          v = verts.length / 3 + v;
        } else {
          v = v - 1;
        }
        idxs.push(v);
      }
      if (idxs.length >= 3) faces.push(idxs);
    }
  }

  return { verts: new Float32Array(verts), faces };
}

export function writeOBJ(verts, faces, comment = "") {
  const lines = [];
  if (comment) lines.push(`# ${comment}`);
  for (let i = 0; i < verts.length; i += 3) {
    lines.push(`v ${verts[i + 0]} ${verts[i + 1]} ${verts[i + 2]}`);
  }
  for (const f of faces) {
    const toks = f.map((v) => `${v + 1}`);
    lines.push(`f ${toks.join(" ")}`);
  }
  lines.push("");
  return lines.join("\n");
}

export function facesToTriangles(faces) {
  const tris = [];
  for (const f of faces) {
    if (f.length < 3) continue;
    const v0 = f[0];
    for (let i = 1; i + 1 < f.length; i++) {
      tris.push(v0, f[i], f[i + 1]);
    }
  }
  return new Int32Array(tris);
}

export function computeBounds(verts) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < verts.length; i += 3) {
    min[0] = Math.min(min[0], verts[i + 0]);
    min[1] = Math.min(min[1], verts[i + 1]);
    min[2] = Math.min(min[2], verts[i + 2]);
    max[0] = Math.max(max[0], verts[i + 0]);
    max[1] = Math.max(max[1], verts[i + 1]);
    max[2] = Math.max(max[2], verts[i + 2]);
  }
  return { min, max };
}

export function normalizeVertsInPlace(verts, center, scale) {
  for (let i = 0; i < verts.length; i += 3) {
    verts[i + 0] = (verts[i + 0] - center[0]) / scale;
    verts[i + 1] = (verts[i + 1] - center[1]) / scale;
    verts[i + 2] = (verts[i + 2] - center[2]) / scale;
  }
}

export function boundsCenterScale(bounds) {
  const c = [
    0.5 * (bounds.min[0] + bounds.max[0]),
    0.5 * (bounds.min[1] + bounds.max[1]),
    0.5 * (bounds.min[2] + bounds.max[2]),
  ];
  const ext = [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ];
  const scale = Math.max(ext[0], ext[1], ext[2]) || 1.0;
  return { center: c, scale };
}

