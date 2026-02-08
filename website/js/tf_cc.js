function ensureTF() {
  if (!window.tf) throw new Error("TensorFlow.js missing (tf)");
  return window.tf;
}

function edgeKey(a, b) {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function samplePointsFromMesh(verts, triIndices, numSamples, seed = 1) {
  const rnd = lcg(seed);
  const numTris = Math.floor(triIndices.length / 3);
  const out = new Float32Array(numSamples * 3);
  for (let i = 0; i < numSamples; i++) {
    const t = Math.floor(rnd() * numTris);
    const i0 = triIndices[3 * t + 0];
    const i1 = triIndices[3 * t + 1];
    const i2 = triIndices[3 * t + 2];

    let u = rnd();
    let v = rnd();
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    const w0 = 1 - u - v;
    const w1 = u;
    const w2 = v;

    const x =
      w0 * verts[3 * i0 + 0] + w1 * verts[3 * i1 + 0] + w2 * verts[3 * i2 + 0];
    const y =
      w0 * verts[3 * i0 + 1] + w1 * verts[3 * i1 + 1] + w2 * verts[3 * i2 + 1];
    const z =
      w0 * verts[3 * i0 + 2] + w1 * verts[3 * i1 + 2] + w2 * verts[3 * i2 + 2];

    out[3 * i + 0] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = z;
  }
  return out;
}

export function buildSampleSpec(triIndices, numSamples, seed = 1337) {
  const tf = ensureTF();
  const rnd = lcg(seed);
  const numTris = Math.floor(triIndices.length / 3);

  const gatherIdx = new Int32Array(numSamples * 3);
  const weights = new Float32Array(numSamples * 3);

  for (let i = 0; i < numSamples; i++) {
    const t = Math.floor(rnd() * numTris);
    gatherIdx[3 * i + 0] = triIndices[3 * t + 0];
    gatherIdx[3 * i + 1] = triIndices[3 * t + 1];
    gatherIdx[3 * i + 2] = triIndices[3 * t + 2];

    let u = rnd();
    let v = rnd();
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    weights[3 * i + 0] = 1 - u - v;
    weights[3 * i + 1] = u;
    weights[3 * i + 2] = v;
  }

  const gather = tf.tensor1d(gatherIdx, "int32");
  const w = tf.tensor2d(weights, [numSamples, 3], "float32");
  const w3 = w.reshape([numSamples, 3, 1]);

  return {
    gather,
    w,
    sample: (vertsTensor) =>
      tf.tidy(() => {
        const triVerts = tf.gather(vertsTensor, gather).reshape([numSamples, 3, 3]);
        return tf.sum(triVerts.mul(w3), 1);
      }),
    dispose: () => {
      gather.dispose();
      w.dispose();
      w3.dispose();
    },
  };
}

export function chamferSymmetric(a, b) {
  const tf = ensureTF();
  return tf.tidy(() => {
    const a2 = tf.sum(tf.square(a), 1).reshape([-1, 1]);
    const b2 = tf.sum(tf.square(b), 1).reshape([1, -1]);
    const inner = tf.matMul(a, b, false, true);
    const d2 = a2.add(b2).sub(inner.mul(2));
    const minA = tf.min(d2, 1);
    const minB = tf.min(d2, 0);
    return tf.mean(minA).add(tf.mean(minB));
  });
}

export function buildAdjacency(numVerts, faces, vertsInit) {
  const tf = ensureTF();
  const neighborSets = Array.from({ length: numVerts }, () => new Set());

  const edgesSet = new Set();
  const edgePairs = [];
  for (const f of faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i];
      const b = f[(i + 1) % f.length];
      neighborSets[a].add(b);
      neighborSets[b].add(a);
      const v0 = Math.min(a, b);
      const v1 = Math.max(a, b);
      const key = `${v0},${v1}`;
      if (edgesSet.has(key)) continue;
      edgesSet.add(key);
      edgePairs.push([v0, v1]);
    }
  }

  const neighborIdxT = neighborSets.map((set) => {
    const arr = Array.from(set);
    return arr.length ? tf.tensor1d(arr, "int32") : null;
  });

  const eV0 = new Int32Array(edgePairs.length);
  const eV1 = new Int32Array(edgePairs.length);
  const eInit = new Float32Array(edgePairs.length);
  for (let i = 0; i < edgePairs.length; i++) {
    const [a, b] = edgePairs[i];
    eV0[i] = a;
    eV1[i] = b;
    const dx = vertsInit[3 * a + 0] - vertsInit[3 * b + 0];
    const dy = vertsInit[3 * a + 1] - vertsInit[3 * b + 1];
    const dz = vertsInit[3 * a + 2] - vertsInit[3 * b + 2];
    eInit[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  const edges = {
    v0: tf.tensor1d(eV0, "int32"),
    v1: tf.tensor1d(eV1, "int32"),
    initLen: tf.tensor1d(eInit, "float32"),
    dispose: () => {
      edges.v0.dispose();
      edges.v1.dispose();
      edges.initLen.dispose();
    },
  };

  return {
    neighborIdxT,
    edges,
    dispose: () => {
      for (const t of neighborIdxT) t?.dispose();
      edges.dispose();
    },
  };
}

export function laplacianSmoothLoss(verts, adjacency) {
  const tf = ensureTF();
  return tf.tidy(() => {
    const terms = [];
    const V = verts.shape[0];
    for (let i = 0; i < V; i++) {
      const nbr = adjacency.neighborIdxT[i];
      if (!nbr) continue;
      const vi = verts.slice([i, 0], [1, 3]).squeeze([0]);
      const meanNbr = tf.mean(tf.gather(verts, nbr), 0);
      terms.push(tf.sum(tf.square(vi.sub(meanNbr))));
    }
    if (!terms.length) return tf.scalar(0);
    return tf.mean(tf.stack(terms));
  });
}

export function edgeLengthLoss(verts, edges) {
  const tf = ensureTF();
  return tf.tidy(() => {
    const p0 = tf.gather(verts, edges.v0);
    const p1 = tf.gather(verts, edges.v1);
    const len = tf.sqrt(tf.sum(tf.square(p0.sub(p1)), 1));
    return tf.mean(tf.square(len.sub(edges.initLen)));
  });
}

function buildCCConn(numVertsIn, facesIn) {
  const tf = ensureTF();
  const F = facesIn.length;

  const edges = new Map();
  const vertexFaces = Array.from({ length: numVertsIn }, () => []);
  const neighborSets = Array.from({ length: numVertsIn }, () => new Set());
  const boundaryNeighborSets = Array.from({ length: numVertsIn }, () => new Set());

  for (let fi = 0; fi < facesIn.length; fi++) {
    const f = facesIn[fi];
    for (const v of f) vertexFaces[v].push(fi);
    for (let i = 0; i < f.length; i++) {
      const a = f[i];
      const b = f[(i + 1) % f.length];
      neighborSets[a].add(b);
      neighborSets[b].add(a);

      const v0 = Math.min(a, b);
      const v1 = Math.max(a, b);
      const key = `${v0},${v1}`;
      const e = edges.get(key);
      if (!e) {
        edges.set(key, { v0, v1, f0: fi, f1: -1 });
      } else if (e.f1 === -1) {
        e.f1 = fi;
      }
    }
  }

  const edgeList = Array.from(edges.values());
  const E = edgeList.length;
  const facePointStart = numVertsIn;
  const edgePointStart = numVertsIn + F;
  const numVertsOut = numVertsIn + F + E;

  const edgeIndexByKey = new Map();
  for (let ei = 0; ei < edgeList.length; ei++) {
    const e = edgeList[ei];
    edgeIndexByKey.set(`${e.v0},${e.v1}`, ei);
    if (e.f1 === -1) {
      boundaryNeighborSets[e.v0].add(e.v1);
      boundaryNeighborSets[e.v1].add(e.v0);
    }
  }

  const vertexNeighbors = neighborSets.map((s) => Array.from(s));
  const boundaryNeighbors = boundaryNeighborSets.map((s) => {
    const arr = Array.from(s);
    return arr.length === 2 ? arr : null;
  });

  const facesOut = [];
  for (let fi = 0; fi < facesIn.length; fi++) {
    const f = facesIn[fi];
    const fp = facePointStart + fi;
    const k = f.length;
    for (let i = 0; i < k; i++) {
      const v = f[i];
      const vNext = f[(i + 1) % k];
      const vPrev = f[(i + k - 1) % k];
      const eNext = edgePointStart + edgeIndexByKey.get(edgeKey(v, vNext));
      const ePrev = edgePointStart + edgeIndexByKey.get(edgeKey(vPrev, v));
      facesOut.push([v, eNext, fp, ePrev]);
    }
  }

  const edgeV0 = new Int32Array(E);
  const edgeV1 = new Int32Array(E);
  const edgeF0 = new Int32Array(E);
  const edgeF1 = new Int32Array(E);
  for (let i = 0; i < E; i++) {
    const e = edgeList[i];
    edgeV0[i] = e.v0;
    edgeV1[i] = e.v1;
    edgeF0[i] = e.f0;
    edgeF1[i] = e.f1;
  }

  const faceIdxT = facesIn.map((f) => tf.tensor1d(f, "int32"));
  const vertexFacesT = vertexFaces.map((arr) => (arr.length ? tf.tensor1d(arr, "int32") : null));
  const vertexNeighborsT = vertexNeighbors.map((arr) => (arr.length ? tf.tensor1d(arr, "int32") : null));
  const boundaryNeighborsT = boundaryNeighbors.map((arr) => (arr ? tf.tensor1d(arr, "int32") : null));

  const conn = {
    numVertsIn,
    numVertsOut,
    facesIn,
    facesOut,
    faceIdxT,
    vertexFacesT,
    vertexNeighborsT,
    boundaryNeighborsT,
    vertexValence: vertexFaces.map((arr) => arr.length),
    edgeV0T: tf.tensor1d(edgeV0, "int32"),
    edgeV1T: tf.tensor1d(edgeV1, "int32"),
    edgeF0T: tf.tensor1d(edgeF0, "int32"),
    edgeF1T: tf.tensor1d(edgeF1, "int32"),
    dispose: () => {
      for (const t of faceIdxT) t.dispose();
      for (const t of vertexFacesT) t?.dispose();
      for (const t of vertexNeighborsT) t?.dispose();
      for (const t of boundaryNeighborsT) t?.dispose();
      conn.edgeV0T.dispose();
      conn.edgeV1T.dispose();
      conn.edgeF0T.dispose();
      conn.edgeF1T.dispose();
    },
  };

  return conn;
}

export function buildCCConns(numVerts0, faces0, levels) {
  const conns = [];
  let numVerts = numVerts0;
  let faces = faces0;
  for (let l = 0; l < levels; l++) {
    const conn = buildCCConn(numVerts, faces);
    conns.push(conn);
    numVerts = conn.numVertsOut;
    faces = conn.facesOut;
  }
  return conns;
}

function ccSubdivideOnceTF(verts, conn) {
  const tf = ensureTF();
  return tf.tidy(() => {
    const V = conn.numVertsIn;
    const F = conn.facesIn.length;
    const E = conn.edgeV0T.shape[0];

    const facePoints = tf.stack(
      conn.faceIdxT.map((idxT) => tf.mean(tf.gather(verts, idxT), 0)),
    );

    const p0 = tf.gather(verts, conn.edgeV0T);
    const p1 = tf.gather(verts, conn.edgeV1T);
    const fp0 = tf.gather(facePoints, conn.edgeF0T);
    const f1Safe = tf.maximum(conn.edgeF1T, tf.scalar(0, "int32"));
    const fp1 = tf.gather(facePoints, f1Safe.toInt());

    const interior = p0.add(p1).add(fp0).add(fp1).mul(0.25);
    const boundary = p0.add(p1).mul(0.5);
    const isBoundary = conn.edgeF1T.equal(-1).reshape([E, 1]);
    const edgePoints = tf.where(isBoundary, boundary, interior);

    const newOld = [];
    for (let i = 0; i < V; i++) {
      const p = verts.slice([i, 0], [1, 3]).squeeze([0]);
      const bNbr = conn.boundaryNeighborsT[i];
      if (bNbr) {
        const meanNbr = tf.mean(tf.gather(verts, bNbr), 0);
        newOld.push(p.mul(0.75).add(meanNbr.mul(0.25)));
        continue;
      }

      const facesT = conn.vertexFacesT[i];
      const nbrT = conn.vertexNeighborsT[i];
      const n = conn.vertexValence[i];
      if (!facesT || !nbrT || n <= 0) {
        newOld.push(p);
        continue;
      }

      const Favg = tf.mean(tf.gather(facePoints, facesT), 0);
      const meanNbr = tf.mean(tf.gather(verts, nbrT), 0);
      const Ravg = p.add(meanNbr).mul(0.5);
      newOld.push(Favg.add(Ravg.mul(2)).add(p.mul(n - 3)).div(n));
    }
    const newOldT = tf.stack(newOld);
    return tf.concat([newOldT, facePoints, edgePoints], 0);
  });
}

export function subdivideLevelsTF(verts0, conns) {
  const tf = ensureTF();
  return tf.tidy(() => {
    let v = verts0;
    for (const conn of conns) v = ccSubdivideOnceTF(v, conn);
    return v;
  });
}

export function samplePointsFromMeshTF(verts, sampleSpec) {
  return sampleSpec.sample(verts);
}
