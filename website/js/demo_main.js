import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js?v=story6";

import {
  boundsCenterScale,
  computeBounds,
  facesToTriangles,
  normalizeVertsInPlace,
  parseOBJ,
} from "./obj_io.js?v=story6";

import {
  buildAdjacency,
  buildCCConns,
  buildSampleSpec,
  chamferSymmetric,
  edgeLengthLoss,
  laplacianSmoothLoss,
  samplePointsFromMesh,
  subdivideLevelsTF,
} from "./tf_cc.js?v=story6";

function $(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: #${id}`);
  return el;
}

// Optional element lookup - returns null if not found
function $opt(id) {
  return document.getElementById(id);
}

function ensureTFReady() {
  if (!window.tf) throw new Error("TensorFlow.js is not loaded.");
  return window.tf;
}

function deepCopyFaces(faces) {
  return faces.map((f) => Array.from(f));
}

function cloneMeshData(mesh) {
  return { verts: new Float32Array(mesh.verts), faces: deepCopyFaces(mesh.faces) };
}

function makeCube0Mesh() {
  const verts = new Float32Array([
    -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1,
  ]);
  const faces = [
    [0, 1, 3, 2],
    [4, 6, 7, 5],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 5, 7, 3],
  ];
  return { verts, faces };
}

function transformVerts(verts, fn) {
  const out = new Float32Array(verts.length);
  for (let i = 0; i < verts.length; i += 3) {
    const p = fn(verts[i + 0], verts[i + 1], verts[i + 2]);
    out[i + 0] = p[0];
    out[i + 1] = p[1];
    out[i + 2] = p[2];
  }
  return out;
}

async function subdivideMesh(mesh, levels) {
  const tf = ensureTFReady();
  const V = mesh.verts.length / 3;
  const conns = buildCCConns(V, mesh.faces, levels);
  const inT = tf.tensor2d(mesh.verts, [V, 3], "float32");
  const outT = tf.tidy(() => subdivideLevelsTF(inT, conns));
  // Use async data() instead of blocking dataSync()
  const outVerts = new Float32Array(await outT.data());
  outT.dispose();
  inT.dispose();
  const outFaces = conns.length ? deepCopyFaces(conns[conns.length - 1].facesOut) : deepCopyFaces(mesh.faces);
  for (const c of conns) c.dispose?.();
  return { verts: outVerts, faces: outFaces };
}


const meshCache = new Map();
let cube1Cache = null;
let sec1ScenarioCache = null;

async function fetchOBJCached(url) {
  if (!meshCache.has(url)) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
    meshCache.set(url, parseOBJ(await res.text()));
  }
  return cloneMeshData(meshCache.get(url));
}

async function getCube1Mesh() {
  if (!cube1Cache) {
    cube1Cache = await subdivideMesh(makeCube0Mesh(), 1);
  }
  return cloneMeshData(cube1Cache);
}

async function getSection1Scenario() {
  if (!sec1ScenarioCache) {
    const cage = makeCube0Mesh();
    const deformed = cloneMeshData(cage);
    deformed.verts = transformVerts(deformed.verts, (x, y, z) => [
      x * 1.38 + 0.28 * z,
      y * 0.24,
      z * 1.46 + 0.34 * x,
    ]);
    const target = await subdivideMesh(deformed, 2);
    sec1ScenarioCache = { cage, target };
  }
  return { cage: cloneMeshData(sec1ScenarioCache.cage), target: cloneMeshData(sec1ScenarioCache.target) };
}

function makeTriMesh(verts, faces, material) {
  const tri = facesToTriangles(faces);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
  geom.setIndex(new THREE.BufferAttribute(new Uint32Array(tri), 1));
  geom.computeVertexNormals();
  const mesh = new THREE.Mesh(geom, material);
  return { mesh, geom };
}

function updateTriMeshGeometry(geom, verts) {
  const pos = geom.getAttribute("position");
  if (pos.count * 3 !== verts.length) throw new Error("Position size mismatch.");
  pos.array.set(verts);
  pos.needsUpdate = true;
  geom.computeVertexNormals();
}

function buildUniqueEdges(faces) {
  const set = new Set();
  const edges = [];
  for (const f of faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i];
      const b = f[(i + 1) % f.length];
      const v0 = Math.min(a, b);
      const v1 = Math.max(a, b);
      const key = `${v0},${v1}`;
      if (set.has(key)) continue;
      set.add(key);
      edges.push([v0, v1]);
    }
  }
  return edges;
}

function makeEdgeLines(verts, faces, material) {
  const edges = buildUniqueEdges(faces);
  const pos = new Float32Array(edges.length * 2 * 3);
  for (let e = 0; e < edges.length; e++) {
    const [a, b] = edges[e];
    for (let k = 0; k < 3; k++) {
      pos[6 * e + k] = verts[3 * a + k];
      pos[6 * e + 3 + k] = verts[3 * b + k];
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(geom, material);
  return { lines, geom, edges };
}

function updateEdgeLines(geom, edges, verts) {
  const pos = geom.getAttribute("position");
  for (let e = 0; e < edges.length; e++) {
    const [a, b] = edges[e];
    for (let k = 0; k < 3; k++) {
      pos.array[6 * e + k] = verts[3 * a + k];
      pos.array[6 * e + 3 + k] = verts[3 * b + k];
    }
  }
  pos.needsUpdate = true;
}

function normalizePair(target, cage) {
  const tb = computeBounds(target.verts);
  const { center, scale } = boundsCenterScale(tb);
  const t = cloneMeshData(target);
  const c = cloneMeshData(cage);
  normalizeVertsInPlace(t.verts, center, scale);
  normalizeVertsInPlace(c.verts, center, scale);
  return { target: t, cage: c, xform: { center, scale } };
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

// Yield to UI with setTimeout - more reliable for heavy computation
function yieldToUI(ms = 0) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const LOAD_BUTTON_KEYS = [
  "prepareBtn",
  "runBtn",
  "prepareABtn",
  "runABtn",
  "prepareBBtn",
  "runBBtn",
  "stopBtn",
  "resetBtn",
];

const RUN_DISABLE_BUTTON_KEYS = [
  "prepareBtn",
  "runBtn",
  "prepareABtn",
  "runABtn",
  "prepareBBtn",
  "runBBtn",
  "resetBtn",
];

const PARAM_INPUT_KEYS = ["lr", "iters", "samples"];

function setButtonGroupDisabled(section, keys, disabled) {
  for (const key of keys) {
    const btn = section.ui[key];
    if (btn) btn.disabled = disabled;
  }
}

function setParamInputsDisabled(section, disabled) {
  for (const key of PARAM_INPUT_KEYS) {
    const input = section.ui[key];
    if (input) input.disabled = disabled;
  }
}

function setSectionLoading(section, isLoading, message = null) {
  if (message) section.ui.state.textContent = message;
  section.ui.state.classList.toggle("isLoading", isLoading);
  if (isLoading) section.ui.state.classList.remove("isRunning");
  setButtonGroupDisabled(section, LOAD_BUTTON_KEYS, isLoading);
  setParamInputsDisabled(section, isLoading);
}

function setSectionRunning(section, isRunning, message = null) {
  if (message) section.ui.state.textContent = message;
  section.ui.state.classList.toggle("isRunning", isRunning);
  if (isRunning) section.ui.state.classList.remove("isLoading");
  setButtonGroupDisabled(section, RUN_DISABLE_BUTTON_KEYS, isRunning);
  setParamInputsDisabled(section, isRunning);
}

function clearSectionBusy(section) {
  section.ui.state.classList.remove("isLoading");
  section.ui.state.classList.remove("isRunning");
  setButtonGroupDisabled(section, LOAD_BUTTON_KEYS, false);
  setParamInputsDisabled(section, false);
}

function runSection(section, message, task) {
  if (section.runPromise) return section.runPromise;
  section.runPromise = (async () => {
    setSectionRunning(section, true, message);
    try {
      await task();
    } finally {
      setSectionRunning(section, false);
      section.runPromise = null;
    }
  })();
  return section.runPromise;
}

class SectionRunner {
  constructor(opts) {
    this.name = opts.name;
    this.ui = opts.ui;
    this.defaults = { ...opts.defaults };
    this.weights = { ...opts.weights };
    this.levels = opts.levels;
    this.onAfterRun = opts.onAfterRun || null;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.chart = null;
    this.mats = null;

    this.targetObj = null;
    this.cageObj = null;
    this.targetWork = null;
    this.cageWork = null;
    this.normXform = null;

    this.cageVar = null;
    this.cageInit = null;
    this.cageAdj = null;
    this.cageEdges = null;
    this.ccConns = null;
    this.finalFaces = null;
    this.sampleSpec = null;
    this.targetSamples = null;
    this.optimizer = null;
    this.iter = 0;
    this.running = false;
    this.runPromise = null;

    this.visTarget = null;
    this.visSubd = null;
    this.visCage = null;

    this.lastScenarioKey = null;
  }

  setStatus(msg) {
    this.ui.status.textContent = msg;
  }

  setState(msg) {
    this.ui.state.textContent = msg;
  }

  setPills(items) {
    this.ui.pills.innerHTML = "";
    for (const [k, v] of items) {
      const el = document.createElement("div");
      el.className = "pill";
      el.textContent = `${k}: ${v}`;
      this.ui.pills.appendChild(el);
    }
  }

  getParams() {
    const lr = Math.max(1e-5, parseFloat(this.ui.lr.value) || this.defaults.lr);
    const iterBudget = Math.max(10, parseInt(this.ui.iters.value, 10) || this.defaults.iterBudget);
    const samples = Math.max(64, parseInt(this.ui.samples.value, 10) || this.defaults.samples);
    return { lr, iterBudget, samples };
  }

  initRenderer() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.ui.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x000000, 0);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(2.5, 3.2, 4.8);
    this.scene.add(dir);

    this.resize();
    const animate = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(animate);
    };
    animate();
  }

  resize() {
    if (!this.renderer) return;
    const rect = this.ui.canvas.getBoundingClientRect();
    this.renderer.setSize(rect.width, rect.height, false);
    this.camera.aspect = rect.width / Math.max(1, rect.height);
    this.camera.updateProjectionMatrix();
  }

  initChart() {
    this.chart = new Chart(this.ui.chartCanvas, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "total",
            data: [],
            borderColor: "#40c7b2",
            backgroundColor: "rgba(64, 199, 178, 0.14)",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.12,
          },
          {
            label: "chamfer",
            data: [],
            borderColor: "#69b0e2",
            backgroundColor: "rgba(105, 176, 226, 0.12)",
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.12,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { labels: { color: "rgba(223,238,255,0.75)" } } },
        scales: {
          x: { display: false },
          y: { ticks: { color: "rgba(223,238,255,0.75)" }, grid: { color: "rgba(223,238,255,0.1)" } },
        },
      },
    });
  }

  clearChart() {
    this.chart.data.labels = [];
    for (const ds of this.chart.data.datasets) ds.data = [];
    this.chart.update("none");
  }

  chartPush(total, chamfer) {
    this.chart.data.labels.push(this.iter);
    this.chart.data.datasets[0].data.push(total);
    this.chart.data.datasets[1].data.push(chamfer);
    if (this.chart.data.labels.length > 450) {
      this.chart.data.labels.shift();
      this.chart.data.datasets[0].data.shift();
      this.chart.data.datasets[1].data.shift();
    }
    this.chart.update("none");
  }

  ensureMaterials() {
    if (this.mats) return this.mats;
    this.mats = {
      target: new THREE.MeshStandardMaterial({
        color: 0x4fa8d8,
        metalness: 0.0,
        roughness: 0.38,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide,
      }),
      subd: new THREE.MeshStandardMaterial({
        color: 0x40c7b2,
        metalness: 0.0,
        roughness: 0.24,
        transparent: true,
        opacity: 0.82,
        side: THREE.DoubleSide,
      }),
      cage: new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.58 }),
    };
    return this.mats;
  }

  disposeTFState() {
    if (this.cageVar) this.cageVar.dispose();
    if (this.cageInit) this.cageInit.dispose();
    if (this.targetSamples) this.targetSamples.dispose();
    if (this.sampleSpec) this.sampleSpec.dispose();
    if (this.cageAdj) this.cageAdj.dispose?.();
    if (this.ccConns) for (const c of this.ccConns) c.dispose?.();
    this.cageVar = null;
    this.cageInit = null;
    this.targetSamples = null;
    this.sampleSpec = null;
    this.cageAdj = null;
    this.cageEdges = null;
    this.ccConns = null;
    this.finalFaces = null;
    this.optimizer = null;
    this.iter = 0;
  }

  resetOptimizer() {
    const tf = ensureTFReady();
    const { lr } = this.getParams();
    this.optimizer = tf.train.adam(lr);
  }

  rebuildTFGraph() {
    const tf = ensureTFReady();
    this.disposeTFState();
    if (!this.targetObj || !this.cageObj) throw new Error(`${this.name}: missing scenario meshes.`);

    const { target, cage, xform } = normalizePair(this.targetObj, this.cageObj);
    this.targetWork = target;
    this.cageWork = cage;
    this.normXform = xform;

    const V = cage.verts.length / 3;
    this.cageVar = tf.variable(tf.tensor2d(cage.verts, [V, 3], "float32"));
    this.cageInit = tf.tensor2d(cage.verts, [V, 3], "float32");
    this.cageAdj = buildAdjacency(V, cage.faces, cage.verts);
    this.cageEdges = this.cageAdj.edges;

    this.ccConns = buildCCConns(V, cage.faces, this.levels);
    this.finalFaces = this.ccConns.length ? this.ccConns[this.ccConns.length - 1].facesOut : cage.faces;

    const { samples } = this.getParams();
    const subdTri = facesToTriangles(this.finalFaces);
    this.sampleSpec = buildSampleSpec(subdTri, samples);

    const targetTri = facesToTriangles(target.faces);
    const targetPts = samplePointsFromMesh(target.verts, targetTri, samples, 2026);
    this.targetSamples = tf.tensor2d(targetPts, [samples, 3], "float32");
    this.resetOptimizer();
  }

  rebuildTargetVis() {
    const { target } = this.ensureMaterials();
    if (this.visTarget) {
      this.scene.remove(this.visTarget.mesh);
      this.visTarget.mesh.geometry.dispose();
    }
    this.visTarget = makeTriMesh(this.targetWork.verts, this.targetWork.faces, target);
    this.scene.add(this.visTarget.mesh);
  }

  rebuildSubdVis() {
    const { subd } = this.ensureMaterials();
    if (this.visSubd) {
      this.scene.remove(this.visSubd.mesh);
      this.visSubd.mesh.geometry.dispose();
    }
    const finalV = this.ccConns.length
      ? this.ccConns[this.ccConns.length - 1].numVertsOut
      : this.cageWork.verts.length / 3;
    this.visSubd = makeTriMesh(new Float32Array(finalV * 3), this.finalFaces, subd);
    this.scene.add(this.visSubd.mesh);
  }

  rebuildCageVis() {
    const { cage } = this.ensureMaterials();
    if (this.visCage) {
      this.scene.remove(this.visCage.lines);
      this.visCage.lines.geometry.dispose();
    }
    this.visCage = makeEdgeLines(this.cageWork.verts, this.cageWork.faces, cage);
    this.scene.add(this.visCage.lines);
  }

  fitCameraToScene() {
    const box = new THREE.Box3();
    let hasObject = false;
    if (this.visTarget?.mesh) {
      box.expandByObject(this.visTarget.mesh);
      hasObject = true;
    }
    if (this.visSubd?.mesh) {
      box.expandByObject(this.visSubd.mesh);
      hasObject = true;
    }
    if (this.visCage?.lines) {
      box.expandByObject(this.visCage.lines);
      hasObject = true;
    }
    if (!hasObject || box.isEmpty()) return;

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-4);
    const fov = this.camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs((maxDim / 2) / Math.tan(fov / 2));
    cameraZ *= 1.28;

    this.camera.position.set(center.x + cameraZ, center.y + cameraZ * 0.45, center.z + cameraZ);
    this.camera.near = Math.max(maxDim / 100, 1e-3);
    this.camera.far = Math.max(maxDim * 100, this.camera.near + 1);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  evaluateLossAndUpdateVis() {
    const tf = ensureTFReady();
    const ws = this.weights;
    const out = tf.tidy(() => {
      const subd = subdivideLevelsTF(this.cageVar, this.ccConns);
      const srcSamp = this.sampleSpec.sample(subd);
      const chamfer = chamferSymmetric(srcSamp, this.targetSamples);
      const smooth = ws.smooth > 0 ? laplacianSmoothLoss(this.cageVar, this.cageAdj) : tf.scalar(0);
      const edge = ws.edge > 0 ? edgeLengthLoss(this.cageVar, this.cageEdges) : tf.scalar(0);
      const anchor = ws.anchor > 0 ? tf.mean(tf.sum(tf.square(this.cageVar.sub(this.cageInit)), 1)) : tf.scalar(0);
      const total = chamfer.add(smooth.mul(ws.smooth)).add(edge.mul(ws.edge)).add(anchor.mul(ws.anchor));
      return { total, chamfer, smooth, edge, anchor, subd };
    });

    const vals = {
      total: out.total.dataSync()[0],
      chamfer: out.chamfer.dataSync()[0],
      smooth: out.smooth.dataSync()[0],
      edge: out.edge.dataSync()[0],
      anchor: out.anchor.dataSync()[0],
    };
    const subdVerts = out.subd.dataSync();
    out.total.dispose();
    out.chamfer.dispose();
    out.smooth.dispose();
    out.edge.dispose();
    out.anchor.dispose();
    out.subd.dispose();

    const cageArr = this.cageVar.dataSync();
    updateEdgeLines(this.visCage.geom, this.visCage.edges, cageArr);
    updateTriMeshGeometry(this.visSubd.geom, subdVerts);
    return vals;
  }

  optimizationStep() {
    const tf = ensureTFReady();
    const ws = this.weights;
    tf.tidy(() => {
      this.optimizer.minimize(() => {
        const subd = subdivideLevelsTF(this.cageVar, this.ccConns);
        const srcSamp = this.sampleSpec.sample(subd);
        const chamfer = chamferSymmetric(srcSamp, this.targetSamples);
        const smooth = ws.smooth > 0 ? laplacianSmoothLoss(this.cageVar, this.cageAdj).mul(ws.smooth) : tf.scalar(0);
        const edge = ws.edge > 0 ? edgeLengthLoss(this.cageVar, this.cageEdges).mul(ws.edge) : tf.scalar(0);
        const anchor =
          ws.anchor > 0 ? tf.mean(tf.sum(tf.square(this.cageVar.sub(this.cageInit)), 1)).mul(ws.anchor) : tf.scalar(0);
        return chamfer.add(smooth).add(edge).add(anchor);
      }, false, [this.cageVar]);
    });
  }

  async run(iterBudget) {
    if (!this.cageVar) return false;
    if (this.runPromise) return this.runPromise;

    const activeRun = (async () => {
      this.running = true;
      this.setStatus("Running...");
      while (this.running && this.iter < iterBudget) {
        const stepsPerFrame = 1;
        for (let i = 0; i < stepsPerFrame && this.iter < iterBudget; i++) {
          this.optimizationStep();
          this.iter++;
        }
        const vals = this.evaluateLossAndUpdateVis();
        if (!Number.isFinite(vals.total) || !Number.isFinite(vals.chamfer)) {
          this.running = false;
          this.setStatus("Diverged (NaN/Inf). Reduce LR.");
          break;
        }
        this.chartPush(vals.total, vals.chamfer);
        this.setStatus(
          `iter ${this.iter} | total ${vals.total.toExponential(3)} | chamfer ${vals.chamfer.toExponential(3)}`,
        );
        await nextFrame();
      }
      this.running = false;
      if (this.onAfterRun) this.onAfterRun(this);
      return this.iter >= iterBudget;
    })();

    this.runPromise = activeRun;
    try {
      return await activeRun;
    } finally {
      if (this.runPromise === activeRun) this.runPromise = null;
    }
  }

  stop() {
    this.running = false;
    this.setStatus("Stopped.");
  }

  reset() {
    if (!this.cageVar || !this.cageInit) return;
    this.stop();
    this.cageVar.assign(this.cageInit);
    this.iter = 0;
    this.clearChart();
    const vals = this.evaluateLossAndUpdateVis();
    this.chartPush(vals.total, vals.chamfer);
    this.setStatus("Reset to initial cage.");
  }

  getCurrentCageWorld() {
    if (!this.cageVar || !this.cageObj) return null;
    const verts = new Float32Array(this.cageVar.dataSync());
    if (this.normXform) {
      for (let i = 0; i < verts.length; i += 3) {
        verts[i + 0] = verts[i + 0] * this.normXform.scale + this.normXform.center[0];
        verts[i + 1] = verts[i + 1] * this.normXform.scale + this.normXform.center[1];
        verts[i + 2] = verts[i + 2] * this.normXform.scale + this.normXform.center[2];
      }
    }
    return { verts, faces: deepCopyFaces(this.cageObj.faces) };
  }

  async loadScenario({ key, cage, target }) {
    this.stop();
    this.lastScenarioKey = key;
    this.targetObj = cloneMeshData(target);
    this.cageObj = cloneMeshData(cage);

    this.rebuildTFGraph();
    this.rebuildTargetVis();
    this.rebuildSubdVis();
    this.rebuildCageVis();
    this.clearChart();
    this.fitCameraToScene();
    const vals = this.evaluateLossAndUpdateVis();
    this.chartPush(vals.total, vals.chamfer);

    const params = this.getParams();
    this.setPills([
      ["scenario", key],
      ["cage V", `${this.cageObj.verts.length / 3}`],
      ["target V", `${this.targetObj.verts.length / 3}`],
      ["samples", `${params.samples}`],
    ]);
    this.setStatus(`Loaded ${key}.`);
  }
}

const sec1 = {
  runner: null,
  runPromise: null,
  ui: {
    canvas: $("sec1Canvas"),
    chartCanvas: $("sec1Chart"),
    status: $opt("sec1Status"),
    state: $("sec1State"),
    pills: $opt("sec1Pills"),
    prepareBtn: $opt("sec1PrepareBtn"),
    runBtn: $("sec1RunBtn"),
    stopBtn: $("sec1StopBtn"),
    resetBtn: $("sec1ResetBtn"),
    lr: $("sec1Lr"),
    iters: $("sec1Iters"),
    samples: $("sec1Samples"),
  },
};

const sec2 = {
  runner: null,
  stageAOptimized: null,
  runPromise: null,
  ui: {
    canvas: $("sec2Canvas"),
    chartCanvas: $("sec2Chart"),
    status: $opt("sec2Status"),
    state: $opt("sec2State"), // Legacy fallback
    stateA: $opt("sec2StateA"),
    stateB: $opt("sec2StateB"),
    stageBGroup: $opt("sec2StageBGroup"),
    pills: $opt("sec2Pills"),
    prepareABtn: $opt("sec2PrepareABtn"),
    runABtn: $("sec2RunABtn"),
    prepareBBtn: $opt("sec2PrepareBBtn"),
    runBBtn: $("sec2RunBBtn"),
    stopBtn: $("sec2StopBtn"),
    resetBtn: $("sec2ResetBtn"),
    lr: $("sec2Lr"),
    iters: $("sec2Iters"),
    samples: $("sec2Samples"),
  },
};

const SEC1_SCENARIO_KEY = "section1_cube_to_squished";
const SEC2_STAGE_A_KEY = "section2_stageA";
const SEC2_STAGE_B_KEY = "section2_stageB_recovery";

let sec1LoadPromise = null;
let sec2StageALoadPromise = null;
let sec2StageBLoadPromise = null;

async function loadSec1Scenario() {
  if (sec1.runner.lastScenarioKey === SEC1_SCENARIO_KEY) return;
  if (sec1LoadPromise) return sec1LoadPromise;

  sec1LoadPromise = (async () => {
    setSectionLoading(sec1, true, "Preparing section 1 (building target + subdivision)...");
    try {
      await yieldToUI(10); // Allow UI to update before heavy work
      const scenario = await getSection1Scenario();
      await yieldToUI(); // Yield after heavy computation
      await sec1.runner.loadScenario({ key: SEC1_SCENARIO_KEY, cage: scenario.cage, target: scenario.target });
      sec1.ui.state.textContent = "Section 1 prepared. Run optimization.";
    } finally {
      setSectionLoading(sec1, false);
    }
  })();

  try {
    await sec1LoadPromise;
  } finally {
    sec1LoadPromise = null;
  }
}


async function loadSec2StageA() {
  if (sec2.runner.lastScenarioKey === SEC2_STAGE_A_KEY) return;
  if (sec2StageALoadPromise) return sec2StageALoadPromise;

  sec2StageALoadPromise = (async () => {
    setSectionLoading(sec2, true, "Preparing stage A (loading target + cube1 cage)...");
    try {
      await yieldToUI(10); // Allow UI to update before heavy work
      const cage = await getCube1Mesh();
      const target = await fetchOBJCached("./assets/target1_subd2.obj");
      await yieldToUI(); // Yield after heavy computation
      await sec2.runner.loadScenario({ key: SEC2_STAGE_A_KEY, cage, target });
      sec2.ui.state.textContent = "Stage A prepared. Run stage A first.";
    } finally {
      setSectionLoading(sec2, false);
    }
  })();

  try {
    await sec2StageALoadPromise;
  } finally {
    sec2StageALoadPromise = null;
  }
}


async function loadSec2StageB() {
  if (!sec2.stageAOptimized) {
    sec2.ui.state.textContent = "Stage B unavailable. Run stage A first.";
    return;
  }
  if (sec2.runner.lastScenarioKey === SEC2_STAGE_B_KEY) return;
  if (sec2StageBLoadPromise) return sec2StageBLoadPromise;

  sec2StageBLoadPromise = (async () => {
    setSectionLoading(sec2, true, "Preparing stage B (recovering level-0 cage target)...");
    try {
      await yieldToUI(10); // Allow UI to update before heavy work
      const cage = makeCube0Mesh();
      const target = cloneMeshData(sec2.stageAOptimized);
      sec2.runner.ui.lr.value = "0.006";
      await sec2.runner.loadScenario({ key: SEC2_STAGE_B_KEY, cage, target });
      sec2.ui.state.textContent = "Stage B prepared. This should plateau above stage A.";
    } finally {
      setSectionLoading(sec2, false);
    }
  })();

  try {
    await sec2StageBLoadPromise;
  } finally {
    sec2StageBLoadPromise = null;
  }
}

function bindEvents() {
  // Section 1 events
  if (sec1.ui.prepareBtn) {
    sec1.ui.prepareBtn.addEventListener("click", () => loadSec1Scenario().catch(handleError));
  }
  sec1.ui.runBtn.addEventListener("click", () =>
    runSection(sec1, "Running section 1 optimization...", async () => {
      if (sec1.runner.lastScenarioKey !== SEC1_SCENARIO_KEY) await loadSec1Scenario();
      await sec1.runner.run(parseInt(sec1.ui.iters.value, 10) || 100);
      sec1.ui.state.textContent = "Optimization complete. Reset to try again.";
    }).catch(handleError),
  );
  sec1.ui.resetBtn.addEventListener("click", () => {
    sec1.runner.reset();
    sec1.ui.state.textContent = "Ready. Click Run optimization.";
  });

  // Section 2 Stage A events
  if (sec2.ui.prepareABtn) {
    sec2.ui.prepareABtn.addEventListener("click", () => loadSec2StageA().catch(handleError));
  }
  sec2.ui.runABtn.addEventListener("click", () =>
    runSection(sec2, "Running stage A optimization...", async () => {
      if (sec2.runner.lastScenarioKey !== SEC2_STAGE_A_KEY) await loadSec2StageA();
      const completed = await sec2.runner.run(parseInt(sec2.ui.iters.value, 10) || 200);
      if (completed) {
        sec2.stageAOptimized = sec2.runner.getCurrentCageWorld();
        setSec2StateA("Stage A finished.");
        unlockStageB();
      } else {
        setSec2StateA("Stage A paused. Resume or reset.");
      }
    }).catch(handleError),
  );

  // Section 2 Stage B events
  if (sec2.ui.prepareBBtn) {
    sec2.ui.prepareBBtn.addEventListener("click", () => loadSec2StageB().catch(handleError));
  }
  sec2.ui.runBBtn.addEventListener("click", () =>
    runSection(sec2, "Running stage B optimization...", async () => {
      if (sec2.runner.lastScenarioKey !== SEC2_STAGE_B_KEY) await loadSec2StageB();
      if (sec2.runner.lastScenarioKey !== SEC2_STAGE_B_KEY) return;
      const completed = await sec2.runner.run(parseInt(sec2.ui.iters.value, 10) || 200);
      if (completed) {
        setSec2StateB("Stage B finished. Compare the higher loss floor vs stage A.");
      } else {
        setSec2StateB("Stage B paused. Resume or reset.");
      }
    }).catch(handleError),
  );

  sec2.ui.stopBtn.addEventListener("click", () => {
    sec2.runner.stop();
    clearSectionBusy(sec2);
  });
  sec2.ui.resetBtn.addEventListener("click", () => {
    sec2.runner.reset();
    setSec2StateA("Ready. Click Run stage A.");
    // Lock stage B again
    if (sec2.ui.stageBGroup) {
      sec2.ui.stageBGroup.classList.remove("unlocked");
    }
    sec2.ui.runBBtn.disabled = true;
    sec2.stageAOptimized = null;
    setSec2StateB("Complete stage A first.");
  });

  sec1.ui.stopBtn.addEventListener("click", () => {
    sec1.runner.stop();
    clearSectionBusy(sec1);
  });

  window.addEventListener("resize", () => {
    sec1.runner.resize();
    sec2.runner.resize();
  });
}

function handleError(err) {
  const msg = `Error: ${err?.message || err}`;
  console.error(err);
  if (sec1.runner) {
    sec1.runner.setStatus(msg);
    clearSectionBusy(sec1);
    if (sec1.ui.state) sec1.ui.state.textContent = msg;
  }
  if (sec2.runner) {
    sec2.runner.setStatus(msg);
    clearSectionBusy(sec2);
    setSec2StateA(msg);
  }
}

async function main() {
  ensureTFReady();

  sec1.runner = new SectionRunner({
    name: "section1",
    ui: sec1.ui,
    levels: 2,
    defaults: { lr: 0.011, iterBudget: 100, samples: 512 },
    weights: { smooth: 0.02, edge: 0.05, anchor: 0.0 },
  });
  sec2.runner = new SectionRunner({
    name: "section2",
    ui: sec2.ui,
    levels: 1,
    defaults: { lr: 0.008, iterBudget: 200, samples: 768 },
    weights: { smooth: 0.02, edge: 0.05, anchor: 0.0 },
  });

  sec1.runner.initRenderer();
  sec2.runner.initRenderer();
  sec1.runner.initChart();
  sec2.runner.initChart();

  bindEvents();

  // Set initial loading states
  sec1.ui.state.textContent = "Loading assets...";
  if (sec2.ui.stateA) sec2.ui.stateA.textContent = "Loading assets...";
  if (sec2.ui.stateB) sec2.ui.stateB.textContent = "Complete stage A first.";
  if (sec2.ui.state) sec2.ui.state.textContent = "Loading assets...";
  sec1.runner.setStatus("Preloading...");
  sec2.runner.setStatus("Preloading...");

  // Preload both sections in parallel (non-blocking)
  await yieldToUI(10);

  // Preload section 1
  const sec1Preload = (async () => {
    try {
      await loadSec1Scenario();
      sec1.ui.state.textContent = "Ready. Click Run optimization.";
      sec1.runner.setStatus("Ready.");
    } catch (err) {
      console.error("Section 1 preload failed:", err);
      sec1.ui.state.textContent = "Click Run to load and start.";
      sec1.runner.setStatus("Idle.");
    }
  })();

  // Preload section 2 stage A
  const sec2Preload = (async () => {
    try {
      await loadSec2StageA();
      if (sec2.ui.stateA) sec2.ui.stateA.textContent = "Ready. Click Run stage A.";
      if (sec2.ui.state) sec2.ui.state.textContent = "Ready. Click Run stage A.";
      sec2.runner.setStatus("Ready.");
    } catch (err) {
      console.error("Section 2 preload failed:", err);
      if (sec2.ui.stateA) sec2.ui.stateA.textContent = "Click Run stage A to load and start.";
      if (sec2.ui.state) sec2.ui.state.textContent = "Click Run stage A to load and start.";
      sec2.runner.setStatus("Idle.");
    }
  })();

  // Don't await - let them load in background
  Promise.allSettled([sec1Preload, sec2Preload]);
}

// Helper to set state text on section 2 (handles both old and new UI)
function setSec2StateA(text) {
  if (sec2.ui.stateA) sec2.ui.stateA.textContent = text;
  if (sec2.ui.state) sec2.ui.state.textContent = text;
}

function setSec2StateB(text) {
  if (sec2.ui.stateB) sec2.ui.stateB.textContent = text;
}

// Unlock stage B after stage A completes
function unlockStageB() {
  if (sec2.ui.stageBGroup) {
    sec2.ui.stageBGroup.classList.add("unlocked");
  }
  sec2.ui.runBBtn.disabled = false;
  setSec2StateB("Ready. Click Run stage B.");
}

main().catch(handleError);
