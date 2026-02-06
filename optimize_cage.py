from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from cc_subdivide import (
    catmull_clark_subdivide,
    faces_quads_to_list,
    read_obj,
    subdivide_n,
    triangulate_faces,
    write_obj,
)

try:
    import largesteps.optimize as ls_opt
    import largesteps.parameterize as ls_param

    _HAS_LARGESTEPS = True
except Exception:
    _HAS_LARGESTEPS = False


@dataclass(frozen=True)
class Stage:
    levels: int
    iters: int


def _parse_stages(spec: Optional[str], final_levels: int, iters: int) -> List[Stage]:
    """
    Parse a stages spec like: "1:200,2:400,3:800".
    If missing, use a simple 2-stage ramp for levels > 1.
    """
    if spec:
        stages: List[Stage] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            lvl_s, it_s = part.split(":")
            stages.append(Stage(levels=int(lvl_s), iters=int(it_s)))
        if not stages:
            raise ValueError("Empty --stages spec")
        return stages

    if final_levels <= 1:
        return [Stage(levels=final_levels, iters=iters)]

    # Default: short warmup at (levels-1), then full levels.
    warmup = max(1, iters // 4)
    return [
        Stage(levels=max(1, final_levels - 1), iters=warmup),
        Stage(levels=final_levels, iters=iters - warmup),
    ]


def _triangulate_to_tensor(faces: Sequence[torch.Tensor]) -> torch.Tensor:
    tris = triangulate_faces(faces)
    if not tris:
        raise ValueError("No triangles after triangulation")
    return torch.stack(tris, dim=0).to(dtype=torch.long)


def _quads_to_tris_tensor(quads: torch.Tensor) -> torch.Tensor:
    quads = quads.to(dtype=torch.long)
    t1 = quads[:, [0, 1, 2]]
    t2 = quads[:, [2, 3, 0]]
    return torch.cat([t1, t2], dim=0)


def _unique_edges_from_faces(faces: Sequence[torch.Tensor]) -> torch.Tensor:
    directed = torch.cat([torch.stack([f, f.roll(-1)], dim=1) for f in faces], dim=0)  # (C, 2)
    undirected, _ = directed.sort(dim=1)
    return torch.unique(undirected, dim=0)


def _uniform_laplacian_loss(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    v_count = int(verts.shape[0])
    edges = edges.to(device=verts.device, dtype=torch.long)
    v0 = edges[:, 0]
    v1 = edges[:, 1]

    sum_n = torch.zeros_like(verts)
    deg = torch.zeros((v_count, 1), device=verts.device, dtype=verts.dtype)

    ones = torch.ones((edges.shape[0], 1), device=verts.device, dtype=verts.dtype)
    sum_n.index_add_(0, v0, verts[v1])
    sum_n.index_add_(0, v1, verts[v0])
    deg.index_add_(0, v0, ones)
    deg.index_add_(0, v1, ones)

    mean_n = sum_n / deg.clamp(min=1.0)
    lap = verts - mean_n
    return (lap.square().sum(dim=1)).mean()


def _edge_lengths(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    edges = edges.to(device=verts.device, dtype=torch.long)
    e0 = verts[edges[:, 0]]
    e1 = verts[edges[:, 1]]
    return torch.linalg.norm(e0 - e1, dim=1)


def _sample_points_on_tri_mesh(
    verts: torch.Tensor,
    tris: torch.Tensor,
    n: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample points on a triangle mesh by area.

    Returns:
      points: (n, 3)
      normals: (n, 3) per-sample triangle normal
    """
    tris = tris.to(device=verts.device, dtype=torch.long)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = 0.5 * torch.linalg.norm(cross, dim=1)
    areas = areas.clamp(min=1e-12)
    probs = areas / areas.sum()

    tri_idx = torch.multinomial(probs, n, replacement=True, generator=generator)
    v0s = v0[tri_idx]
    v1s = v1[tri_idx]
    v2s = v2[tri_idx]

    u = torch.rand((n,), device=verts.device, generator=generator)
    v = torch.rand((n,), device=verts.device, generator=generator)
    su = torch.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v

    pts = w0.unsqueeze(1) * v0s + w1.unsqueeze(1) * v1s + w2.unsqueeze(1) * v2s

    nrm = torch.cross(v1s - v0s, v2s - v0s, dim=1)
    nrm = torch.nn.functional.normalize(nrm, dim=1, eps=1e-12)
    return pts, nrm


def _make_tri_sample_template(
    verts: torch.Tensor,
    tris: torch.Tensor,
    n: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a deterministic sampling template for a triangle mesh.

    The template stores:
      - tri_idx: (n,) chosen triangle indices (area-weighted for this mesh)
      - w0,w1,w2: (n,) barycentric weights

    You can re-apply this template to the same triangle connectivity with different vertex
    positions, and the sampled points will move smoothly with the verts (useful for loss
    landscape probes).
    """
    tris = tris.to(device=verts.device, dtype=torch.long)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    areas = 0.5 * torch.linalg.norm(cross, dim=1)
    areas = areas.clamp(min=1e-12)
    probs = areas / areas.sum()

    tri_idx = torch.multinomial(probs, n, replacement=True, generator=generator)

    u = torch.rand((n,), device=verts.device, generator=generator)
    v = torch.rand((n,), device=verts.device, generator=generator)
    su = torch.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v
    return tri_idx.to(dtype=torch.long), w0, w1, w2


def _apply_tri_sample_template(
    verts: torch.Tensor,
    tris: torch.Tensor,
    tri_idx: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tris = tris.to(device=verts.device, dtype=torch.long)
    tri_idx = tri_idx.to(device=verts.device, dtype=torch.long)

    v0 = verts[tris[tri_idx, 0]]
    v1 = verts[tris[tri_idx, 1]]
    v2 = verts[tris[tri_idx, 2]]

    pts = w0.unsqueeze(1) * v0 + w1.unsqueeze(1) * v1 + w2.unsqueeze(1) * v2
    nrm = torch.cross(v1 - v0, v2 - v0, dim=1)
    nrm = torch.nn.functional.normalize(nrm, dim=1, eps=1e-12)
    return pts, nrm


def _knn1_min_and_argmin(
    src: torch.Tensor,
    tgt: torch.Tensor,
    *,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each point in src, find the closest point in tgt.
    Returns (min_squared_dist, argmin_index).
    """
    if src.ndim != 2 or src.shape[1] != 3 or tgt.ndim != 2 or tgt.shape[1] != 3:
        raise ValueError("Expected src/tgt shape (N,3)/(M,3)")

    device = src.device
    dtype = src.dtype
    tgt = tgt.to(device=device, dtype=dtype)

    tgt_t = tgt.t().contiguous()  # (3, M)
    tgt_norm2 = tgt.square().sum(dim=1)  # (M,)

    mins = torch.empty((src.shape[0],), device=device, dtype=dtype)
    argmins = torch.empty((src.shape[0],), device=device, dtype=torch.long)

    for start in range(0, src.shape[0], block_size):
        end = min(start + block_size, src.shape[0])
        s = src[start:end]
        s_norm2 = s.square().sum(dim=1, keepdim=True)  # (B, 1)
        d2 = s_norm2 + tgt_norm2.unsqueeze(0) - 2.0 * (s @ tgt_t)  # (B, M)
        d2_min, idx = d2.min(dim=1)
        mins[start:end] = d2_min.clamp(min=0.0)
        argmins[start:end] = idx

    return mins, argmins


def _chamfer_and_correspondences(
    src: torch.Tensor,
    tgt: torch.Tensor,
    *,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d2_st, idx_st = _knn1_min_and_argmin(src, tgt, block_size=block_size)
    d2_ts, idx_ts = _knn1_min_and_argmin(tgt, src, block_size=block_size)
    return d2_st, idx_st, d2_ts, idx_ts


def _normalize_like_target(
    cage_verts: torch.Tensor, target_verts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    min_v = target_verts.min(dim=0).values
    max_v = target_verts.max(dim=0).values
    center = 0.5 * (min_v + max_v)
    scale = (max_v - min_v).max().clamp(min=1e-8)
    cage_n = (cage_verts - center) / scale
    target_n = (target_verts - center) / scale
    return cage_n, target_n, center, scale


def _build_graph_laplacian_matrix(
    v_count: int, tri_faces: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    tri_faces = tri_faces.to(device=device, dtype=torch.long)
    edges = torch.cat(
        [tri_faces[:, [0, 1]], tri_faces[:, [1, 2]], tri_faces[:, [2, 0]]], dim=0
    )
    edges, _ = edges.sort(dim=1)
    edges = torch.unique(edges, dim=0)

    src = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
    dst = torch.cat([edges[:, 1], edges[:, 0]], dim=0)

    neg_ones = -torch.ones((src.shape[0],), device=device, dtype=dtype)
    deg = torch.zeros((v_count,), device=device, dtype=dtype)
    deg.index_add_(0, src, torch.ones_like(src, dtype=dtype))

    diag = torch.arange(v_count, device=device, dtype=torch.long)
    idx = torch.cat([torch.stack([src, dst], dim=0), torch.stack([diag, diag], dim=0)], dim=1)
    vals = torch.cat([neg_ones, deg], dim=0)
    return torch.sparse_coo_tensor(idx, vals, (v_count, v_count)).coalesce()


def _build_parameterization_matrix(
    verts: torch.Tensor, faces: Sequence[torch.Tensor], *, lambda_: float
) -> torch.Tensor:
    tri = _triangulate_to_tensor(faces)
    L = _build_graph_laplacian_matrix(int(verts.shape[0]), tri, verts.device, verts.dtype)
    diag = torch.arange(int(verts.shape[0]), device=verts.device, dtype=torch.long)
    eye = torch.sparse_coo_tensor(
        torch.stack([diag, diag], dim=0),
        torch.ones((int(verts.shape[0]),), device=verts.device, dtype=verts.dtype),
        (int(verts.shape[0]), int(verts.shape[0])),
    ).coalesce()
    return (eye + float(lambda_) * L).coalesce()


def _select_optimizer(
    name: str,
    params: Sequence[torch.nn.Parameter],
    lr: float,
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name in ("adam_uniform", "adamuniform"):
        if not _HAS_LARGESTEPS:
            raise RuntimeError("adam_uniform requires largesteps (pip install largesteps).")
        return ls_opt.AdamUniform(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimize a Catmull-Clark cage (vertex positions) so its subdivided surface matches a target mesh."
    )
    parser.add_argument("--target", required=True, help="Target mesh OBJ path.")
    parser.add_argument("--cage", required=True, help="Initial cage OBJ path (topology is fixed).")
    parser.add_argument("--out-dir", default="out_opt", help="Output directory.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Torch device.")
    parser.add_argument("--levels", type=int, default=2, help="Final Catmull-Clark subdivision levels.")
    parser.add_argument("--iters", type=int, default=800, help="Total iterations (split across stages).")
    parser.add_argument(
        "--stages",
        default=None,
        help='Optional stage schedule like "1:200,2:600". If omitted, uses a simple warmup ramp.',
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "adamw", "sgd", "adam_uniform"],
        help="Optimizer for the cage parameters.",
    )
    parser.add_argument("--src-samples", type=int, default=12000, help="Samples on subdivided mesh per iter.")
    parser.add_argument("--tgt-samples", type=int, default=20000, help="Fixed samples on target mesh.")
    parser.add_argument("--block-size", type=int, default=2048, help="Block size for nearest neighbor search.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--normalize", action="store_true", help="Normalize both meshes to target bbox (recommended).")
    parser.add_argument("--save-every", type=int, default=100, help="Write intermediate OBJs every N iters (0=never).")
    parser.add_argument("--triangulate-out", action="store_true", help="Triangulate OBJ output faces.")

    # Loss weights.
    parser.add_argument("--w-chamfer", type=float, default=1.0, help="Weight for symmetric Chamfer (squared).")
    parser.add_argument("--w-p2plane", type=float, default=0.0, help="Weight for point-to-plane loss (src->tgt).")
    parser.add_argument("--w-lap", type=float, default=1e-3, help="Weight for cage Laplacian smoothing.")
    parser.add_argument("--w-edge", type=float, default=1e-2, help="Weight for preserving cage edge lengths.")
    parser.add_argument("--w-anchor", type=float, default=0.0, help="Weight for anchoring cage to init verts.")

    # LargeSteps-style differential parameterization (optional).
    parser.add_argument(
        "--parameterization",
        choices=["xyz", "differential"],
        default="xyz",
        help="Optimize in XYZ space or in differential coordinates (requires largesteps).",
    )
    parser.add_argument(
        "--diff-lambda",
        type=float,
        default=10.0,
        help="Differential parameterization strength lambda (only used for --parameterization differential).",
    )
    parser.add_argument(
        "--diff-solver",
        choices=["auto", "cholesky", "cg"],
        default="auto",
        help="Linear solver for differential parameterization.",
    )
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Clip grad norm (0=off).")

    # Diagnostics / visualization helpers.
    parser.add_argument(
        "--dump-landscape",
        action="store_true",
        help="Dump a 2D loss landscape slice around the optimized cage to out-dir/landscape.json.",
    )
    parser.add_argument(
        "--landscape-res",
        type=int,
        default=15,
        help="Landscape grid resolution per axis (total evals = res^2).",
    )
    parser.add_argument(
        "--landscape-span",
        type=float,
        default=1.0,
        help="Landscape span in units of mean initial cage edge length (evaluates a,b in [-span*ref, +span*ref]).",
    )
    parser.add_argument(
        "--landscape-src-samples",
        type=int,
        default=1000,
        help="Surface samples on subdivided mesh per landscape evaluation.",
    )
    parser.add_argument(
        "--landscape-tgt-samples",
        type=int,
        default=2000,
        help="Target point samples used for landscape evaluation (subsample of --tgt-samples).",
    )

    args = parser.parse_args(argv)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    # Load meshes.
    cage_verts0, cage_faces = read_obj(args.cage, device=device)
    target_verts0, target_faces = read_obj(args.target, device=device)

    if args.normalize:
        cage_verts0, target_verts0, norm_center, norm_scale = _normalize_like_target(
            cage_verts0, target_verts0
        )
    else:
        norm_center = torch.zeros((3,), device=device, dtype=cage_verts0.dtype)
        norm_scale = torch.tensor(1.0, device=device, dtype=cage_verts0.dtype)

    # Precompute target triangle soup + fixed samples.
    target_tris = _triangulate_to_tensor(target_faces).to(device=device)
    tgt_pts, tgt_nrm = _sample_points_on_tri_mesh(
        target_verts0, target_tris, int(args.tgt_samples), generator=gen
    )

    # Cage topology info / regularizers.
    cage_edges = _unique_edges_from_faces(cage_faces).to(device=device)
    cage_edge_len0 = _edge_lengths(cage_verts0, cage_edges).detach()

    stages = _parse_stages(args.stages, int(args.levels), int(args.iters))

    if args.parameterization == "differential":
        if not _HAS_LARGESTEPS:
            raise RuntimeError("--parameterization differential requires largesteps (pip install largesteps).")

        M = _build_parameterization_matrix(cage_verts0, cage_faces, lambda_=float(args.diff_lambda))
        solver_name = args.diff_solver
        if solver_name == "auto":
            solver_name = "cholesky" if device.type == "cpu" else "cg"

        if solver_name == "cholesky":
            if device.type != "cpu":
                raise RuntimeError("diff solver 'cholesky' requires CPU tensors (use --diff-solver cg on CUDA).")
            method = "Cholesky"
        else:
            method = "CG"

        u0 = ls_param.to_differential(M, cage_verts0).detach()
        u = torch.nn.Parameter(u0.clone())
        params = [u]

        def current_cage_verts() -> torch.Tensor:
            return ls_param.from_differential(M, u, method=method)

    else:
        cage_verts = torch.nn.Parameter(cage_verts0.clone())
        params = [cage_verts]

        def current_cage_verts() -> torch.Tensor:
            return cage_verts

    optimizer = _select_optimizer(args.optimizer, params, float(args.lr))

    # Save initial.
    cage_init_world = cage_verts0 * norm_scale + norm_center
    write_obj(out_dir / "cage_init.obj", cage_init_world, cage_faces, triangulate=bool(args.triangulate_out))

    csv_path = out_dir / "loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "global_iter",
                "stage_levels",
                "loss_total",
                "loss_chamfer",
                "loss_p2plane",
                "loss_lap",
                "loss_edge",
                "loss_anchor",
            ]
        )

        global_iter = 0
        for stage in stages:
            for local_iter in range(stage.iters):
                optimizer.zero_grad(set_to_none=True)

                cage_v = current_cage_verts()

                # Forward subdiv.
                subd_v, subd_quads = subdivide_n(cage_v, cage_faces, levels=int(stage.levels))
                subd_tris = _quads_to_tris_tensor(subd_quads)

                src_pts, _src_nrm = _sample_points_on_tri_mesh(
                    subd_v, subd_tris, int(args.src_samples), generator=gen
                )

                d2_st, idx_st, d2_ts, _idx_ts = _chamfer_and_correspondences(
                    src_pts, tgt_pts, block_size=int(args.block_size)
                )
                loss_chamfer = d2_st.mean() + d2_ts.mean()

                if args.w_p2plane > 0.0:
                    nn = tgt_pts[idx_st]
                    nn_n = tgt_nrm[idx_st]
                    p2 = ((src_pts - nn) * nn_n).sum(dim=1)
                    loss_p2plane = p2.square().mean()
                else:
                    loss_p2plane = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                if args.w_lap > 0.0:
                    loss_lap = _uniform_laplacian_loss(cage_v, cage_edges)
                else:
                    loss_lap = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                if args.w_edge > 0.0:
                    edge_len = _edge_lengths(cage_v, cage_edges)
                    loss_edge = (edge_len - cage_edge_len0).square().mean()
                else:
                    loss_edge = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                if args.w_anchor > 0.0:
                    loss_anchor = (cage_v - cage_verts0).square().mean()
                else:
                    loss_anchor = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                loss = (
                    float(args.w_chamfer) * loss_chamfer
                    + float(args.w_p2plane) * loss_p2plane
                    + float(args.w_lap) * loss_lap
                    + float(args.w_edge) * loss_edge
                    + float(args.w_anchor) * loss_anchor
                )

                loss.backward()
                if args.clip_grad and float(args.clip_grad) > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad))
                optimizer.step()

                if global_iter % 10 == 0:
                    loss_s = float(loss.detach().cpu())
                    chamfer_s = float(loss_chamfer.detach().cpu())
                    print(
                        f"iter {global_iter:05d}  lvl={stage.levels}  loss={loss_s:.6f}  chamfer={chamfer_s:.6f}"
                    )

                loss_s = float(loss.detach().cpu())
                loss_chamfer_s = float(loss_chamfer.detach().cpu())
                loss_p2plane_s = float(loss_p2plane.detach().cpu())
                loss_lap_s = float(loss_lap.detach().cpu())
                loss_edge_s = float(loss_edge.detach().cpu())
                loss_anchor_s = float(loss_anchor.detach().cpu())
                writer.writerow(
                    [
                        global_iter,
                        stage.levels,
                        loss_s,
                        loss_chamfer_s,
                        loss_p2plane_s,
                        loss_lap_s,
                        loss_edge_s,
                        loss_anchor_s,
                    ]
                )

                if args.save_every and int(args.save_every) > 0 and (global_iter % int(args.save_every) == 0):
                    with torch.no_grad():
                        cage_world = current_cage_verts() * norm_scale + norm_center
                        write_obj(
                            out_dir / f"cage_iter_{global_iter:05d}.obj",
                            cage_world,
                            cage_faces,
                            triangulate=bool(args.triangulate_out),
                        )
                        subd_world = subd_v * norm_scale + norm_center
                        write_obj(
                            out_dir / f"subd_iter_{global_iter:05d}.obj",
                            subd_world,
                            faces_quads_to_list(subd_quads),
                            triangulate=bool(args.triangulate_out),
                        )

                global_iter += 1

    # Save final.
    with torch.no_grad():
        cage_final = current_cage_verts()
        subd_final_v, subd_final_q = subdivide_n(cage_final, cage_faces, levels=int(args.levels))

        cage_world = cage_final * norm_scale + norm_center
        subd_world = subd_final_v * norm_scale + norm_center

        write_obj(out_dir / "cage_optimized.obj", cage_world, cage_faces, triangulate=bool(args.triangulate_out))
        write_obj(
            out_dir / "subd_final.obj",
            subd_world,
            faces_quads_to_list(subd_final_q),
            triangulate=bool(args.triangulate_out),
        )

        if args.dump_landscape:
            # Loss landscape: evaluate L(a,b) around the final cage along two random directions.
            # This is a diagnostic tool. Keep it deterministic by reusing fixed target samples and
            # a fixed source sampling template (same triangle ids + barycentric weights).
            res = int(args.landscape_res)
            if res < 3:
                raise ValueError("--landscape-res must be >= 3")

            tgt_n = min(int(args.landscape_tgt_samples), int(tgt_pts.shape[0]))
            tgt_pts_l = tgt_pts[:tgt_n].detach()
            tgt_nrm_l = tgt_nrm[:tgt_n].detach()

            subd_tris_final = _quads_to_tris_tensor(subd_final_q)

            gen_land = torch.Generator(device=device)
            gen_land.manual_seed(int(args.seed) + 4242)

            tri_idx, w0, w1, w2 = _make_tri_sample_template(
                subd_final_v.detach(),
                subd_tris_final,
                int(args.landscape_src_samples),
                generator=gen_land,
            )

            ref = float(cage_edge_len0.mean().detach().cpu())
            span = float(args.landscape_span) * ref
            a_vals = torch.linspace(-span, span, steps=res, device=device, dtype=cage_final.dtype)
            b_vals = torch.linspace(-span, span, steps=res, device=device, dtype=cage_final.dtype)

            # Random orthonormal directions in cage vertex space.
            d1 = torch.randn(
                cage_final.shape, device=device, dtype=cage_final.dtype, generator=gen_land
            )
            d1 = d1 / d1.square().sum().sqrt().clamp(min=1e-12)
            d2 = torch.randn(
                cage_final.shape, device=device, dtype=cage_final.dtype, generator=gen_land
            )
            d2 = d2 - (d1 * d2).sum() * d1
            d2 = d2 / d2.square().sum().sqrt().clamp(min=1e-12)

            def eval_loss(cage_v_eval: torch.Tensor) -> Tuple[torch.Tensor, dict]:
                subd_v_eval, subd_q_eval = subdivide_n(
                    cage_v_eval, cage_faces, levels=int(args.levels)
                )
                subd_tris_eval = _quads_to_tris_tensor(subd_q_eval)
                src_pts_eval, _ = _apply_tri_sample_template(
                    subd_v_eval, subd_tris_eval, tri_idx, w0, w1, w2
                )

                d2_st, idx_st, d2_ts, _idx_ts = _chamfer_and_correspondences(
                    src_pts_eval, tgt_pts_l, block_size=int(args.block_size)
                )
                loss_chamfer = d2_st.mean() + d2_ts.mean()

                if args.w_p2plane > 0.0:
                    nn = tgt_pts_l[idx_st]
                    nn_n = tgt_nrm_l[idx_st]
                    p2 = ((src_pts_eval - nn) * nn_n).sum(dim=1)
                    loss_p2plane = p2.square().mean()
                else:
                    loss_p2plane = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                if args.w_lap > 0.0:
                    loss_lap = _uniform_laplacian_loss(cage_v_eval, cage_edges)
                else:
                    loss_lap = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                if args.w_edge > 0.0:
                    edge_len = _edge_lengths(cage_v_eval, cage_edges)
                    loss_edge = (edge_len - cage_edge_len0).square().mean()
                else:
                    loss_edge = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                if args.w_anchor > 0.0:
                    loss_anchor = (cage_v_eval - cage_verts0).square().mean()
                else:
                    loss_anchor = torch.zeros((), device=device, dtype=loss_chamfer.dtype)

                loss_total = (
                    float(args.w_chamfer) * loss_chamfer
                    + float(args.w_p2plane) * loss_p2plane
                    + float(args.w_lap) * loss_lap
                    + float(args.w_edge) * loss_edge
                    + float(args.w_anchor) * loss_anchor
                )
                terms = {
                    "loss_chamfer": loss_chamfer,
                    "loss_p2plane": loss_p2plane,
                    "loss_lap": loss_lap,
                    "loss_edge": loss_edge,
                    "loss_anchor": loss_anchor,
                }
                return loss_total, terms

            loss_total_grid = torch.empty((res, res), device=device, dtype=cage_final.dtype)
            loss_chamfer_grid = torch.empty((res, res), device=device, dtype=cage_final.dtype)

            for jb in range(res):
                for ia in range(res):
                    cage_v_eval = cage_final + a_vals[ia] * d1 + b_vals[jb] * d2
                    lt, tt = eval_loss(cage_v_eval)
                    loss_total_grid[jb, ia] = lt
                    loss_chamfer_grid[jb, ia] = tt["loss_chamfer"]

            land = {
                "a_vals": [float(x) for x in a_vals.detach().cpu().tolist()],
                "b_vals": [float(x) for x in b_vals.detach().cpu().tolist()],
                "loss_total": loss_total_grid.detach().cpu().tolist(),
                "loss_chamfer": loss_chamfer_grid.detach().cpu().tolist(),
                "meta": {
                    "levels": int(args.levels),
                    "src_samples": int(args.landscape_src_samples),
                    "tgt_samples": int(tgt_n),
                    "span": float(span),
                    "ref_mean_edge_len": float(ref),
                    "weights": {
                        "w_chamfer": float(args.w_chamfer),
                        "w_p2plane": float(args.w_p2plane),
                        "w_lap": float(args.w_lap),
                        "w_edge": float(args.w_edge),
                        "w_anchor": float(args.w_anchor),
                    },
                },
            }
            land_path = out_dir / "landscape.json"
            land_path.write_text(json.dumps(land, indent=2), encoding="utf-8")
            print("wrote:", land_path)

    print("wrote:", csv_path)
    print("wrote:", out_dir / "cage_optimized.obj")
    print("wrote:", out_dir / "subd_final.obj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
