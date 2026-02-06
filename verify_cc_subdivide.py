from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch

from cc_subdivide import (
    catmull_clark_subdivide,
    read_obj,
    subdivide_n,
    write_obj,
    faces_quads_to_list,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_cube_quads(device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    verts = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    faces = [
        torch.tensor([0, 1, 3, 2], dtype=torch.long, device=device),  # x=-1
        torch.tensor([4, 6, 7, 5], dtype=torch.long, device=device),  # x=+1
        torch.tensor([0, 4, 5, 1], dtype=torch.long, device=device),  # y=-1
        torch.tensor([2, 3, 7, 6], dtype=torch.long, device=device),  # y=+1
        torch.tensor([0, 2, 6, 4], dtype=torch.long, device=device),  # z=-1
        torch.tensor([1, 5, 7, 3], dtype=torch.long, device=device),  # z=+1
    ]
    return verts, faces


def make_quad_grid(nx: int, ny: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    xs = torch.linspace(-1.0, 1.0, steps=nx + 1, device=device)
    ys = torch.linspace(-1.0, 1.0, steps=ny + 1, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    verts = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.zeros_like(xx).reshape(-1)], dim=1
    )

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    faces: List[torch.Tensor] = []
    for j in range(ny):
        for i in range(nx):
            faces.append(
                torch.tensor(
                    [vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)],
                    dtype=torch.long,
                    device=device,
                )
            )
    return verts, faces


def test_cube() -> None:
    device = _device()
    verts0, faces0 = make_cube_quads(device)
    v1, f1, dbg = catmull_clark_subdivide(verts0, faces0, return_debug=True)

    V0 = verts0.shape[0]
    F0 = len(faces0)
    E0 = dbg["unique_edges"].shape[0]

    assert v1.shape[0] == V0 + F0 + E0
    assert f1.ndim == 2 and f1.shape[1] == 4
    assert f1.shape[0] == sum(int(f.numel()) for f in faces0)

    face_points = dbg["face_points"]
    assert torch.allclose(face_points[0], torch.tensor([-1.0, 0.0, 0.0], device=device), atol=1e-6)

    unique_edges = dbg["unique_edges"]
    edge_points = dbg["edge_points"]
    edge01_sorted = torch.tensor([0, 1], device=device)
    matches = (unique_edges == edge01_sorted).all(dim=1)
    assert int(matches.sum().item()) == 1
    edge01_id = int(torch.nonzero(matches, as_tuple=False)[0].item())
    assert torch.allclose(edge_points[edge01_id], torch.tensor([-0.75, -0.75, 0.0], device=device), atol=1e-6)

    expected_v0 = torch.tensor([-5.0 / 9.0, -5.0 / 9.0, -5.0 / 9.0], device=device)
    assert torch.allclose(dbg["verts_new"][0], expected_v0, atol=1e-6)

    offset = torch.tensor([0.3, -0.2, 1.7], device=device)
    v1_shift, f1_shift = catmull_clark_subdivide(verts0 + offset, faces0)
    assert torch.allclose(v1_shift, v1 + offset, atol=1e-6)
    assert torch.equal(f1_shift, f1)


def test_boundary_single_quad() -> None:
    device = _device()
    verts_q, faces_q = make_quad_grid(nx=1, ny=1, device=device)
    v1, f1, dbg = catmull_clark_subdivide(verts_q, faces_q, return_debug=True)

    assert f1.shape[1] == 4
    assert int((dbg["edge_face_count"] == 1).sum().item()) == int(dbg["unique_edges"].shape[0])

    ue = dbg["unique_edges"]
    ep = dbg["edge_points"]
    e01_sorted = torch.tensor([0, 1], device=device)
    idx01 = int(torch.nonzero((ue == e01_sorted).all(dim=1), as_tuple=False)[0].item())
    mid01 = 0.5 * (verts_q[0] + verts_q[1])
    assert torch.allclose(ep[idx01], mid01, atol=1e-6)

    v0_expected = (6.0 * verts_q[0] + verts_q[1] + verts_q[2]) / 8.0
    assert torch.allclose(dbg["verts_new"][0], v0_expected, atol=1e-6)


def test_autograd() -> None:
    device = _device()
    verts0, faces0 = make_cube_quads(device)
    verts0 = verts0.clone().requires_grad_(True)
    v1, _ = catmull_clark_subdivide(verts0, faces0)
    loss = (v1 ** 2).sum()
    loss.backward()
    assert verts0.grad is not None
    assert torch.isfinite(verts0.grad).all()


def test_obj_roundtrip() -> None:
    device = _device()
    out_dir = Path("out_verify")
    out_dir.mkdir(exist_ok=True)

    verts_t = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        device=device,
    )
    faces_t = [
        torch.tensor([0, 1, 2], dtype=torch.long, device=device),
        torch.tensor([0, 2, 3], dtype=torch.long, device=device),
    ]

    write_obj(out_dir / "tri_square.obj", verts_t, faces_t, triangulate=False)
    verts_l, faces_l = read_obj(out_dir / "tri_square.obj", device=device)
    assert verts_l.shape[0] == 4 and len(faces_l) == 2, (tuple(verts_l.shape), len(faces_l))

    v1, f1 = catmull_clark_subdivide(verts_l, faces_l)
    assert f1.shape[1] == 4
    assert f1.shape[0] == sum(int(f.numel()) for f in faces_l)

    write_obj(out_dir / "tri_square_subd.obj", v1, faces_quads_to_list(f1), triangulate=False)


def main() -> None:
    test_cube()
    test_boundary_single_quad()
    test_autograd()
    test_obj_roundtrip()
    print("All tests passed.")


if __name__ == "__main__":
    main()

