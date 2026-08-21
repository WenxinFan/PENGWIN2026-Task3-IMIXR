"""Shared mesh-fragment sampling for clinical training and inference."""

from typing import Sequence

import numpy as np
import trimesh


def allocate_equal(total_points: int, n_fragments: int, min_points: int = 20) -> list[int]:
    """Allocate exactly ``total_points`` as evenly as possible across fragments."""
    if total_points <= 0:
        raise ValueError(f"total_points must be positive, got {total_points}")
    if n_fragments <= 0:
        return []
    if total_points < n_fragments * min_points:
        raise ValueError(
            f"{total_points} points cannot provide min_points={min_points} "
            f"to {n_fragments} fragments"
        )

    base, remainder = divmod(total_points, n_fragments)
    return [base + (i < remainder) for i in range(n_fragments)]


def allocate_legacy_equal(
    total_points: int,
    n_fragments: int,
    min_points: int = 20,
) -> list[int]:
    """Reproduce the original clinical loader's floor-division allocation."""
    if total_points <= 0:
        raise ValueError(f"total_points must be positive, got {total_points}")
    if n_fragments <= 0:
        return []
    points_per_fragment = max(total_points // n_fragments, min_points)
    return [points_per_fragment] * n_fragments


def _as_generator(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    seed = int(rng.randint(0, np.iinfo(np.uint32).max))
    return np.random.default_rng(seed)


def _remove_close_without_scipy(
    points: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match trimesh remove_close using a deterministic spatial hash."""
    if len(points) == 0 or radius <= 0:
        return points, np.ones(len(points), dtype=bool)

    cells: dict[tuple[int, int, int], list[int]] = {}
    pairs = []
    inv_radius = 1.0 / radius
    radius_sq = radius * radius

    for idx, point in enumerate(points):
        cell = tuple(np.floor(point * inv_radius).astype(np.int64))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for prev_idx in cells.get(neighbor, ()):
                        delta = point - points[prev_idx]
                        if float(delta @ delta) < radius_sq:
                            pairs.append((prev_idx, idx))
        cells.setdefault(cell, []).append(idx)

    if not pairs:
        return points, np.ones(len(points), dtype=bool)

    pairs_array = np.asarray(sorted(pairs), dtype=np.int64)
    counts = np.bincount(pairs_array.ravel(), minlength=len(points))
    column = counts[pairs_array].argmax(axis=1)
    highest = pairs_array.ravel()[column + 2 * np.arange(len(column))]
    mask = np.ones(len(points), dtype=bool)
    mask[highest] = False
    return points[mask], mask


def _sample_surface_even_without_scipy(
    mesh: trimesh.Trimesh,
    n_points: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sqrt(mesh.area / (3 * n_points))
    points, face_indices = trimesh.sample.sample_surface(
        mesh, n_points * 3, seed=generator
    )
    points, mask = _remove_close_without_scipy(points, radius)
    face_indices = face_indices[mask]
    if len(points) >= n_points:
        return points[:n_points], face_indices[:n_points]
    return points, face_indices


def allocate_by_volume(
    meshes: Sequence[trimesh.Trimesh],
    total_points: int,
    min_points: int = 20,
) -> list[int]:
    """Allocate points in proportion to mesh volume, preserving every fragment."""
    if total_points <= 0:
        raise ValueError(f"total_points must be positive, got {total_points}")
    if not meshes:
        return []
    if total_points < len(meshes) * min_points:
        raise ValueError(
            f"{total_points} points cannot provide min_points={min_points} "
            f"to {len(meshes)} fragments"
        )

    volumes = np.asarray(
        [mesh.volume if hasattr(mesh, "volume") and mesh.volume > 0 else 0.001 for mesh in meshes],
        dtype=np.float64,
    )
    volume_sum = float(volumes.sum())
    if not np.isfinite(volume_sum) or volume_sum <= 0:
        volumes = np.ones(len(meshes), dtype=np.float64)
        volume_sum = float(volumes.sum())

    allocations = [
        max(int(volume / volume_sum * total_points), min_points)
        for volume in volumes
    ]
    diff = total_points - sum(allocations)
    if diff != 0:
        max_idx = int(np.argmax(volumes))
        allocations[max_idx] = max(allocations[max_idx] + diff, min_points)
    return allocations


def allocate_small_fragment_boost(
    meshes: Sequence[trimesh.Trimesh],
    total_points: int,
    min_points: int = 20,
    boost_factor: float = 1.8,
    area_percentile: float = 30.0,
) -> list[int]:
    """Allocate more training samples to small fragments within a fixed budget."""
    if total_points <= 0:
        raise ValueError(f"total_points must be positive, got {total_points}")
    if not meshes:
        return []
    if total_points < len(meshes) * min_points:
        raise ValueError(
            f"{total_points} points cannot provide min_points={min_points} "
            f"to {len(meshes)} fragments"
        )

    boost_factor = max(float(boost_factor), 1.0)
    areas = np.asarray([max(float(mesh.area), 0.0) for mesh in meshes], dtype=np.float64)
    finite_positive = areas[np.isfinite(areas) & (areas > 0)]
    weights = np.ones(len(meshes), dtype=np.float64)

    if len(meshes) > 1 and len(finite_positive) > 0 and boost_factor > 1.0:
        percentile = float(np.clip(area_percentile, 0.0, 100.0))
        threshold = float(np.percentile(finite_positive, percentile))
        small_mask = areas <= threshold
        small_mask |= ~np.isfinite(areas) | (areas <= 0)
        weights[small_mask] *= boost_factor

    base = np.full(len(meshes), min_points, dtype=np.int64)
    remaining = total_points - int(base.sum())
    if remaining <= 0:
        return base.tolist()

    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        weights = np.ones(len(meshes), dtype=np.float64)
        weights_sum = float(weights.sum())

    raw_extra = weights / weights_sum * remaining
    extra = np.floor(raw_extra).astype(np.int64)
    diff = remaining - int(extra.sum())
    if diff > 0:
        order = np.argsort(-(raw_extra - extra))
        extra[order[:diff]] += 1

    return (base + extra).astype(int).tolist()


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    n_points: int,
    rng,
    surface_sample: str = "even",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an exact number of surface points using an explicit random stream."""
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # The historical clinical route called trimesh directly, without passing a
    # generator or padding sample_surface_even back to the requested count.
    if surface_sample == "legacy_even" and len(mesh.faces) > 0:
        points, face_indices = trimesh.sample.sample_surface_even(mesh, n_points)
        normals = mesh.face_normals[face_indices]
        return points.astype(np.float32), normals.astype(np.float32)

    generator = _as_generator(rng)

    if len(mesh.faces) == 0:
        if len(mesh.vertices) == 0:
            return (
                np.zeros((n_points, 3), dtype=np.float32),
                np.zeros((n_points, 3), dtype=np.float32),
            )
        indices = generator.integers(0, len(mesh.vertices), size=n_points)
        return (
            np.asarray(mesh.vertices[indices], dtype=np.float32),
            np.zeros((n_points, 3), dtype=np.float32),
        )

    if surface_sample == "random":
        points, face_indices = trimesh.sample.sample_surface(
            mesh, n_points, seed=generator
        )
    elif surface_sample == "even_noscipy":
        points, face_indices = _sample_surface_even_without_scipy(
            mesh, n_points, generator
        )
    else:
        points, face_indices = trimesh.sample.sample_surface_even(
            mesh, n_points, seed=generator
        )
    normals = mesh.face_normals[face_indices]

    current = len(points)
    if current < n_points:
        if current == 0:
            points, face_indices = trimesh.sample.sample_surface(
                mesh, n_points, seed=generator
            )
            normals = mesh.face_normals[face_indices]
        else:
            extra = generator.integers(0, current, size=n_points - current)
            points = np.vstack([points, points[extra]])
            normals = np.vstack([normals, normals[extra]])
    elif current > n_points:
        points = points[:n_points]
        normals = normals[:n_points]

    return points.astype(np.float32), normals.astype(np.float32)


def sample_fragments_equal(
    meshes: Sequence[trimesh.Trimesh],
    total_points: int,
    rng,
    min_points: int = 20,
    allocation: str = "equal",
    surface_sample: str = "even",
    small_fragment_boost_factor: float = 1.0,
    small_fragment_area_percentile: float = 30.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample fragments while preserving the exact per-bone point budget."""
    if small_fragment_boost_factor > 1.0 and allocation in {"equal", "small_boost"}:
        allocations = allocate_small_fragment_boost(
            meshes,
            total_points,
            min_points=min_points,
            boost_factor=small_fragment_boost_factor,
            area_percentile=small_fragment_area_percentile,
        )
    elif allocation == "volume":
        allocations = allocate_by_volume(meshes, total_points, min_points=min_points)
    elif allocation == "small_boost":
        allocations = allocate_small_fragment_boost(
            meshes,
            total_points,
            min_points=min_points,
            boost_factor=small_fragment_boost_factor,
            area_percentile=small_fragment_area_percentile,
        )
    elif allocation == "legacy_equal":
        allocations = allocate_legacy_equal(
            total_points, len(meshes), min_points=min_points
        )
    else:
        allocations = allocate_equal(total_points, len(meshes), min_points=min_points)
    return [
        sample_mesh_surface(mesh, count, rng, surface_sample=surface_sample)
        for mesh, count in zip(meshes, allocations)
    ]
