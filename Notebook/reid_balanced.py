"""Balanced helpers for estimated multi-camera identity association.

The active notebook keeps retail analytics camera-local.  These helpers only
support the review overlay and the estimated ``global_track_id`` mapping.
They are deliberately deterministic and side-effect free so their behavior can
be validated without loading YOLO, OSNet weights, or any video.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def color_for_identity(identity: object) -> tuple[int, int, int]:
    """Return a deterministic BGR color shared by every view of an identity."""

    seed = sum(
        (index + 1) * ord(character)
        for index, character in enumerate(str(identity))
    )
    return (
        80 + seed % 150,
        80 + (seed // 7) % 150,
        80 + (seed // 17) % 150,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronized_cosine_distance(
    left_times: Iterable[float],
    left_vectors: np.ndarray,
    right_times: Iterable[float],
    right_vectors: np.ndarray,
    *,
    tolerance_sec: float,
    min_samples: int,
    max_samples: int,
) -> tuple[float, int]:
    """Compare one-to-one appearance samples that were observed at nearby times.

    Samples are greedily paired by the smallest time difference.  Every sample
    can be used once, preventing a long tracklet from inflating the evidence by
    repeatedly matching the same crop from the other camera.
    """

    left_times_array = np.asarray(list(left_times), dtype=float)
    right_times_array = np.asarray(list(right_times), dtype=float)
    left_matrix = np.asarray(left_vectors, dtype=float)
    right_matrix = np.asarray(right_vectors, dtype=float)

    if (
        left_matrix.ndim != 2
        or right_matrix.ndim != 2
        or len(left_times_array) != len(left_matrix)
        or len(right_times_array) != len(right_matrix)
        or left_matrix.shape[1:] != right_matrix.shape[1:]
    ):
        raise ValueError("Embedding times and matrices must have compatible shapes.")

    possible_pairs = []
    for left_index, left_time in enumerate(left_times_array):
        for right_index, right_time in enumerate(right_times_array):
            delta = abs(float(left_time) - float(right_time))
            if delta <= tolerance_sec:
                possible_pairs.append((delta, left_index, right_index))

    used_left: set[int] = set()
    used_right: set[int] = set()
    aligned_pairs: list[tuple[int, int]] = []
    for _, left_index, right_index in sorted(possible_pairs):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        aligned_pairs.append((left_index, right_index))

    if len(aligned_pairs) < min_samples:
        return float("nan"), len(aligned_pairs)
    if len(aligned_pairs) > max_samples:
        positions = np.linspace(0, len(aligned_pairs) - 1, max_samples, dtype=int)
        aligned_pairs = [aligned_pairs[int(position)] for position in positions]

    distances = []
    for left_index, right_index in aligned_pairs:
        left = left_matrix[left_index]
        right = right_matrix[right_index]
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm <= 0 or right_norm <= 0:
            continue
        similarity = float(np.dot(left, right) / (left_norm * right_norm))
        distances.append(1.0 - float(np.clip(similarity, -1.0, 1.0)))

    if len(distances) < min_samples:
        return float("nan"), len(distances)
    return float(np.median(distances)), len(distances)


def select_mutual_best_matches(
    candidates: pd.DataFrame,
    *,
    max_score: float,
    min_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select cross-camera matches that are mutual best and unambiguous."""

    output_columns = [*candidates.columns, "match_margin"]
    rejection_columns = [*output_columns, "rejection_reason"]
    if candidates.empty:
        return (
            pd.DataFrame(columns=output_columns),
            pd.DataFrame(columns=rejection_columns),
        )

    accepted_rows = []
    rejected_rows = []
    group_columns = ["store_id", "camera_id_a", "camera_id_b"]
    for _, group in candidates.loc[candidates["score"] <= max_score].groupby(
        group_columns, sort=True
    ):
        group = group.sort_values(
            ["score", "tracklet_id_a", "tracklet_id_b"], kind="stable"
        )
        left_rankings = {
            tracklet_id: rows.sort_values("score", kind="stable")
            for tracklet_id, rows in group.groupby("tracklet_id_a", sort=False)
        }
        right_rankings = {
            tracklet_id: rows.sort_values("score", kind="stable")
            for tracklet_id, rows in group.groupby("tracklet_id_b", sort=False)
        }

        for row_index, row in group.iterrows():
            left_rows = left_rankings[row["tracklet_id_a"]]
            right_rows = right_rankings[row["tracklet_id_b"]]
            is_mutual = row_index == left_rows.index[0] and row_index == right_rows.index[0]
            left_margin = (
                float(left_rows.iloc[1]["score"] - row["score"])
                if len(left_rows) > 1
                else float(max_score - row["score"])
            )
            right_margin = (
                float(right_rows.iloc[1]["score"] - row["score"])
                if len(right_rows) > 1
                else float(max_score - row["score"])
            )
            match_margin = min(left_margin, right_margin)
            payload = {**row.to_dict(), "match_margin": match_margin}
            if not is_mutual:
                rejected_rows.append({**payload, "rejection_reason": "not_mutual_best"})
            elif match_margin < min_margin:
                rejected_rows.append({**payload, "rejection_reason": "ambiguous_margin"})
            else:
                accepted_rows.append(payload)

    accepted = pd.DataFrame(accepted_rows, columns=output_columns)
    rejected = pd.DataFrame(rejected_rows, columns=rejection_columns)
    return accepted, rejected


def _pair_key(camera_id_a: object, camera_id_b: object) -> frozenset[str]:
    """Return an order-independent camera-pair key."""

    return frozenset((str(camera_id_a), str(camera_id_b)))


def select_anchor_staged_matches(
    candidates: pd.DataFrame,
    *,
    stage_rules: Iterable[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply mutual-best matching separately for an explicit camera topology.

    A later stage cannot compete with, or replace, a prior stage.  This is the
    central guard against a weak Cam19/Cam20 association changing an identity
    that was already established by the trusted Cam17/Cam18 anchor pair.
    """

    base_columns = [*candidates.columns, "association_stage", "evidence_camera_pair"]
    accepted_columns = [*base_columns, "match_margin"]
    rejected_columns = [*accepted_columns, "rejection_reason"]
    accepted_frames: list[pd.DataFrame] = []
    rejected_frames: list[pd.DataFrame] = []

    for rule in stage_rules:
        stage_name = str(rule["name"])
        parent_camera = str(rule["parent_camera"])
        target_camera = str(rule["target_camera"])
        pair = _pair_key(parent_camera, target_camera)
        pair_label = f"{parent_camera}<->{target_camera}"
        stage_candidates = candidates.loc[
            candidates.apply(
                lambda row: _pair_key(row["camera_id_a"], row["camera_id_b"]) == pair,
                axis=1,
            )
        ].copy()
        if stage_candidates.empty:
            continue

        stage_candidates["association_stage"] = stage_name
        stage_candidates["evidence_camera_pair"] = pair_label
        min_samples = int(rule.get("min_synchronized_samples", 0))
        if "synchronized_samples" in stage_candidates:
            evidence = pd.to_numeric(
                stage_candidates["synchronized_samples"], errors="coerce"
            ).fillna(0)
        else:
            evidence = pd.Series(0, index=stage_candidates.index, dtype=float)
        insufficient = stage_candidates.loc[evidence < min_samples].copy()
        if not insufficient.empty:
            insufficient["match_margin"] = np.nan
            insufficient["rejection_reason"] = "insufficient_direct_evidence"
            rejected_frames.append(insufficient[rejected_columns])

        eligible = stage_candidates.loc[evidence >= min_samples].copy()
        if eligible.empty:
            continue
        selected, rejected = select_mutual_best_matches(
            eligible,
            max_score=float(rule["max_score"]),
            min_margin=float(rule["min_margin"]),
        )
        if not selected.empty:
            accepted_frames.append(selected[accepted_columns])
        if not rejected.empty:
            rejected_frames.append(rejected[rejected_columns])

    accepted = (
        pd.concat(accepted_frames, ignore_index=True, sort=False)
        if accepted_frames
        else pd.DataFrame(columns=accepted_columns)
    )
    rejected = (
        pd.concat(rejected_frames, ignore_index=True, sort=False)
        if rejected_frames
        else pd.DataFrame(columns=rejected_columns)
    )
    return accepted, rejected


def cluster_anchor_stages(
    tracklets: pd.DataFrame,
    *,
    local_matches: pd.DataFrame,
    staged_cross_matches: pd.DataFrame,
    stage_rules: Iterable[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build global IDs without allowing later camera stages to rewire anchors.

    Local fragments are merged first.  Each cross-camera stage then attaches a
    previously unassociated target component to a component containing its
    configured parent camera.  The target cannot already contain another
    camera, and a merged identity can never hold simultaneous tracklets from
    the same camera.
    """

    records = {str(row["tracklet_id"]): row for row in tracklets.to_dict("records")}
    parent = {tracklet_id: tracklet_id for tracklet_id in records}

    def find(tracklet_id: str) -> str:
        while parent[tracklet_id] != tracklet_id:
            parent[tracklet_id] = parent[parent[tracklet_id]]
            tracklet_id = parent[tracklet_id]
        return tracklet_id

    def members(tracklet_id: str) -> list[str]:
        root = find(tracklet_id)
        return [candidate for candidate in parent if find(candidate) == root]

    def member_cameras(member_ids: Iterable[str]) -> set[str]:
        return {str(records[member_id]["camera_id"]) for member_id in member_ids}

    def has_overlap(left_members: list[str], right_members: list[str]) -> bool:
        for left_id in left_members:
            for right_id in right_members:
                left, right = records[left_id], records[right_id]
                if str(left["camera_id"]) != str(right["camera_id"]):
                    continue
                overlap = not (
                    float(left["end_sec"]) < float(right["start_sec"])
                    or float(right["end_sec"]) < float(left["start_sec"])
                )
                if overlap:
                    return True
        return False

    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []

    for row in local_matches.to_dict("records"):
        left_id, right_id = str(row["tracklet_id_a"]), str(row["tracklet_id_b"])
        if left_id not in records or right_id not in records:
            raise ValueError("A local match references an unknown tracklet.")
        left_root, right_root = find(left_id), find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root
        accepted_rows.append(
            {
                **row,
                "association_stage": "local_stitch",
                "evidence_camera_pair": str(records[left_id]["camera_id"]),
            }
        )

    rules_by_name = {str(rule["name"]): rule for rule in stage_rules}
    for stage_name, rule in rules_by_name.items():
        parent_camera = str(rule["parent_camera"])
        target_camera = str(rule["target_camera"])
        rows = staged_cross_matches.loc[
            staged_cross_matches["association_stage"].astype(str) == stage_name
        ].sort_values(["score", "tracklet_id_a", "tracklet_id_b"], kind="stable")
        for row in rows.to_dict("records"):
            left_id, right_id = str(row["tracklet_id_a"]), str(row["tracklet_id_b"])
            left_members, right_members = members(left_id), members(right_id)
            if set(left_members) == set(right_members):
                continue
            left_cameras, right_cameras = member_cameras(left_members), member_cameras(right_members)
            if parent_camera in left_cameras and target_camera in right_cameras:
                parent_members, target_members = left_members, right_members
            elif parent_camera in right_cameras and target_camera in left_cameras:
                parent_members, target_members = right_members, left_members
            else:
                rejected_rows.append({**row, "rejection_reason": "stage_parent_missing"})
                continue
            if member_cameras(target_members) != {target_camera}:
                rejected_rows.append({**row, "rejection_reason": "target_already_associated"})
                continue
            if has_overlap(parent_members, target_members):
                rejected_rows.append({**row, "rejection_reason": "overlapping_same_camera_tracklets"})
                continue
            parent[find(target_members[0])] = find(parent_members[0])
            accepted_rows.append(row)

    groups: dict[str, list[str]] = {}
    for tracklet_id in parent:
        groups.setdefault(find(tracklet_id), []).append(tracklet_id)
    global_ids: dict[str, str] = {}
    camera_counts: dict[str, int] = {}
    for index, member_ids in enumerate(
        sorted(groups.values(), key=lambda values: min(values)), start=1
    ):
        global_id = f"global_{index:06d}"
        camera_counts[global_id] = len(member_cameras(member_ids))
        for tracklet_id in member_ids:
            global_ids[tracklet_id] = global_id

    mapping = tracklets[["store_id", "camera_id", "local_track_id", "tracklet_id"]].copy()
    mapping["global_track_id"] = mapping["tracklet_id"].map(global_ids)
    mapping["is_cross_camera_identity"] = mapping["global_track_id"].map(
        lambda global_id: camera_counts[global_id] > 1
    )
    return mapping, pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows)


def select_disjoint_local_stitches(
    candidates: pd.DataFrame,
    *,
    max_score: float,
) -> pd.DataFrame:
    """Greedily select non-overlapping same-camera fragment stitches."""

    output_columns = [*candidates.columns, "match_margin"]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)
    selected = []
    used_as_predecessor: set[str] = set()
    used_as_successor: set[str] = set()
    for _, row in candidates.loc[candidates["score"] <= max_score].sort_values(
        ["score", "tracklet_id_a", "tracklet_id_b"], kind="stable"
    ).iterrows():
        left_id = str(row["tracklet_id_a"])
        right_id = str(row["tracklet_id_b"])
        if left_id in used_as_predecessor or right_id in used_as_successor:
            continue
        selected.append({**row.to_dict(), "match_margin": float("inf")})
        used_as_predecessor.add(left_id)
        used_as_successor.add(right_id)
    return pd.DataFrame(selected, columns=output_columns)


def build_local_components(
    tracklet_ids: Iterable[str], local_matches: pd.DataFrame
) -> dict[str, str]:
    """Return a deterministic representative for each same-camera component."""

    parent = {str(tracklet_id): str(tracklet_id) for tracklet_id in tracklet_ids}

    def find(tracklet_id: str) -> str:
        while parent[tracklet_id] != tracklet_id:
            parent[tracklet_id] = parent[parent[tracklet_id]]
            tracklet_id = parent[tracklet_id]
        return tracklet_id

    for row in local_matches.to_dict("records"):
        left_id = str(row["tracklet_id_a"])
        right_id = str(row["tracklet_id_b"])
        if left_id not in parent or right_id not in parent:
            raise ValueError("A local stitch references an unknown tracklet.")
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            parent[second] = first

    groups: dict[str, list[str]] = {}
    for tracklet_id in parent:
        groups.setdefault(find(tracklet_id), []).append(tracklet_id)
    component_map = {}
    for members in groups.values():
        representative = min(members)
        for tracklet_id in members:
            component_map[tracklet_id] = representative
    return component_map


def aggregate_local_components(
    tracklets: pd.DataFrame,
    component_by_tracklet: dict[str, str],
    embeddings: dict[str, np.ndarray],
    embedding_samples: dict[str, dict[str, np.ndarray]],
    floor_points: dict[str, pd.DataFrame],
    pixel_points: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    """Aggregate stitched local fragments before cross-camera association."""

    work = tracklets.copy()
    work["component_id"] = work["tracklet_id"].map(component_by_tracklet)
    if work["component_id"].isna().any():
        raise ValueError("Every tracklet must belong to one local component.")

    records = []
    component_samples: dict[str, dict[str, np.ndarray]] = {}
    component_floor_points: dict[str, pd.DataFrame] = {}
    component_pixel_points: dict[str, pd.DataFrame] = {}
    for component_id, group in work.groupby("component_id", sort=True):
        member_ids = sorted(group["tracklet_id"].astype(str).tolist())
        camera_ids = group["camera_id"].astype(str).unique().tolist()
        store_ids = group["store_id"].astype(str).unique().tolist()
        employee_values = group["is_employee"].astype(bool).unique().tolist()
        if len(camera_ids) != 1 or len(store_ids) != 1 or len(employee_values) != 1:
            raise ValueError("A local component crossed a camera, store, or role boundary.")

        available_embeddings = [np.asarray(embeddings[member]) for member in member_ids if member in embeddings]
        if not available_embeddings:
            continue
        mean_embedding = np.mean(np.stack(available_embeddings), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm <= 0:
            continue

        sample_times = []
        sample_vectors = []
        for member in member_ids:
            sample = embedding_samples.get(member)
            if sample is None:
                continue
            sample_times.extend(np.asarray(sample["times"], dtype=float).tolist())
            sample_vectors.extend(np.asarray(sample["vectors"], dtype=float))
        if not sample_vectors:
            continue
        order = np.argsort(np.asarray(sample_times, dtype=float), kind="stable")
        component_samples[component_id] = {
            "times": np.asarray(sample_times, dtype=float)[order],
            "vectors": np.asarray(sample_vectors, dtype=float)[order],
        }
        component_floor_points[component_id] = pd.concat(
            [floor_points[member] for member in member_ids], ignore_index=True
        ).sort_values("time_sec", kind="stable").reset_index(drop=True)
        component_pixel_points[component_id] = pd.concat(
            [pixel_points[member] for member in member_ids], ignore_index=True
        ).sort_values("time_sec", kind="stable").reset_index(drop=True)

        representative_row = group.loc[group["tracklet_id"] == component_id]
        representative_row = representative_row.iloc[0] if not representative_row.empty else group.iloc[0]
        records.append(
            {
                "store_id": store_ids[0],
                "camera_id": camera_ids[0],
                "local_track_id": int(representative_row["local_track_id"]),
                "tracklet_id": component_id,
                "start_sec": float(group["start_sec"].min()),
                "end_sec": float(group["end_sec"].max()),
                "is_employee": employee_values[0],
                "embedding": mean_embedding / norm,
                "member_tracklets": tuple(member_ids),
            }
        )

    return (
        pd.DataFrame(records),
        component_samples,
        component_floor_points,
        component_pixel_points,
    )


def annotate_mapping_quality(
    mapping: pd.DataFrame,
    accepted_matches: pd.DataFrame,
    rejected_matches: pd.DataFrame,
    *,
    target_camera_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Attach review-oriented confidence evidence to every mapping row."""

    output = mapping.copy()
    tracklet_to_global = dict(zip(output["tracklet_id"], output["global_track_id"]))
    ambiguous_tracklets: set[str] = set()
    if not rejected_matches.empty and "rejection_reason" in rejected_matches:
        ambiguous = rejected_matches.loc[
            rejected_matches["rejection_reason"].isin(
                {"ambiguous_margin", "not_mutual_best"}
            )
        ]
        ambiguous_tracklets.update(ambiguous.get("tracklet_id_a", pd.Series(dtype=str)).astype(str))
        ambiguous_tracklets.update(ambiguous.get("tracklet_id_b", pd.Series(dtype=str)).astype(str))

    group_evidence: dict[str, list[int]] = {}
    group_margins: dict[str, list[float]] = {}
    group_stages: dict[str, list[str]] = {}
    group_pairs: dict[str, list[str]] = {}
    group_appearance: dict[str, list[float]] = {}
    group_spatial: dict[str, list[float]] = {}
    locally_stitched_groups: set[str] = set()
    if not accepted_matches.empty:
        for row in accepted_matches.to_dict("records"):
            global_id = tracklet_to_global.get(row["tracklet_id_a"])
            if global_id is None:
                continue
            if bool(row.get("is_cross_camera", False)):
                group_evidence.setdefault(global_id, []).append(
                    int(row.get("synchronized_samples", 0))
                )
                stage = str(row.get("association_stage", "cross_camera"))
                pair = str(row.get("evidence_camera_pair", ""))
                group_stages.setdefault(global_id, []).append(stage)
                if pair:
                    group_pairs.setdefault(global_id, []).append(pair)
                appearance = float(row.get("appearance_distance", float("nan")))
                spatial = float(row.get("floor_distance", float("nan")))
                if np.isfinite(appearance):
                    group_appearance.setdefault(global_id, []).append(appearance)
                if np.isfinite(spatial):
                    group_spatial.setdefault(global_id, []).append(spatial)
            else:
                locally_stitched_groups.add(global_id)
            margin = float(row.get("match_margin", float("nan")))
            if np.isfinite(margin):
                group_margins.setdefault(global_id, []).append(margin)

    camera_counts = output.groupby("global_track_id")["camera_id"].nunique().to_dict()
    statuses = []
    evidence = []
    margins = []
    stages = []
    pairs = []
    appearance_distances = []
    spatial_distances = []
    direct_evidence_counts = []
    target_cameras = {str(camera_id) for camera_id in target_camera_ids}
    for row in output.to_dict("records"):
        global_id = row["global_track_id"]
        if camera_counts.get(global_id, 0) > 1:
            status = "matched_cross_camera"
        elif global_id in locally_stitched_groups:
            status = "stitched_camera_local"
        elif str(row["tracklet_id"]) in ambiguous_tracklets:
            status = "ambiguous_rejected"
        elif str(row["camera_id"]) in target_cameras:
            status = "unmatched_target"
        else:
            status = "singleton"
        statuses.append(status)
        evidence.append(max(group_evidence.get(global_id, [0])))
        margins.append(min(group_margins.get(global_id, [float("nan")])))
        stages.append(
            ",".join(dict.fromkeys(group_stages.get(global_id, []))) or "none"
        )
        pairs.append(
            ",".join(dict.fromkeys(group_pairs.get(global_id, []))) or ""
        )
        appearance_values = group_appearance.get(global_id, [])
        spatial_values = group_spatial.get(global_id, [])
        appearance_distances.append(
            float(np.median(appearance_values)) if appearance_values else float("nan")
        )
        spatial_distances.append(
            float(np.median(spatial_values)) if spatial_values else float("nan")
        )
        direct_evidence_counts.append(sum(group_evidence.get(global_id, [])))

    output["identity_status"] = statuses
    output["evidence_samples"] = evidence
    output["match_margin"] = margins
    output["association_stage"] = stages
    output["evidence_camera_pair"] = pairs
    output["appearance_distance"] = appearance_distances
    output["spatial_distance"] = spatial_distances
    output["direct_evidence_count"] = direct_evidence_counts
    return output


def interpolate_display_rows(
    camera_rows: pd.DataFrame,
    *,
    max_frame_gap: int = 2,
) -> pd.DataFrame:
    """Fill only the single display frame skipped by ``FRAME_STRIDE = 2``.

    No leading/trailing hold is used and no gap larger than ``max_frame_gap``
    is bridged, so this cannot create a long-lived ghost identity.
    """

    if camera_rows.empty:
        output = camera_rows.copy()
        output["display_interpolated"] = pd.Series(dtype=bool)
        return output
    required = {
        "camera_id",
        "local_track_id",
        "frame_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "global_track_id",
    }
    missing = sorted(required.difference(camera_rows.columns))
    if missing:
        raise ValueError(f"Display interpolation is missing columns: {missing}")

    originals = camera_rows.copy()
    originals["display_interpolated"] = False
    generated = []
    group_columns = ["camera_id", "local_track_id"]
    numeric_columns = [
        column
        for column in ("timestamp_sec", "x1", "y1", "x2", "y2", "foot_x", "foot_y")
        if column in originals.columns
    ]
    for _, group in originals.sort_values("frame_index").groupby(group_columns, sort=False):
        records = group.to_dict("records")
        for left, right in zip(records, records[1:]):
            left_frame = int(left["frame_index"])
            right_frame = int(right["frame_index"])
            gap = right_frame - left_frame
            if gap != max_frame_gap:
                continue
            for frame_index in range(left_frame + 1, right_frame):
                ratio = (frame_index - left_frame) / gap
                row = dict(left)
                row["frame_index"] = frame_index
                row["display_interpolated"] = True
                for column in numeric_columns:
                    row[column] = float(left[column]) + ratio * (
                        float(right[column]) - float(left[column])
                    )
                generated.append(row)

    if generated:
        originals = pd.concat(
            [originals, pd.DataFrame(generated, columns=originals.columns)],
            ignore_index=True,
        )
    return originals.sort_values(
        ["frame_index", "local_track_id", "display_interpolated"], kind="stable"
    ).reset_index(drop=True)
