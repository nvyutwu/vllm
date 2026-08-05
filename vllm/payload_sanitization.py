# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ASSET_ID_URI_RE = re.compile(
    r"(data:(?:image|video|audio)/[^;]+;asset_id,)([^\s\"'<>]+)",
    re.IGNORECASE,
)

_NVCF_ASSET_DIR_HEADERS = ("nvcf-input-asset-dir", "nvcf-asset-dir")


def has_nvcf_asset_headers(headers: Mapping[str, str] | None) -> bool:
    """Cheap check for whether NVCF asset headers are present.

    Lets request handlers skip re-parsing the body when a request carries no
    asset references, which is the common case.
    """
    if not headers:
        return False
    present = {str(k).lower() for k in headers}
    return any(name in present for name in _NVCF_ASSET_DIR_HEADERS)


def materialize_nvcf_asset_refs(
    payload: Any, *, headers: Mapping[str, str] | None
) -> Any:
    """Rewrite NVCF ``asset_id`` multimodal refs into local ``file://`` URIs.

    NVCF delivers uploaded assets as files in a directory and advertises them
    via request headers. Multimodal parts arrive as
    ``data:<mime>;asset_id,<id>`` URIs; this resolves each allow-listed id to
    the corresponding file on disk so the normal media loader can read it.

    Args:
        payload: The parsed request body (nested dict/list/str structure).
        headers: The incoming request headers.

    Returns:
        The payload with resolvable asset refs rewritten to ``file://`` URIs.
        The input is returned unchanged when the required headers are absent
        or the asset directory is missing.
    """
    if payload is None or not headers:
        return payload

    header_lookup = {str(k).lower(): str(v) for k, v in headers.items()}
    asset_dir = header_lookup.get("nvcf-input-asset-dir") or header_lookup.get(
        "nvcf-asset-dir"
    )
    allowed_ids_hdr = header_lookup.get(
        "nvcf-input-asset-references"
    ) or header_lookup.get("nvcf-function-asset-ids")
    if not asset_dir or not allowed_ids_hdr:
        return payload

    asset_root = Path(asset_dir).resolve()
    allowed_ids = {
        _normalize_asset_id(item) for item in allowed_ids_hdr.split(",") if item.strip()
    }
    if not allowed_ids or not asset_root.exists() or not asset_root.is_dir():
        return payload

    def replace_text(text: str) -> str:
        return _ASSET_ID_URI_RE.sub(
            lambda match: _asset_match_to_file_url(
                match,
                asset_root=asset_root,
                allowed_ids=allowed_ids,
            ),
            text,
        )

    def visit(value: Any) -> Any:
        if isinstance(value, str):
            return replace_text(value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value

    return visit(payload)


def _asset_match_to_file_url(
    match: re.Match[str],
    *,
    asset_root: Path,
    allowed_ids: set[str],
) -> str:
    asset_id = _normalize_asset_id(match.group(2) or "")
    if asset_id not in allowed_ids:
        return match.group(0)

    filepath = (asset_root / asset_id).resolve()
    if asset_root not in filepath.parents and filepath != asset_root:
        return match.group(0)

    if not filepath.exists() or not filepath.is_file():
        return match.group(0)

    return filepath.as_uri()


def _normalize_asset_id(value: str) -> str:
    normalized = (value or "").strip().strip(",").strip()
    while (
        len(normalized) >= 2
        and normalized[0] in ("'", '"')
        and normalized[-1] == normalized[0]
    ):
        normalized = normalized[1:-1].strip()
    return normalized
