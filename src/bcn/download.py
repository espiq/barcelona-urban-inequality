"""Download raw files from the two open-data portals used by the study.

* Open Data BCN (Ajuntament de Barcelona) - CKAN API, file resources.
* Dades Obertes de Catalunya (Generalitat) - Socrata API, paginated queries.

Every file is cached under ``data/raw/<dataset>/`` so the pipeline is
idempotent: re-running it only fetches what is missing. A ``manifest.json``
records the source URL, size and SHA-256 of every file for provenance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import requests

from .config import (
    CKAN_API,
    DATA_RAW,
    DATASETS,
    EDU_DATASETS,
    PADRO_EDU_YEARS,
    SOCRATA_API,
    YEARS,
)

log = logging.getLogger(__name__)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "barcelona-urban-inequality/1.0 (research project)"


def _get(url: str, retries: int = 4, **kwargs) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=120, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            log.warning("GET %s failed (%s); retrying in %ss", url, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def package_resources(package: str) -> list[dict]:
    """Return the resource list of a CKAN package."""
    resp = _get(f"{CKAN_API}/package_show", params={"id": package})
    return resp.json()["result"]["resources"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(out_dir: Path) -> dict:
    path = out_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _provenance(path: Path, **source) -> dict:
    return {
        **source,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def fetch(
    key: str,
    select: Callable[[dict], bool],
    rename: Callable[[dict], str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download every resource of dataset ``key`` for which ``select`` is true."""
    package = DATASETS[key]
    out_dir = DATA_RAW / key
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out_dir)

    paths = []
    for res in package_resources(package):
        if not select(res):
            continue
        fname = rename(res) if rename else res["name"]
        path = out_dir / fname
        if path.exists() and not force:
            paths.append(path)
            continue
        log.info("Downloading %s/%s", key, fname)
        path.write_bytes(_get(res["url"]).content)
        manifest[fname] = _provenance(path, package=package, resource_id=res["id"], url=res["url"])
        paths.append(path)

    _save_manifest(out_dir, manifest)
    if not paths:
        raise FileNotFoundError(f"No resources matched for dataset '{key}' ({package})")
    return paths


def fetch_socrata(key: str, page_size: int = 200_000, force: bool = False) -> Path:
    """Download a whole Socrata dataset (optionally filtered) as one CSV.

    Uses the SODA ``/resource`` endpoint, which returns machine field names,
    paging with ``$limit``/``$offset`` ordered by the internal ``:id`` so pages
    are stable and non-overlapping.
    """
    spec = EDU_DATASETS[key]
    out_dir = DATA_RAW / key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}.csv"
    if path.exists() and not force:
        return path

    url = f"{SOCRATA_API}/resource/{spec['id']}.csv"
    base = {"$order": ":id", "$limit": page_size}
    if spec["where"]:
        base["$where"] = spec["where"]

    tmp = path.with_suffix(".part")
    rows, offset = 0, 0
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        while True:
            text = _get(url, params={**base, "$offset": offset}).content.decode("utf-8")
            records = list(csv.reader(io.StringIO(text)))
            header, page = records[0], records[1:]
            if offset == 0:
                writer.writerow(header)
            writer.writerows(page)
            rows += len(page)
            offset += page_size
            log.info("%s: %s rows", key, f"{rows:,}")
            if len(page) < page_size:
                break
    tmp.replace(path)

    manifest = _load_manifest(out_dir)
    manifest[path.name] = _provenance(path, dataset_id=spec["id"], url=url, where=spec["where"])
    _save_manifest(out_dir, manifest)
    return path


def _is_csv(res: dict) -> bool:
    return res.get("format", "").upper() == "CSV"


def _year_csv(years: Iterable[int]) -> Callable[[dict], bool]:
    prefixes = tuple(str(y) for y in years)
    return lambda res: _is_csv(res) and res["name"].startswith(prefixes)


def _year_name(key: str) -> Callable[[dict], str]:
    return lambda res: f"{res['name'][:4]}_{key}.csv"


def download_all(force: bool = False) -> None:
    """Fetch every raw input used by the study."""
    fetch(
        "boundaries",
        lambda r: (
            _is_csv(r)
            and r["name"] in ("BarcelonaCiutat_Barris.csv", "BarcelonaCiutat_SeccionsCensals.csv")
        ),
        force=force,
    )
    for key in ("income", "gini", "population"):
        fetch(key, _year_csv(YEARS), _year_name(key), force=force)
    fetch("education", _year_csv(PADRO_EDU_YEARS), _year_name("education"), force=force)
    fetch("cadastre", _year_csv([YEARS[-1]]), _year_name("cadastre"), force=force)
    fetch(
        "tourist_flats",
        lambda r: _is_csv(r) and r["name"] == "hut_comunicacio_opendata.csv",
        force=force,
    )


def download_education(force: bool = False) -> None:
    """Fetch the Part II (education) inputs. The register-based attainment
    series is part of ``download_all``; this adds the Generalitat datasets."""
    fetch("education", _year_csv(PADRO_EDU_YEARS), _year_name("education"), force=force)
    for key in EDU_DATASETS:
        fetch_socrata(key, force=force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_all()
