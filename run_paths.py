"""Dated run folders for pipeline outputs: ``assets/<YYYY-MM-DD>/<stage>/``.

Every run date has its own folder under ``assets/`` with one sub-folder per
top-level code folder (``00`` … ``06``). A stage writes only into today's folder.
Files it reads from other stages (or from earlier steps of its own stage) are
copied into today's folder first, from the newest earlier run that has them, so
each date folder is self-contained. Copies keep their original modification
time, so a file dated before its run folder was carried over, not produced there.

Consistency check
-----------------
Copying makes it possible to combine files that don't belong together, e.g. new
topic IDs from a re-run of 02 with topic names from an older run of 03. Whenever a
notebook reads files, an upstream file (earlier stage, same corpus) must not be
newer than a downstream one. If it is, and its content differs from the version
that existed when the downstream file was made, a ``StaleInputError`` is raised
naming the step to re-run.

Environment overrides
---------------------
``IGEM_RUN_DATE``     run date ``YYYY-MM-DD`` (default: today, fixed at first import)
``IGEM_ASSETS_DIR``   assets root (default: ``<repo>/assets``)
``IGEM_ALLOW_STALE``  ``1`` turns consistency errors into printed warnings

Command line
------------
``python run_paths.py [YYYY-MM-DD]``   status of a run (default: the newest)
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.environ.get("IGEM_ASSETS_DIR") or REPO_ROOT / "assets").resolve()
RUN_DATE = os.environ.get("IGEM_RUN_DATE") or date.today().isoformat()

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
if not _DATE_RE.match(RUN_DATE):
    raise ValueError(f"IGEM_RUN_DATE must look like YYYY-MM-DD, got {RUN_DATE!r}")

STAGES = ("00", "01", "02", "03", "04", "05", "06")
CORPORA = ("teams", "papers")
# Dataset of each corpus; ``{data}`` in a setup script's NEEDS expands to it.
DATASETS = {"teams": "00/igem.txt", "papers": "01/synbio_openalex.txt"}

# Files that later steps read, and the step that writes each one ({dir} is the
# corpus sub-folder). Used for "not found" / "re-run" messages and by the status
# command.
_CORPUS_DIR = {"teams": "01-teams", "papers": "02-papers"}
_PRODUCERS = [
    ("00", "igem.txt", "00-IGEM_teams_dataset/read_igem_data.ipynb"),
    ("01", "synbio_openalex_PART_*.txt", "01-SynBio_OpenAlex_dataset/get_synbio_data.ipynb"),
    ("01", "synbio_openalex.txt", "01-SynBio_OpenAlex_dataset/merge_and_describe.ipynb"),
    ("02", "*_corpus.txt", "02-topic_model/{dir}/get_embeddings.ipynb"),
    ("02", "*_embeddings.npy", "02-topic_model/{dir}/get_embeddings.ipynb"),
    ("02", "*_topic_model", "02-topic_model/{dir}/get_topics_with_evaluation.ipynb"),
    ("02", "*_topic_info.txt", "02-topic_model/{dir}/get_topics_with_evaluation.ipynb"),
    ("02", "*_doc_topics.txt", "02-topic_model/{dir}/get_topics_with_evaluation.ipynb"),
    ("03", "*_topic_names.txt", "03-topic_names/{dir}/get_topic_names_part1.ipynb + part2"),
    ("04", "*_topic_hierarchy_map.tsv", "04-topic_hierarchy/{dir}/get_topic_hierarchy.ipynb"),
    ("04", "*_topic_name_hierarchy.tsv",
     "04-topic_hierarchy/{dir}/get_topic_hierarchy.ipynb + name_hierarchy_levels.ipynb"),
    ("05", "cluster_summary_igem.tsv", "05-reporting/01-cluster_summary_IGEM.ipynb"),
    ("05", "cluster_summary_papers.tsv", "05-reporting/02-cluster_summary_papers.ipynb"),
    ("05", "joint_umap_*_xy.tsv", "05-reporting/03-compute_coords.ipynb"),
    ("05", "overlap_precedence_full.tsv", "05-reporting/06-density_comparison.ipynb"),
    ("05", "*_preceded.tsv", "05-reporting/06-density_comparison.ipynb"),
    ("05", "geography_location_quotient.tsv", "05-reporting/08-appendix.ipynb (geography_lq.py)"),
    ("05", "geography_outsiderness_teams.tsv",
     "05-reporting/08-appendix.ipynb (geography_outsiderness.py)"),
]

# Production order *within* a stage, for files where it matters (a later tier is
# built from an earlier one). Files not listed are not ordered within their stage.
_TIERS = {
    "01": [["synbio_openalex_PART_*.txt"], ["synbio_openalex.txt"]],
    "02": [["*_corpus.txt", "*_embeddings.npy"],
           ["*_topic_model", "*_topic_info.txt", "*_doc_topics.txt"]],
    "05": [["joint_umap_*_xy.tsv"],
           ["overlap_precedence_full.tsv", "*_preceded.tsv", "geography_outsiderness_teams.tsv"]],
}

_MTIME_SLACK = 2.0   # seconds; files written in the same step can share a timestamp


class StaleInputError(RuntimeError):
    """Inputs combine a downstream file with a newer version of what it was built from."""


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_dates() -> list[str]:
    """All run dates under ``ASSETS_DIR``, oldest first."""
    if not ASSETS_DIR.is_dir():
        return []
    return sorted(p.name for p in ASSETS_DIR.iterdir() if p.is_dir() and _DATE_RE.match(p.name))


def _split(key: str) -> tuple[str, str]:
    stage, _, name = key.partition("/")
    if stage not in STAGES or not name:
        raise ValueError(f"expected '<stage>/<file>' with stage in {STAGES}, got {key!r}")
    return stage, name


def _corpora(key: str) -> set[str]:
    """Which corpus a file belongs to: by name prefix, else by stage (05/06 → both)."""
    stage, name = _split(key)
    if name.startswith("teams_") or name == "cluster_summary_igem.tsv":
        return {"teams"}
    if name.startswith("papers_") or name == "cluster_summary_papers.tsv":
        return {"papers"}
    if stage == "00":
        return {"teams"}
    if stage == "01":
        return {"papers"}
    return set(CORPORA)


def _tier(key: str) -> int | None:
    stage, name = _split(key)
    for i, patterns in enumerate(_TIERS.get(stage, [])):
        if any(fnmatch.fnmatch(name, p) for p in patterns):
            return i
    return None


def _upstream(a: str, b: str) -> str | None:
    """Return whichever of two keys comes first in the pipeline, or None if unordered."""
    if not (_corpora(a) & _corpora(b)):
        return None
    sa, sb = _split(a)[0], _split(b)[0]
    if sa != sb:
        return a if sa < sb else b
    ta, tb = _tier(a), _tier(b)
    if ta is None or tb is None or ta == tb:
        return None
    return a if ta < tb else b


def producer(key: str) -> str:
    """The notebook/script that writes ``key``."""
    stage, name = _split(key)
    corpus = next((c for c in CORPORA if name.startswith(c + "_")), None)
    for s, pattern, who in _PRODUCERS:
        if s == stage and fnmatch.fnmatch(name, pattern):
            return who.replace("{dir}", _CORPUS_DIR.get(corpus, "<corpus>"))
    return f"stage {stage}"


def _is_interface(key: str) -> bool:
    stage, name = _split(key)
    return any(s == stage and fnmatch.fnmatch(name, p) for s, p, _ in _PRODUCERS)


def _mtime(path: Path) -> float:
    if path.is_dir():
        times = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
        return max(times) if times else path.stat().st_mtime
    return path.stat().st_mtime


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


_HASHES: dict[tuple[str, float], str] = {}


def _sha256(path: Path) -> str:
    cache_key = (str(path), _mtime(path))
    if cache_key not in _HASHES:
        h = hashlib.sha256()
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for f in files:
            if path.is_dir():
                h.update(str(f.relative_to(path)).encode())
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
        _HASHES[cache_key] = h.hexdigest()
    return _HASHES[cache_key]


def _copy(src: Path, dst: Path) -> None:
    """Copy keeping the modification time (an APFS clone on macOS: no extra disk space)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
    done = False
    if sys.platform == "darwin":
        done = subprocess.run(["cp", "-cpR", str(src), str(tmp)], capture_output=True).returncode == 0
    if not done:
        if tmp.exists():
            shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()
        if src.is_dir():
            shutil.copytree(src, tmp, copy_function=shutil.copy2)
        else:
            shutil.copy2(src, tmp)
    tmp.rename(dst)


def _newest_before(stage: str, name: str, before: str) -> Path | None:
    """The copy of ``stage/name`` in the newest run dated before ``before``."""
    for d in reversed(run_dates()):
        if d < before and (ASSETS_DIR / d / stage / name).exists():
            return ASSETS_DIR / d / stage / name
    return None


def _version_at(stage: str, name: str, t: float, exclude: Path) -> Path | None:
    """The newest copy of ``stage/name`` (in any run) that existed at time ``t``."""
    best, best_t = None, None
    for d in run_dates():
        p = ASSETS_DIR / d / stage / name
        if p.exists() and p != exclude:
            pt = _mtime(p)
            if pt <= t + _MTIME_SLACK and (best_t is None or pt > best_t):
                best, best_t = p, pt
    return best


def find_problems(files: dict[str, Path]) -> list[tuple[str, str]]:
    """Pairs where an upstream file is newer, with different content, than a downstream one.

    Returns ``(downstream_key, upstream_key)`` tuples.
    """
    problems = []
    keys = sorted(files)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            up = _upstream(a, b)
            if up is None:
                continue
            down = b if up == a else a
            t_up, t_down = _mtime(files[up]), _mtime(files[down])
            if t_up <= t_down + _MTIME_SLACK:
                continue
            stage, name = _split(up)
            before = _version_at(stage, name, t_down, exclude=files[up])
            if before is not None and _sha256(before) == _sha256(files[up]):
                continue   # upstream was re-run but produced identical content
            problems.append((down, up))
    return problems


def describe_problems(problems: list[tuple[str, str]], files: dict[str, Path]) -> str:
    """Readable report, grouped by the upstream file that changed."""
    by_up: dict[str, list[str]] = {}
    for down, up in problems:
        by_up.setdefault(up, []).append(down)
    lines = []
    for up in sorted(by_up):
        lines.append(f"  • {up} changed on {_fmt(_mtime(files[up]))}, after these files built on it:")
        for down in sorted(set(by_up[up])):
            lines.append(f"      {down:<38} ({_fmt(_mtime(files[down]))})  → re-run {producer(down)}")
    return "\n".join(lines)


def _allow_stale(flag: bool | None) -> bool:
    if flag is not None:
        return flag
    return os.environ.get("IGEM_ALLOW_STALE", "").strip().lower() in ("1", "true", "yes")


# ── Run handle ────────────────────────────────────────────────────────────────
class Run:
    """Today's run folder for one stage.

    ``run.dir``            this stage's folder, ``assets/<RUN_DATE>/<stage>/``
    ``run.out(name)``      path to write an output (parent folders created)
    ``run.get(stage, name)``  path to read an input from today's run, copying it
                           from the newest earlier run first if missing
    ``run.get_glob(stage, pattern)``  all matches from a single run
    """

    def __init__(self, stage: str, allow_stale: bool | None = None):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        self.stage = stage
        self.date = RUN_DATE
        self.root = ASSETS_DIR / RUN_DATE
        self._allow_stale = allow_stale
        self._read: dict[str, Path] = {}
        self._reported: set[tuple[str, str]] = set()   # problems already reported (warn once)

    def __repr__(self):
        return f"Run(date={self.date!r}, stage={self.stage!r}, dir={str(self.path)!r})"

    @property
    def path(self) -> Path:
        """This stage's folder in today's run (not created)."""
        return self.root / self.stage

    @property
    def dir(self) -> Path:
        d = self.root / self.stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def out(self, name: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _fetch(self, stage: str, name: str) -> tuple[Path, str | None]:
        """Make sure ``stage/name`` is in today's run; return (path, source date if copied)."""
        path = self.root / stage / name
        if path.exists():
            return path, None
        src = _newest_before(stage, name, self.date)
        if src is None:
            raise FileNotFoundError(
                f"{stage}/{name} is not in this run ({self.date}) or any earlier run under "
                f"{ASSETS_DIR}. It is produced by {producer(f'{stage}/{name}')}."
            )
        _copy(src, path)
        return path, src.parent.parent.name

    def get(self, stage: str, name: str | None = None) -> Path:
        """Path of ``stage/name`` in today's run (also accepts one ``"stage/name"`` key)."""
        if name is None:
            stage, name = _split(stage)
        key = f"{stage}/{name}"
        path, copied_from = self._fetch(stage, name)
        if copied_from:
            print(f"↳ copied {key} from {copied_from}")
        self._register({key: path})
        return path

    def get_glob(self, stage: str, pattern: str) -> list[Path]:
        """All ``stage/pattern`` matches, taken together from one run (never mixed)."""
        here = sorted((self.root / stage).glob(pattern))
        if not here:
            for d in reversed(run_dates()):
                if d < self.date:
                    found = sorted((ASSETS_DIR / d / stage).glob(pattern))
                    if found:
                        for src in found:
                            _copy(src, self.root / stage / src.name)
                        print(f"↳ copied {len(found)} × {stage}/{pattern} from {d}")
                        here = sorted((self.root / stage).glob(pattern))
                        break
        if not here:
            raise FileNotFoundError(
                f"no {stage}/{pattern} in this run ({self.date}) or any earlier run. "
                f"Produced by {producer(f'{stage}/{pattern}')}."
            )
        self._register({f"{stage}/{p.name}": p for p in here})
        return here

    def _register(self, new: dict[str, Path]) -> None:
        """Record newly read files and check them."""
        self._read.update(new)
        self._check(focus=set(new))

    def _check(self, focus: set[str] | None = None) -> None:
        """Check the files read so far against each other and against the newest
        version (up to this run's date) of everything upstream of them, so a file
        built on data that has since changed is caught even if that data isn't read.
        """
        files = {**latest_files(self.date), **self._read}
        problems = [(down, up) for down, up in find_problems(files)
                    if down in self._read and (focus is None or down in focus or up in focus)
                    and (down, up) not in self._reported]
        if not problems:
            return
        self._reported.update(problems)
        text = "Inputs are out of date:\n" + describe_problems(problems, files)
        if _allow_stale(self._allow_stale):
            print("⚠ " + text + "\n  (continuing because IGEM_ALLOW_STALE is set)")
        else:
            raise StaleInputError(text + "\nSet IGEM_ALLOW_STALE=1 to continue anyway.")


def open_run(stage: str, allow_stale: bool | None = None) -> Run:
    """Run handle for a stage with no upstream (00, 01) or for ad-hoc use."""
    run = Run(stage, allow_stale)
    print(f"Run {run.date} · stage {stage} → {run.path}")
    return run


def expand_needs(needs: list[str], corpus: str | None = None) -> list[str]:
    """Expand ``{c}`` (corpus) and ``{data}`` (its dataset) in NEEDS templates."""
    corpora = CORPORA if corpus is None else (corpus,)
    if corpus is not None and corpus not in CORPORA:
        raise ValueError(f"corpus must be one of {CORPORA} or None, got {corpus!r}")
    keys: list[str] = []
    for template in needs:
        if "{c}" in template or "{data}" in template:
            for c in corpora:
                key = DATASETS[c] if template == "{data}" else template.replace("{c}", c)
                if key not in keys:
                    keys.append(key)
        elif template not in keys:
            keys.append(template)
    return keys


def prepare(stage: str, needs: list[str], corpus: str | None = None,
            allow_stale: bool | None = None) -> Run:
    """Setup for one stage: copy missing upstream files into today's run, then check them.

    Files found in no run at all are reported but don't stop the setup; the notebook
    fails later only if it actually reads them.
    """
    run = Run(stage, allow_stale)
    found: dict[str, Path] = {}
    copied, missing = [], []
    for key in expand_needs(needs, corpus):
        s, name = _split(key)
        try:
            path, copied_from = run._fetch(s, name)
        except FileNotFoundError:
            missing.append(key)
            continue
        found[key] = path
        if copied_from:
            copied.append((key, copied_from))

    scope = f" ({corpus})" if corpus else ""
    print(f"Run {run.date} · stage {stage}{scope} → {run.path}")
    for key, d in copied:
        print(f"  ↳ copied {key} from {d}")
    for key in missing:
        print(f"  ✗ {key} not found in any run — produced by {producer(key)}")
    print(f"  {len(found) - len(copied)} upstream file(s) already in this run, "
          f"{len(copied)} copied, {len(missing)} missing")

    run._read.update(found)
    run._check()
    return run


def latest_files(up_to: str) -> dict[str, Path]:
    """The newest copy of every file later steps read, over all runs dated ``<= up_to``."""
    latest: dict[str, Path] = {}
    for d in run_dates():
        if d > up_to:
            break
        for stage in STAGES:
            folder = ASSETS_DIR / d / stage
            if folder.is_dir():
                for p in folder.iterdir():
                    key = f"{stage}/{p.name}"
                    if not p.name.endswith(".partial") and _is_interface(key):
                        latest[key] = p
    return latest


# ── Status command ────────────────────────────────────────────────────────────
def status(run_date: str | None = None) -> None:
    """Print what a run folder holds, and whether its latest inputs are consistent."""
    dates = run_dates()
    if not dates:
        print(f"No run folders under {ASSETS_DIR}")
        return
    run_date = run_date or dates[-1]
    root = ASSETS_DIR / run_date
    print(f"Runs: {', '.join(dates)}\n")
    print(f"Run {run_date}  ({root})")
    if not root.is_dir():
        print("  (no such run folder)")
    else:
        for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            items = sorted(p for p in stage_dir.iterdir() if not p.name.startswith(("_", ".")))
            produced = [p for p in items if _fmt(_mtime(p))[:10] >= run_date]
            carried = [p for p in items if p not in produced]
            print(f"  {stage_dir.name}  {len(produced)} produced in this run, {len(carried)} carried over")
            for p in carried:
                print(f"        {p.name:<44} produced {_fmt(_mtime(p))[:10]}")

    # The newest version of every file later steps read: what a new run would pick up.
    latest = latest_files(run_date)
    problems = find_problems(latest)
    print(f"\nConsistency of the newest inputs up to {run_date}: "
          f"{'OK' if not problems else f'{len(problems)} problem(s)'}")
    if problems:
        print(describe_problems(problems, latest))


if __name__ == "__main__":
    status(sys.argv[1] if len(sys.argv) > 1 else None)
