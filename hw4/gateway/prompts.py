import hashlib
import json
import random
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Optional

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class PromptError(Exception):
    pass


class PromptNotFound(PromptError):
    pass


class PromptConflict(PromptError):
    pass


@dataclass
class ResolvedPrompt:
    name: str
    version: str
    template: str
    source: str  # explicit | ab_test | active | latest

    def render(self, **variables) -> str:
        # safe_substitute：缺失变量原样保留，不抛 KeyError
        return Template(self.template).safe_substitute(**variables)


def _version_key(v: str):
    m = re.search(r"\d+", v)
    return (int(m.group()) if m else 0, v)


class PromptManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # A/B 分流统计：{prompt_name: {version: count}}（内存态，重启清零）
        self.stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # ---------------- 基础读写 ---------------- #
    def _dir(self, name: str) -> Path:
        if not _NAME_RE.match(name):
            raise PromptError(f"invalid prompt name: {name!r}")
        return self.root / name

    def _meta(self, name: str) -> dict:
        p = self._dir(name) / "meta.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {"active": None, "experiment": {"enabled": False, "weights": {}}}

    def _save_meta(self, name: str, meta: dict) -> None:
        (self._dir(name) / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def versions(self, name: str) -> list[str]:
        d = self._dir(name)
        if not d.exists():
            raise PromptNotFound(f"prompt not found: {name}")
        return sorted((f.stem for f in d.glob("*.txt")), key=_version_key)

    def names(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir() and any(d.glob("*.txt")))

    def _template(self, name: str, version: str) -> str:
        if not _NAME_RE.match(version):
            raise PromptError(f"invalid version: {version!r}")
        f = self._dir(name) / f"{version}.txt"
        if not f.exists():
            raise PromptNotFound(f"prompt version not found: {name}@{version}")
        return f.read_text(encoding="utf-8")

    # ---------------- 版本管理 ---------------- #
    def create_version(self, name: str, version: str, template: str, activate: bool = False) -> dict:
        if not _NAME_RE.match(version):
            raise PromptError(f"invalid version: {version!r}")
        d = self._dir(name)
        with self._lock:
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{version}.txt"
            if f.exists():
                raise PromptConflict(f"{name}@{version} already exists (versions are immutable)")
            f.write_text(template, encoding="utf-8")
            meta = self._meta(name)
            if activate or not meta["active"]:
                meta["active"] = version
            self._save_meta(name, meta)
        return self.info(name)

    def set_active(self, name: str, version: str) -> dict:
        self._template(name, version)  # 校验存在
        with self._lock:
            meta = self._meta(name)
            meta["active"] = version
            self._save_meta(name, meta)
        return self.info(name)

    def set_experiment(self, name: str, enabled: bool, weights: dict[str, float]) -> dict:
        existing = set(self.versions(name))
        if enabled:
            unknown = set(weights) - existing
            if unknown:
                raise PromptError(f"unknown versions in weights: {sorted(unknown)}")
            if not weights or sum(weights.values()) <= 0 or any(w < 0 for w in weights.values()):
                raise PromptError("weights must be non-negative and sum > 0")
        with self._lock:
            meta = self._meta(name)
            meta["experiment"] = {"enabled": enabled, "weights": weights}
            self._save_meta(name, meta)
        return self.info(name)

    def info(self, name: str) -> dict:
        meta = self._meta(name)
        return {"name": name, "versions": self.versions(name), "active": meta["active"],
                "experiment": meta["experiment"]}

    # ---------------- 解析（选择用哪个版本） ---------------- #
    def resolve(self, name: str, *, version: Optional[str] = None,
                user_key: Optional[str] = None) -> ResolvedPrompt:
        versions = self.versions(name)
        if not versions:
            raise PromptNotFound(f"prompt has no versions: {name}")
        meta = self._meta(name)
        exp = meta.get("experiment") or {}

        if version:
            chosen, source = version, "explicit"
        elif exp.get("enabled") and exp.get("weights"):
            chosen, source = self._pick(name, exp["weights"], versions, user_key), "ab_test"
        elif meta.get("active") in versions:
            chosen, source = meta["active"], "active"
        else:
            chosen, source = versions[-1], "latest"

        with self._lock:
            self.stats[name][chosen] += 1
        return ResolvedPrompt(name, chosen, self._template(name, chosen), source)

    @staticmethod
    def _pick(name: str, weights: dict[str, float], existing: list[str], user_key: Optional[str]) -> str:
        items = [(v, w) for v, w in weights.items() if v in existing and w > 0]
        if not items:
            return existing[-1]
        total = sum(w for _, w in items)
        if user_key:
            h = int(hashlib.sha256(f"{name}:{user_key}".encode()).hexdigest(), 16)
            r = (h % 10_000) / 10_000 * total
        else:
            r = random.random() * total
        acc = 0.0
        for v, w in items:
            acc += w
            if r < acc:
                return v
        return items[-1][0]

    def get_stats(self, name: str) -> dict:
        return {"name": name, "served": dict(self.stats.get(name, {}))}
