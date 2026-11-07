"""AST-based code graph builder producing harness-compatible node-link JSON.

Node ids are dotted module paths rooted at the package name (``pkg.mod.Class.method``).
Edges: ``contains``, ``imports``, ``inherits``, ``calls`` (calls carry a ``via`` strategy).
"""
import ast
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = _FUNCS + (ast.ClassDef,)


def _module_id(parts: list[str]) -> str:
    parts = [*parts[:-1], parts[-1].removesuffix(".py")]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _segment(lines: list[str], node) -> str:
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    seg = lines[start - 1:node.end_lineno]
    seg[0] = seg[0].lstrip()
    return "\n".join(seg)


def _flatten(expr) -> list[str] | None:
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        return [expr.id, *reversed(parts)]
    return None


class _Collector:
    """Pass 1: nodes, contains edges, per-module import tables, class bases.

    Ids drop enclosing *function* scopes: a def nested in a function is named after the nearest
    enclosing class or module (a class nested in a method is ``Outer.Inner``). ``members`` records
    true class membership for call resolution. Colliding ids (property getter/setter, overloads,
    if/else defs, flattened nested defs) keep the first node's text and accumulate every AST body
    in ``scopes`` so calls from all of them are collected.
    """

    def __init__(self, repo_root: Path):
        self.root = repo_root
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str, str | None]] = []
        self.imports: dict[str, dict[str, str]] = {}  # module -> alias -> dotted target
        self.bases: dict[str, list[ast.expr]] = {}
        self.scopes: dict[str, tuple[str, list[ast.AST]]] = {}  # symbol id -> (module id, ast defs)
        self.owner: dict[str, str] = {}  # symbol id -> lexically enclosing symbol/module id
        self.cls_ctx: dict[str, str | None] = {}  # symbol id -> nearest enclosing class id
        self.local_defs: dict[str, dict[str, str]] = {}  # function id -> {nested name: id}
        self.members: dict[str, dict[str, str]] = {}  # class id -> {member name: id}
        self.cur_cls: str | None = None  # nearest enclosing class, for id parents
        self.module_imports: list[tuple[str, str]] = []  # (module, dotted target)
        self.is_package: dict[str, bool] = {}
        self.skipped: list[Path] = []

    def add_module(self, mid: str, path: Path) -> None:
        if mid in self.nodes:
            return
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError) as exc:
            logger.warning("skipping unparseable file %s: %s", path, exc)
            self.skipped.append(path)
            return
        self.is_package[mid] = path.name == "__init__.py"
        self.nodes[mid] = self._node(mid, ast.get_docstring(tree, clean=False) or "", "module",
                                     path, 1, False, len(src.splitlines()))
        self.imports[mid] = {}
        self.cur_cls = None
        self._visit_body(tree.body, mid, mid, None, False, path, src.splitlines())

    def _node(self, nid, text, kind, path, lineno, is_async, end_lineno) -> dict:
        try:
            file = path.relative_to(self.root).as_posix()
        except ValueError:
            file = str(path)
        return {"id": nid, "name": nid, "text": text, "kind": kind,
                "file": file, "lineno": lineno, "end_lineno": end_lineno, "is_async": is_async}

    def _visit_body(self, body, mid, owner, cls_ctx, in_class, path, lines) -> None:
        for stmt in body:
            if isinstance(stmt, _SCOPES):
                is_class = isinstance(stmt, ast.ClassDef)
                is_async = isinstance(stmt, ast.AsyncFunctionDef)
                nid = f"{self.cur_cls or mid}.{stmt.name}"
                if nid in self.nodes:
                    self.scopes[nid][1].append(stmt)
                else:
                    if is_class:
                        kind = "class"
                    elif in_class:
                        kind = "method"
                    else:
                        kind = "async_function" if is_async else "function"
                    self.nodes[nid] = self._node(nid, _segment(lines, stmt), kind, path,
                                                 stmt.lineno, is_async, stmt.end_lineno)
                    self.scopes[nid] = (mid, [stmt])
                    self.owner[nid] = owner
                    self.cls_ctx[nid] = nid if is_class else cls_ctx
                if owner != nid:
                    self.edges.append((owner, nid, "contains", None))
                if in_class:
                    self.members.setdefault(owner, {}).setdefault(stmt.name, nid)
                if is_class:
                    self.bases.setdefault(nid, []).extend(stmt.bases)
                    prev = self.cur_cls
                    self.cur_cls = nid
                    self._visit_body(stmt.body, mid, nid, nid, True, path, lines)
                    self.cur_cls = prev
                else:
                    if self.nodes[owner]["kind"] not in ("class", "module"):
                        self.local_defs.setdefault(owner, {}).setdefault(stmt.name, nid)
                    self._visit_body(stmt.body, mid, nid, cls_ctx, False, path, lines)
                continue
            if isinstance(stmt, ast.Import):
                self._import(mid, stmt)
                continue
            if isinstance(stmt, ast.ImportFrom):
                self._import_from(mid, stmt)
                continue
            for field in ("body", "orelse", "finalbody"):  # defs under if/try/with
                self._visit_body(getattr(stmt, field, None) or [], mid, owner, cls_ctx,
                                 in_class, path, lines)
            for h in getattr(stmt, "handlers", None) or []:
                self._visit_body(h.body, mid, owner, cls_ctx, in_class, path, lines)

    def _import(self, mid: str, stmt: ast.Import) -> None:
        for a in stmt.names:
            if a.asname:
                self.imports[mid][a.asname] = a.name
            else:
                self.imports[mid][a.name.split(".")[0]] = a.name.split(".")[0]
            self.module_imports.append((mid, a.name))

    def _import_from(self, mid: str, stmt: ast.ImportFrom) -> None:
        base = stmt.module or ""
        if stmt.level:
            pkg = mid if self.is_package[mid] else mid.rpartition(".")[0]
            for _ in range(stmt.level - 1):
                pkg = pkg.rpartition(".")[0]
            base = f"{pkg}.{base}" if base else pkg
        for a in stmt.names:
            if a.name != "*":
                self.imports[mid][a.asname or a.name] = f"{base}.{a.name}"
        self.module_imports.append((mid, base))
        for a in stmt.names:  # `from pkg import submodule`
            if a.name != "*":
                self.module_imports.append((mid, f"{base}.{a.name}"))


class _Resolver:
    def __init__(self, c: _Collector):
        self.c = c
        self.by_bare: dict[str, list[str]] = {}
        for nid, n in c.nodes.items():
            if n["kind"] != "module":
                self.by_bare.setdefault(nid.rsplit(".", 1)[1], []).append(nid)

    def dotted(self, name: str, depth: int = 0) -> str | None:
        """Resolve a dotted path to a node id, following re-exports through imports."""
        if name in self.c.nodes:
            return name
        if depth > 5 or "." not in name:
            return None
        mod, _, last = name.rpartition(".")
        target = self.c.imports.get(mod, {}).get(last)
        if target and target != name:
            return self.dotted(target, depth + 1)
        return None

    def mro(self, cls_id: str, seen: set | None = None) -> list[str]:
        seen = seen if seen is not None else set()
        if cls_id in seen:
            return []
        seen.add(cls_id)
        out = [cls_id]
        mid = self.c.scopes[cls_id][0]
        for b in self.c.bases.get(cls_id, []):
            t, _ = self.name_target(b, mid, cls_id)
            if t and self.c.nodes[t]["kind"] == "class":
                out += self.mro(t, seen)
        return out

    def member(self, cls_id: str, name: str, skip_self: bool = False) -> str | None:
        """First member called ``name`` along the class's MRO-ish base chain."""
        for k in self.mro(cls_id)[1 if skip_self else 0:]:
            hit = self.c.members.get(k, {}).get(name)
            if hit:
                return hit
        return None

    def _head(self, head: str, mid: str, scope: str) -> str | None:
        """Lexical lookup: defs nested in enclosing functions, then module level."""
        s = scope
        while s in self.c.owner:  # class scope is not visible from methods (no local_defs)
            d = self.c.local_defs.get(s, {}).get(head)
            if d:
                return d
            s = self.c.owner[s]
        cand = f"{mid}.{head}"
        return cand if self.c.owner.get(cand) == mid else None

    def name_target(self, expr, mid: str, scope: str) -> tuple[str | None, str | None]:
        """Resolve a Name/Attribute expression (no instance attrs) to (node id, strategy)."""
        parts = _flatten(expr)
        if not parts:
            return None, None
        head, rest = parts[0], parts[1:]
        nid = self._head(head, mid, scope)
        if nid:
            return self._descend(nid, rest), "scope"
        imp = self.c.imports.get(mid, {}).get(head)
        if imp:
            d = self.dotted(imp)
            return (self._descend(d, rest) if d else self.dotted(".".join([imp, *rest]))), "import"
        return None, None

    def _descend(self, nid: str | None, rest: list[str]) -> str | None:
        for r in rest:
            if nid is None:
                return None
            if self.c.nodes[nid]["kind"] == "class":
                nid = self.member(nid, r)
            else:
                nid = self.dotted(f"{nid}.{r}")
        return nid

    def call_target(self, func, mid: str, scope: str) -> tuple[str, str] | None:
        parts = _flatten(func)
        ecls = self.c.cls_ctx.get(scope)
        if parts and len(parts) == 2 and parts[0] in ("self", "cls") and ecls:
            hit = self.member(ecls, parts[1])
            if hit:
                return hit, "self"
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name) and func.value.func.id == "super"
                and ecls):
            hit = self.member(ecls, func.attr, skip_self=True)
            if hit:
                return hit, "self"
        t, how = self.name_target(func, mid, scope)
        if t and self.c.nodes[t]["kind"] != "module":
            return t, how
        if isinstance(func, ast.Attribute):  # x.y(): only methods are plausible targets
            cands = [n for n in self.by_bare.get(func.attr, []) if self.c.nodes[n]["kind"] == "method"]
        elif isinstance(func, ast.Name):
            cands = self.by_bare.get(func.id, [])
        else:
            cands = []
        return (cands[0], "unique") if len(cands) == 1 else None


def _calls_in(node):
    """Yield Call nodes in ``node``'s own body, not descending into nested defs/classes."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, _SCOPES):
            continue
        if isinstance(n, ast.Call):
            yield n
        stack.extend(ast.iter_child_nodes(n))


def _module_files(repo_root: Path, package_dirs: list[Path] | None):
    """Yield (module id, path). Default: every .py under the root, ``src/`` prefix stripped."""
    if package_dirs is not None:
        for pkg in map(Path, package_dirs):
            for path in sorted(pkg.rglob("*.py")):
                yield _module_id([pkg.name, *path.relative_to(pkg).parts]), path
        return
    for path in sorted(repo_root.rglob("*.py")):
        parts = path.relative_to(repo_root).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts[:-1]):
            continue
        if parts[0] == "src" and len(parts) > 1:
            parts = parts[1:]
        mid = _module_id(list(parts))
        if mid:
            yield mid, path


def build_graph(repo_root: Path, repo_short: str, package_dirs: list[Path] | None = None) -> dict:
    repo_root = Path(repo_root)
    c = _Collector(repo_root)
    for mid, path in _module_files(repo_root, package_dirs):
        c.add_module(mid, path)
    if c.skipped:
        logger.warning("%s: skipped %d unparseable file(s)", repo_short, len(c.skipped))
    r = _Resolver(c)

    edges: dict[tuple[str, str, str], str | None] = {}
    for s, t, ty, via in c.edges:
        edges.setdefault((s, t, ty), via)
    for mid, target in c.module_imports:
        t = r.dotted(target)
        if t and c.nodes[t]["kind"] == "module" and t != mid:
            edges.setdefault((mid, t, "imports"), None)
    for cid, bases in c.bases.items():
        mid = c.scopes[cid][0]
        for b in bases:
            t, _ = r.name_target(b, mid, c.owner[cid])
            if t and c.nodes[t]["kind"] == "class" and t != cid:
                edges.setdefault((cid, t, "inherits"), None)
    # calls from functions and methods (module/class-body calls are not recorded)
    for sid, (mid, defs) in c.scopes.items():
        for node in defs:
            if isinstance(node, ast.ClassDef):
                continue
            for call in _calls_in(node):
                hit = r.call_target(call.func, mid, sid)
                if hit and hit[0] != sid:
                    edges.setdefault((sid, hit[0], "calls"), hit[1])

    counter: dict[tuple[str, str], int] = {}
    out_edges = []
    for (s, t, ty), via in sorted(edges.items()):
        key = counter.get((s, t), 0)
        counter[(s, t)] = key + 1
        e = {"source": s, "target": t, "type": ty, "key": key}
        if via:
            e["via"] = via
        out_edges.append(e)
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": list(c.nodes.values()),
            "edges": out_edges}


def filter_edges(graph: dict, keep: set[str]) -> dict:
    counter: dict[tuple[str, str], int] = {}
    edges = []
    for e in graph["edges"]:
        if e["type"] in keep:
            k = counter.get((e["source"], e["target"]), 0)
            counter[(e["source"], e["target"])] = k + 1
            edges.append({**e, "key": k})
    return {**graph, "edges": edges}


def write_graph(graph: dict, out_dir: Path, repo_short: str, commit: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{repo_short}_{commit}.json"
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return path
