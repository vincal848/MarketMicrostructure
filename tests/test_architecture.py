"""The layering in docs/ARCHITECTURE.md, enforced.

Every module belongs to one layer and may import only from its own layer or
lower ones. The core (layer 0) does no I/O. A new module must be added to
LAYERS; an import that reaches upward fails this test.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "microstructure"

LAYERS: dict[str, int] = {
    # 0 core: pure, typed, no I/O
    "events": 0,
    "book": 0,
    "hawkes": 0,
    "hawkes_estimation": 0,
    "avellaneda_stoikov": 0,
    # 1 order-level source adapters
    "itch": 1,
    "lobster": 1,
    # 2 replay
    "replay": 2,
    # 3 flow: classification, calibration, level-data adapter, stylized facts
    "flow": 3,
    "calibration": 3,
    "databento": 3,
    "stylized": 3,
    # 4 simulation
    "simulator": 4,
    # 5 agents and accounting
    "accounting": 5,
    "agents": 5,
    # 6 experiments over the simulator
    "evaluation": 6,
    "env": 6,
    "rl": 6,
    # 7 entry points
    "experiment": 7,
    "bench": 7,
    "cli": 7,
}

IO_CALLS = {"open", "print", "input"}
IO_MODULES = {"logging", "pathlib", "os", "sys", "subprocess", "shutil"}


def _internal_imports(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("microstructure."):
            imported.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            imported |= {a.name.split(".")[1] for a in node.names if a.name.startswith("microstructure.")}
    return imported


def _modules() -> list[str]:
    return sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__")


def test_every_module_is_assigned_a_layer() -> None:
    assert set(_modules()) == set(LAYERS)


@pytest.mark.parametrize("module", sorted(LAYERS))
def test_imports_never_reach_a_higher_layer(module: str) -> None:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    upward = {name for name in _internal_imports(tree) if LAYERS[name] > LAYERS[module]}
    assert not upward, f"{module} (layer {LAYERS[module]}) imports higher layers: {sorted(upward)}"


@pytest.mark.parametrize("module", sorted(m for m, layer in LAYERS.items() if layer == 0))
def test_core_modules_do_no_io(module: str) -> None:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imports |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not calls & IO_CALLS, f"{module} calls {sorted(calls & IO_CALLS)}"
    assert not imports & IO_MODULES, f"{module} imports {sorted(imports & IO_MODULES)}"
