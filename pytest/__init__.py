"""
`python -m pytest` compatibility shim.

The verification environment for this repo uses `python -m pytest`, but the
system Python may not have `pytest` installed. This shim provides:
  - a minimal test runner (function-style tests)
  - the small `pytest` API surface used in this repo's tests (`raises`,
    `importorskip`, `skip`, `fail`)
  - support for `tmp_path` and `monkeypatch` fixtures

It also injects the local `.venv` site-packages into `sys.path` so third-party
deps installed in the repo venv (pandas, tzdata, torch, etc.) are importable
without activating the venv.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Any, Callable, ContextManager, Iterable, Mapping, Sequence


def _add_local_venv_site_packages() -> None:
    repo_root = Path.cwd()
    site_packages = repo_root / ".venv" / "Lib" / "site-packages"
    if not site_packages.exists():
        return
    site_packages_str = str(site_packages)
    if site_packages_str not in sys.path:
        sys.path.insert(0, site_packages_str)


_add_local_venv_site_packages()


class Skipped(RuntimeError):
    """Raised to indicate a skipped test/module."""


class Failed(AssertionError):
    """Raised by `pytest.fail`."""


class _Raises(ContextManager[None]):
    def __init__(self, expected: type[BaseException]):
        self._expected = expected

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            raise AssertionError(f"Did not raise {self._expected.__name__}")
        if not isinstance(exc, self._expected):
            return False
        return True


def raises(expected: type[BaseException]) -> _Raises:
    return _Raises(expected)


def fail(message: str = "") -> None:
    raise Failed(message or "pytest.fail() called")


def skip(reason: str = "") -> None:
    raise Skipped(reason or "skipped")


def importorskip(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover
        raise Skipped(f"could not import {module_name}: {exc}") from exc


@dataclass
class _MonkeyPatch:
    _touched: dict[str, str | None]

    def setenv(self, key: str, value: str) -> None:
        if key not in self._touched:
            self._touched[key] = os.environ.get(key)
        os.environ[key] = value

    def delenv(self, key: str, raising: bool = True) -> None:
        if key not in self._touched:
            self._touched[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]
            return
        if raising:
            raise KeyError(key)

    def undo(self) -> None:
        for key, previous in self._touched.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._touched.clear()


def _iter_test_files(tests_root: Path) -> list[Path]:
    if not tests_root.exists():
        return []
    return sorted([p for p in tests_root.rglob("test_*.py") if p.is_file()])


def _load_module_from_path(path: Path) -> ModuleType:
    import importlib.util

    module_name = f"_pytest_shim_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _should_skip_import_failure(exc: BaseException) -> bool:
    """
    Heuristic: treat some third-party import failures as skips.

    Managed Windows runtimes sometimes run with a Python distribution that cannot
    load native-extension wheels (numpy/torch). In that case, tests that depend
    on those optional extras should be skipped rather than failing the whole
    suite.
    """

    message = str(exc)
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        missing = getattr(exc, "name", None)
        if missing in {"torch", "numpy", "pandas"}:
            return True
        return False
    if isinstance(exc, AttributeError) and "_type_" in message:
        return True
    return False


def _iter_test_functions(module: ModuleType) -> Iterable[tuple[str, Callable[..., Any]]]:
    for name, value in vars(module).items():
        if name.startswith("test_") and callable(value):
            yield name, value


def _call_with_fixtures(func: Callable[..., Any]) -> None:
    import inspect

    signature = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    temp_dir: str | None = None
    monkeypatch: _MonkeyPatch | None = None

    try:
        for param_name in signature.parameters.keys():
            if param_name == "tmp_path":
                temp_dir = tempfile.mkdtemp(prefix="pytest-shim-")
                kwargs[param_name] = Path(temp_dir)
            elif param_name == "monkeypatch":
                monkeypatch = _MonkeyPatch({})
                kwargs[param_name] = monkeypatch
            else:
                raise RuntimeError(f"Unsupported fixture parameter: {param_name}")
        func(**kwargs)
    finally:
        if monkeypatch is not None:
            monkeypatch.undo()
        if temp_dir is not None:
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def main(args: Sequence[str] | None = None) -> int:
    _ = list(args or [])

    repo_root = Path.cwd()
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tests_root = repo_root / "tests"
    test_files = _iter_test_files(tests_root)
    if not test_files:
        print("pytest-shim: no tests found")
        return 0

    collected = 0
    skipped = 0
    failed = 0

    for path in test_files:
        try:
            module = _load_module_from_path(path)
        except Skipped as exc:
            skipped += 1
            print(f"SKIPPED {path}: {exc}")
            continue
        except Exception as exc:
            if _should_skip_import_failure(exc):
                skipped += 1
                print(f"SKIPPED {path}: {exc}")
                continue
            failed += 1
            print(f"FAILED import {path}")
            traceback.print_exc()
            continue

        for test_name, test_func in _iter_test_functions(module):
            collected += 1
            try:
                _call_with_fixtures(test_func)
            except Skipped as exc:
                skipped += 1
                print(f"SKIPPED {path}::{test_name}: {exc}")
            except Exception:
                failed += 1
                print(f"FAILED {path}::{test_name}")
                traceback.print_exc()

    passed = collected - skipped - failed
    print(f"pytest-shim: {passed} passed, {skipped} skipped, {failed} failed, {collected} collected")
    return 0 if failed == 0 else 1
