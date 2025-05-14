# xderived/core.py

"""Core components for the xderived plugin: DerivedVariable and DerivedVariableRegistry."""

from typing import Callable, List, Dict, Optional, Any, Set, Tuple
import xarray as xr
import numpy as np # For dtype hinting
from .utils import RegistrationError, ComputationError

class DerivedVariable:
    """Represents a definition for a derived scientific variable."""
    def __init__(
        self,
        name: str,
        dependencies: List[str],
        func: Callable[[xr.Dataset], xr.DataArray],
        description: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        formula_str: Optional[str] = None,
        standard_name: Optional[str] = None,
        long_name: Optional[str] = None,
        # New fields for HTML repr hinting (Phase 3)
        output_dims_hint: Optional[Tuple[str, ...]] = None, # e.g., ("lat", "lon") or ("time", "level", "lat", "lon")
        output_dtype_hint: Optional[Any] = None # e.g., np.float64 or "float32"
    ):
        if not name or not isinstance(name, str):
            raise ValueError("DerivedVariable name must be a non-empty string.")
        if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
            raise ValueError("DerivedVariable dependencies must be a list of strings.")
        if not callable(func):
            raise ValueError("DerivedVariable func must be a callable.")

        self.name = name
        self.description = description or name
        self.dependencies = dependencies
        self.func = func
        self.attrs = attrs if attrs is not None else {}
        self.formula_str = formula_str
        self.standard_name = standard_name
        self.long_name = long_name

        self.output_dims_hint = output_dims_hint
        self.output_dtype_hint = output_dtype_hint

        if self.standard_name and "standard_name" not in self.attrs:
            self.attrs["standard_name"] = self.standard_name
        if self.long_name and "long_name" not in self.attrs:
            self.attrs["long_name"] = self.long_name
        if self.output_dtype_hint and "dtype" not in self.attrs: # Maybe add dtype to attrs if hinted
            try:
                self.attrs["_expected_dtype"] = str(np.dtype(self.output_dtype_hint))
            except TypeError:
                self.attrs["_expected_dtype"] = str(self.output_dtype_hint)

    def __repr__(self) -> str:
        return f"DerivedVariable(name=\"{self.name}\", dependencies={self.dependencies}, description=\"{self.description}\")"

class DerivedVariableRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DerivedVariableRegistry, cls).__new__(cls)
            cls._instance._registry: Dict[str, DerivedVariable] = {}
        return cls._instance

    def register(self, derived_var: DerivedVariable) -> None:
        if not isinstance(derived_var, DerivedVariable):
            raise RegistrationError("Only DerivedVariable instances can be registered.")
        if derived_var.name in self._registry:
            raise RegistrationError(f"DerivedVariable with name \"{derived_var.name}\" is already registered.")
        self._registry[derived_var.name] = derived_var

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            raise RegistrationError(f"No DerivedVariable with name \"{name}\" found to unregister.")
        del self._registry[name]

    def get_variable(self, name: str) -> Optional[DerivedVariable]:
        return self._registry.get(name)

    def list_all(self) -> List[DerivedVariable]:
        return list(self._registry.values())

    def _is_computable(self, var_name: str, ds: xr.Dataset, resolving_stack: Optional[Set[str]] = None) -> bool:
        if resolving_stack is None:
            resolving_stack = set()
        if var_name in resolving_stack:
            return False
        var_def = self.get_variable(var_name)
        if not var_def:
            return False
        resolving_stack.add(var_name)
        can_compute_all_deps = True
        for dep_name in var_def.dependencies:
            if dep_name in ds.variables or dep_name in ds.coords:
                continue
            elif self.get_variable(dep_name) is not None:
                if not self._is_computable(dep_name, ds, resolving_stack.copy()):
                    can_compute_all_deps = False
                    break
            else:
                can_compute_all_deps = False
                break
        resolving_stack.remove(var_name)
        return can_compute_all_deps

    def check_availability(self, dataset: xr.Dataset) -> Dict[str, DerivedVariable]:
        available_vars = {}
        for name, var_def in self._registry.items():
            if self._is_computable(name, dataset, resolving_stack=set()):
                available_vars[name] = var_def
        return available_vars

    def get_dependencies(self, variable_name: str, recursive: bool = False, ds: Optional[xr.Dataset] = None, _resolving_stack: Optional[Set[str]] = None) -> Dict[str, Any]:
        if _resolving_stack is None:
            _resolving_stack = set()
        if variable_name in _resolving_stack:
            return {"status": "cycle_detected"}
        var_def = self.get_variable(variable_name)
        if not var_def:
            raise RegistrationError(f"DerivedVariable with name \"{variable_name}\" not found.")
        _resolving_stack.add(variable_name)
        dependencies_info: Dict[str, Any] = {}
        for dep_name in var_def.dependencies:
            is_in_ds = ds and (dep_name in ds.variables or dep_name in ds.coords)
            if is_in_ds:
                dependencies_info[dep_name] = {"status": "base_variable_in_dataset"}
            elif self.get_variable(dep_name):
                if recursive:
                    dependencies_info[dep_name] = {
                        "status": "derived_variable",
                        "dependencies": self.get_dependencies(dep_name, recursive=True, ds=ds, _resolving_stack=_resolving_stack.copy())
                    }
                else:
                    dependencies_info[dep_name] = {"status": "derived_variable"}
            elif ds:
                 dependencies_info[dep_name] = {"status": "missing_base_variable"}
            else:
                 dependencies_info[dep_name] = {"status": "unknown_or_missing_base_variable"}
        _resolving_stack.remove(variable_name)
        return {
            "name": variable_name,
            "description": var_def.description,
            "dependencies": dependencies_info
        }

    def get_metadata(self, variable_name: str) -> Optional[Dict[str, Any]]:
        var_def = self.get_variable(variable_name)
        if not var_def:
            return None
        return {
            "name": var_def.name,
            "dependencies": var_def.dependencies,
            "description": var_def.description,
            "attrs": var_def.attrs,
            "formula_str": var_def.formula_str,
            "standard_name": var_def.standard_name,
            "long_name": var_def.long_name,
            "output_dims_hint": var_def.output_dims_hint,
            "output_dtype_hint": str(np.dtype(var_def.output_dtype_hint)) if var_def.output_dtype_hint else None
        }

    def list_all_computable(self, ds: xr.Dataset) -> List[str]:
        computable_vars = []
        for name in self._registry.keys():
            if self._is_computable(name, ds, resolving_stack=set()):
                computable_vars.append(name)
        return sorted(computable_vars)

    def search_variables(self, keyword: str, search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if search_fields is None:
            search_fields = ["name", "description", "standard_name", "long_name", "formula_str"]
        matches = []
        keyword_lower = keyword.lower()
        for var_def in self.list_all():
            for field_name in search_fields:
                field_value = getattr(var_def, field_name, None)
                if field_value and isinstance(field_value, str) and keyword_lower in field_value.lower():
                    meta = self.get_metadata(var_def.name)
                    if meta: matches.append(meta)
                    break
        return matches

    def clear(self) -> None:
        self._registry.clear()

registry = DerivedVariableRegistry()

