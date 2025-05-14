# /home/ubuntu/xderived_project/tests/test_derived_variables.py

import pytest
import xarray as xr
import numpy as np
import os
import sys
import re # For escaping regex special characters if needed

# Add project root to sys.path to allow importing xderived
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # This should be xderived_project/
sys.path.insert(0, project_root)

from xderived.core import DerivedVariable, registry, DerivedVariableRegistry
from xderived.utils import MissingDependencyError, ComputationError, RegistrationError
import xderived # This import registers the accessor and standard variables

# --- Fixtures for test data ---
@pytest.fixture
def sample_dataset_base():
    """A base dataset with common meteorological variables."""
    ds = xr.Dataset(
        {
            "air_temperature": (("level", "lat", "lon"), np.array([[[280., 282.], [285., 287.]], [[270., 272.], [275., 277.]]], dtype=np.float32)), # K
            "air_pressure": (("level", "lat", "lon"), np.array([[[100000., 99000.], [98000., 97000.]], [[85000., 84000.], [83000., 82000.]]], dtype=np.float32)),    # Pa
            "eastward_wind": (("level", "lat", "lon"), np.array([[[5., 10.], [-5., -10.]], [[8., 12.], [-8., -12.]]], dtype=np.float32)),      # m/s
            "northward_wind": (("level", "lat", "lon"), np.array([[[2., -2.], [3., -3.]], [[4., -4.], [6., -6.]]], dtype=np.float32)),     # m/s
            "specific_humidity": (("level", "lat", "lon"), np.array([[[0.005, 0.006], [0.007, 0.008]], [[0.003, 0.004], [0.002, 0.001]]], dtype=np.float32)) # kg/kg
        },
        coords={
            "level": [1000, 850],      # hPa (though air_pressure is Pa)
            "lat": [30, 40],
            "lon": [-100, -90],
        }
    )
    ds["air_temperature"].attrs["units"] = "K"
    ds["air_pressure"].attrs["units"] = "Pa"
    ds["eastward_wind"].attrs["units"] = "m s-1"
    ds["northward_wind"].attrs["units"] = "m s-1"
    ds["specific_humidity"].attrs["units"] = "kg kg-1"
    return ds

@pytest.fixture
def dask_dataset(sample_dataset_base):
    """Converts the sample dataset to use Dask arrays."""
    return sample_dataset_base.chunk({"lat": 1, "lon": 1, "level": 1})

@pytest.fixture(autouse=True)
def clear_registry_before_each_test():
    """Ensures a clean registry for each test and re-registers standard variables."""
    registry.clear()
    # Assuming standard_variables is now part of the xderived package
    xderived.standard_variables.register_standard_variables() # Updated from register_standard_variables_updated
    yield # Test runs here
    registry.clear() # Clean up after test

# --- Tests for DerivedVariable and Registry ---

def test_derived_variable_creation():
    def dummy_func(ds): return ds["air_temperature"]
    dv = DerivedVariable("test_var", ["air_temperature"], dummy_func, "A test var", {"units": "K"})
    assert dv.name == "test_var"
    assert dv.dependencies == ["air_temperature"]
    assert dv.description == "A test var"
    assert dv.attrs == {"units": "K"}

def test_registry_singleton():
    reg1 = DerivedVariableRegistry()
    reg2 = DerivedVariableRegistry()
    assert reg1 is reg2
    assert registry is reg1 # Global instance

def test_register_and_get_variable():
    def dummy_func(ds): return ds["air_temperature"]
    dv = DerivedVariable("custom_temp", ["air_temperature"], dummy_func)
    registry.register(dv)
    retrieved_dv = registry.get_variable("custom_temp")
    assert retrieved_dv is dv

def test_register_duplicate_error():
    def dummy_func(ds): return ds["air_temperature"]
    dv1 = DerivedVariable("duplicate_var", ["air_temperature"], dummy_func)
    registry.register(dv1)
    expected_message = "DerivedVariable with name \"duplicate_var\" is already registered."
    with pytest.raises(RegistrationError, match=re.escape(expected_message)):
        registry.register(dv1) 
    dv2 = DerivedVariable("duplicate_var", ["air_pressure"], dummy_func) 
    with pytest.raises(RegistrationError, match=re.escape(expected_message)):
        registry.register(dv2)

def test_unregister_variable():
    def dummy_func(ds): return ds["air_temperature"]
    dv = DerivedVariable("to_unregister", ["air_temperature"], dummy_func)
    registry.register(dv)
    assert registry.get_variable("to_unregister") is not None
    registry.unregister("to_unregister")
    assert registry.get_variable("to_unregister") is None

def test_unregister_nonexistent_error():
    expected_message = "No DerivedVariable with name \"non_existent_var\" found to unregister."
    with pytest.raises(RegistrationError, match=re.escape(expected_message)):
        registry.unregister("non_existent_var")

def test_check_availability(sample_dataset_base):
    # Using the accessor method which internally uses registry.check_availability
    # and provides a more user-facing way to check
    available_vars = sample_dataset_base.derived.available_variables()
    assert "potential_temperature" in available_vars
    assert "wind_speed" in available_vars
    assert "wind_from_direction" in available_vars

    ds_missing_pressure = sample_dataset_base.drop_vars(["air_pressure"])
    available_missing = ds_missing_pressure.derived.available_variables()
    assert "potential_temperature" not in available_missing
    assert "wind_speed" in available_missing

# --- Tests for Accessor ---

def test_accessor_repr(sample_dataset_base):
    repr_str = repr(sample_dataset_base.derived)
    assert "xderived Accessor" in repr_str # Updated name
    assert "potential_temperature: Air Potential Temperature" in repr_str
    assert "(computable)" in repr_str # Updated status string
    
    def dummy_func_needs_foo(ds): return ds["foo"]
    needs_foo_var_name = "needs_foo_for_repr_test"
    if registry.get_variable(needs_foo_var_name) is None:
        registry.register(DerivedVariable(needs_foo_var_name, ["foo"], dummy_func_needs_foo))
    
    repr_str_with_unavailable = repr(sample_dataset_base.derived)
    assert needs_foo_var_name in repr_str_with_unavailable
    expected_match_in_repr = r"\(unavailable \(missing: \['foo'\]\)\)" # Updated status string
    assert re.search(expected_match_in_repr, repr_str_with_unavailable) is not None
    if registry.get_variable(needs_foo_var_name) is not None:
         registry.unregister(needs_foo_var_name)

def test_accessor_dir(sample_dataset_base):
    dir_list = dir(sample_dataset_base.derived)
    assert "potential_temperature" in dir_list
    assert "wind_speed" in dir_list
    assert "available_variables" in dir_list # Method from accessor
    assert "list_computable" in dir_list # Method from accessor

def test_accessor_getattr_and_getitem(sample_dataset_base):
    pt = sample_dataset_base.derived.potential_temperature
    assert isinstance(pt, xr.DataArray)
    assert pt.name == "potential_temperature"
    pt_item = sample_dataset_base.derived["potential_temperature"]
    assert isinstance(pt_item, xr.DataArray)
    xr.testing.assert_identical(pt, pt_item)

def test_accessor_caching(sample_dataset_base):
    accessor = sample_dataset_base.derived
    pt1 = accessor.potential_temperature
    pt2 = accessor.potential_temperature 
    assert pt1 is pt2 
    accessor.clear_cache()
    pt3 = accessor.potential_temperature 
    assert pt1 is not pt3 

def test_accessor_missing_dependency_error(sample_dataset_base):
    ds_missing = sample_dataset_base.drop_vars(["air_pressure"])
    # Match the exact error message from the accessor
    expected_pattern_fragment = "Derived variable \"potential_temperature\" cannot be computed. Missing dependencies: [\'air_pressure\']"
    with pytest.raises(MissingDependencyError, match=re.escape(expected_pattern_fragment)):
        _ = ds_missing.derived.potential_temperature

def test_accessor_attribute_error_nonexistent(sample_dataset_base):
    expected_pattern_fragment = "No derived variable named \"nonexistent_var\" is registered or available for this dataset."
    with pytest.raises(AttributeError, match=re.escape(expected_pattern_fragment)):
        _ = sample_dataset_base.derived.nonexistent_var

def test_computation_error_in_derived_func(sample_dataset_base):
    def failing_func(ds): raise ValueError("Intentional failure")
    failing_var_name = "failing_var_for_test"
    if registry.get_variable(failing_var_name) is None:
        registry.register(DerivedVariable(failing_var_name, ["air_temperature"], failing_func))
    
    expected_pattern_fragment = "Error computing derived variable \"failing_var_for_test\": Intentional failure"
    with pytest.raises(ComputationError, match=re.escape(expected_pattern_fragment)):
        _ = sample_dataset_base.derived.failing_var_for_test
    if registry.get_variable(failing_var_name) is not None:
        registry.unregister(failing_var_name)

# --- Tests for Standard Derived Variables ---

def test_potential_temperature_calculation(sample_dataset_base):
    pt = sample_dataset_base.derived.potential_temperature
    assert pt.attrs["units"] == "K"
    assert pt.attrs["standard_name"] == "air_potential_temperature"
    np.testing.assert_allclose(pt.isel(level=0, lat=0, lon=0).item(), 280.0, rtol=1e-5)
    expected_val_l1 = 270.0 * (100000.0 / 85000.0)**(287.058 / 1005.0)
    np.testing.assert_allclose(pt.isel(level=1, lat=0, lon=0).item(), expected_val_l1, rtol=1e-5)

def test_wind_speed_calculation(sample_dataset_base):
    ws = sample_dataset_base.derived.wind_speed
    assert ws.attrs["units"] == "m s-1"
    assert ws.attrs["standard_name"] == "wind_speed"
    np.testing.assert_allclose(ws.isel(level=0, lat=0, lon=0).item(), np.sqrt(5**2 + 2**2), rtol=1e-5)
    np.testing.assert_allclose(ws.isel(level=0, lat=1, lon=0).item(), np.sqrt((-5)**2 + 3**2), rtol=1e-5)

def test_wind_direction_calculation(sample_dataset_base):
    wd = sample_dataset_base.derived.wind_from_direction
    assert wd.attrs["units"] == "degree"
    assert wd.attrs["standard_name"] == "wind_from_direction"
    expected_wd_val1 = 248.19859 
    np.testing.assert_allclose(wd.isel(level=0, lat=0, lon=0).item(), expected_wd_val1, rtol=1e-4)
    expected_wd_val2 = 120.963745 
    np.testing.assert_allclose(wd.isel(level=0, lat=1, lon=0).item(), expected_wd_val2, rtol=1e-4)

def test_dask_integration(dask_dataset):
    assert hasattr(dask_dataset["air_temperature"].data, "dask")
    pt_dask = dask_dataset.derived.potential_temperature
    assert hasattr(pt_dask.data, "dask"), "Potential Temperature should be Dask-backed"
    ws_dask = dask_dataset.derived.wind_speed
    assert hasattr(ws_dask.data, "dask"), "Wind Speed should be Dask-backed"
    assert len(pt_dask.data.dask) > 0
    computed_val = pt_dask.isel(level=0, lat=0, lon=0).compute()
    assert isinstance(computed_val, xr.DataArray)
    assert not hasattr(computed_val.data, "dask")
    np.testing.assert_allclose(computed_val.item(), 280.0, rtol=1e-5)

def test_custom_variable_registration_and_use(sample_dataset_base):
    def celsius_func(ds):
        temp_c = ds["air_temperature"] - 273.15
        temp_c.attrs["units"] = "degC"
        temp_c.name = "air_temp_celsius"
        return temp_c

    celsius_var_name = "air_temp_celsius_for_test"
    celsius_var = DerivedVariable(
        name=celsius_var_name,
        dependencies=["air_temperature"],
        func=celsius_func,
        description="Air Temperature in Celsius",
        attrs={"units": "degC"}
    )
    if registry.get_variable(celsius_var_name) is None:
        registry.register(celsius_var)

    assert celsius_var_name in sample_dataset_base.derived.available_variables()
    temp_c_da = getattr(sample_dataset_base.derived, celsius_var_name)
    assert isinstance(temp_c_da, xr.DataArray)
    assert temp_c_da.attrs["units"] == "degC"
    expected_celsius_val = sample_dataset_base["air_temperature"].isel(level=0, lat=0, lon=0).item() - 273.15
    np.testing.assert_allclose(temp_c_da.isel(level=0, lat=0, lon=0).item(), expected_celsius_val, rtol=1e-5)

    repr_str = repr(sample_dataset_base.derived)
    assert celsius_var_name + ": Air Temperature in Celsius" in repr_str
    assert "(computable)" in repr_str # Updated status string
    
    if registry.get_variable(celsius_var_name) is not None:
        registry.unregister(celsius_var_name)

# --- Tests for Chained Variables and Discovery (from Phase 2) ---

def test_chained_relative_humidity(sample_dataset_base):
    # Ensure standard_variables (which includes chained ones) are registered
    # This is handled by the autouse fixture clear_registry_before_each_test
    assert "relative_humidity_from_mixing_ratios" in sample_dataset_base.derived.available_variables()
    rh = sample_dataset_base.derived.relative_humidity_from_mixing_ratios
    assert isinstance(rh, xr.DataArray)
    assert rh.attrs.get("units") == "%"
    # Basic check, not validating exact values here, just that it runs and has expected attrs
    assert rh.shape == sample_dataset_base["specific_humidity"].shape

def test_chained_equivalent_potential_temperature(sample_dataset_base):
    assert "equivalent_potential_temperature_approx" in sample_dataset_base.derived.available_variables()
    theta_e = sample_dataset_base.derived.equivalent_potential_temperature_approx
    assert isinstance(theta_e, xr.DataArray)
    assert theta_e.attrs.get("units") == "K"
    assert theta_e.shape == sample_dataset_base["air_temperature"].shape

def test_discovery_methods(sample_dataset_base):
    computable = sample_dataset_base.derived.list_computable()
    assert "potential_temperature" in computable
    assert "relative_humidity_from_mixing_ratios" in computable

    metadata = sample_dataset_base.derived.get_metadata("potential_temperature")
    assert metadata["name"] == "potential_temperature"
    assert "Air Potential Temperature" in metadata["description"]

    deps = sample_dataset_base.derived.get_dependencies("relative_humidity_from_mixing_ratios", recursive=True)
    assert "mixing_ratio_from_specific_humidity" in deps
    assert "saturation_mixing_ratio" in deps["mixing_ratio_from_specific_humidity"]["dependencies"] # Example of deeper check

    search_results = sample_dataset_base.derived.search_variables("temperature")
    assert any(item["name"] == "potential_temperature" for item in search_results)
    assert any(item["name"] == "equivalent_potential_temperature_approx" for item in search_results)

def test_cycle_detection_in_accessor(sample_dataset_base):
    # Setup a cycle
    def func_a(ds_input): return ds_input["cycle_var_b"]
    def func_b(ds_input): return ds_input["cycle_var_a"]
    
    registry.register(DerivedVariable(name="cycle_var_a", dependencies=["cycle_var_b"], func=func_a, description="Cycle A"))
    registry.register(DerivedVariable(name="cycle_var_b", dependencies=["cycle_var_a"], func=func_b, description="Cycle B"))

    status_a = sample_dataset_base.derived.get_status("cycle_var_a")
    assert not status_a["computable"]
    assert "cycle detected" in status_a.get("reason", "").lower()

    with pytest.raises(ComputationError, match="Circular dependency detected for variable cycle_var_a"):
        _ = sample_dataset_base.derived.cycle_var_a

    # Clean up
    registry.unregister("cycle_var_a")
    registry.unregister("cycle_var_b")

