# /home/ubuntu/xderived_project/examples/chained_discovery_example.py

"""Example script demonstrating chained derived variable computations and discovery features for the xderived plugin."""

import xarray as xr
import numpy as np
import os
import sys

# Add project root to sys.path to allow importing xderived
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # This should be xderived_project/
sys.path.insert(0, project_root)

import xderived # This import registers the accessor and standard variables
from xderived.core import registry, DerivedVariable, RegistrationError # For direct interaction if needed

def create_sample_dataset_for_chaining():
    """Creates a sample dataset with variables needed for chained calculations."""
    ds = xr.Dataset(
        {
            "air_temperature": (("level", "lat", "lon"), np.array([[[290., 292.], [295., 297.]], [[280., 282.], [285., 287.]]], dtype=np.float32)), # K
            "air_pressure": (("level", "lat", "lon"), np.array([[[100000., 99000.], [98000., 97000.]], [[85000., 84000.], [83000., 82000.]]], dtype=np.float32)),    # Pa
            "specific_humidity": (("level", "lat", "lon"), np.array([[[0.01, 0.012], [0.008, 0.009]], [[0.005, 0.006], [0.003, 0.004]]], dtype=np.float32)), # kg/kg
            "eastward_wind": (("level", "lat", "lon"), np.array([[[5., 10.], [-5., -10.]], [[8., 12.], [-8., -12.]]], dtype=np.float32)),      # m/s
            "northward_wind": (("level", "lat", "lon"), np.array([[[2., -2.], [3., -3.]], [[4., -4.], [6., -6.]]], dtype=np.float32)),     # m/s
        },
        coords={
            "level": [1000, 850],      # hPa (though air_pressure is Pa)
            "lat": [30, 40],
            "lon": [-100, -90],
        }
    )
    ds["air_temperature"].attrs["units"] = "K"
    ds["air_pressure"].attrs["units"] = "Pa"
    ds["specific_humidity"].attrs["units"] = "kg kg-1"
    ds["eastward_wind"].attrs["units"] = "m s-1"
    ds["northward_wind"].attrs["units"] = "m s-1"
    return ds

if __name__ == "__main__":
    print("--- xderived Plugin: Chained Discovery Example ---")
    print("--- Creating Sample Dataset ---")
    ds = create_sample_dataset_for_chaining()
    print(ds)
    print("\n")

    print("--- Accessor Representation (shows computability) ---")
    print(ds.derived)
    print("\n")

    print("--- Discovery: List Computable Variables ---")
    computable_vars = ds.derived.list_computable()
    print(f"Computable derived variables: {computable_vars}")
    assert "relative_humidity_from_mixing_ratios" in computable_vars
    assert "equivalent_potential_temperature_approx" in computable_vars
    print("\n")

    print("--- Discovery: Get Metadata for a Variable (e.g., relative_humidity_from_mixing_ratios) ---")
    rh_metadata = ds.derived.get_metadata("relative_humidity_from_mixing_ratios")
    if rh_metadata:
        print(f"Metadata for relative_humidity_from_mixing_ratios:")
        for key, value in rh_metadata.items():
            print(f"  {key}: {value}")
    else:
        print("Could not retrieve metadata for relative_humidity_from_mixing_ratios")
    print("\n")

    print("--- Discovery: Get Dependencies (e.g., relative_humidity_from_mixing_ratios, recursive) ---")
    rh_deps_recursive = ds.derived.get_dependencies("relative_humidity_from_mixing_ratios", recursive=True)
    print(f"Recursive dependencies for relative_humidity_from_mixing_ratios:")
    import json
    print(json.dumps(rh_deps_recursive, indent=2))
    print("\n")
    
    print("--- Discovery: Get Dependencies (e.g., equivalent_potential_temperature_approx, recursive) ---")
    theta_e_deps_recursive = ds.derived.get_dependencies("equivalent_potential_temperature_approx", recursive=True)
    print(f"Recursive dependencies for equivalent_potential_temperature_approx:")
    print(json.dumps(theta_e_deps_recursive, indent=2))
    print("\n")

    print("--- Discovery: Search Variables (keyword: 'humidity') ---")
    humidity_search = ds.derived.search_variables("humidity")
    print(f"Search results for 'humidity':")
    for item in humidity_search:
        print(f"  - Name: {item.get('name')}, Description: {item.get('description')}")
    print("\n")
    
    print("--- Discovery: Search Variables (keyword: 'potential') ---")
    potential_search = ds.derived.search_variables("potential")
    print(f"Search results for 'potential':")
    for item in potential_search:
        print(f"  - Name: {item.get('name')}, Description: {item.get('description')}")
    print("\n")

    print("--- Computation: Chained Derived Variable (Relative Humidity) ---")
    try:
        rh = ds.derived.relative_humidity_from_mixing_ratios
        print("Computed Relative Humidity (from mixing ratios):")
        print(rh)
        print(f"Units: {rh.attrs.get('units')}")
        assert rh.attrs.get("units") == "%"
    except Exception as e:
        print(f"Error computing relative_humidity_from_mixing_ratios: {e}")
    print("\n")

    print("--- Computation: Chained Derived Variable (Equivalent Potential Temperature Approx) ---")
    try:
        theta_e = ds.derived.equivalent_potential_temperature_approx
        print("Computed Equivalent Potential Temperature (Approximate):")
        print(theta_e)
        print(f"Units: {theta_e.attrs.get('units')}")
        assert theta_e.attrs.get("units") == "K"
    except Exception as e:
        print(f"Error computing equivalent_potential_temperature_approx: {e}")
    print("\n")

    print("--- Testing a variable that should be unavailable (if a base var is dropped) ---")
    ds_missing_temp = ds.drop_vars(["air_temperature"])
    print("Accessor repr for dataset missing air_temperature:")
    print(ds_missing_temp.derived)
    print(f"Computable from ds_missing_temp: {ds_missing_temp.derived.list_computable()}")
    try:
        print("Attempting to compute potential_temperature from ds_missing_temp...")
        pt_missing = ds_missing_temp.derived.potential_temperature
        print(pt_missing) # Should not reach here
    except Exception as e:
        print(f"Successfully caught error: {e}")
    print("\n")
    
    print("--- Testing cycle detection (Example - Manual Registration for Test) ---")
    # Clean up any previous test registration if this script is run multiple times in one session
    try: registry.unregister("cycle_var_a")
    except RegistrationError: pass
    try: registry.unregister("cycle_var_b")
    except RegistrationError: pass

    def func_a(ds_input): return ds_input["cycle_var_b"]
    def func_b(ds_input): return ds_input["cycle_var_a"]
    
    registry.register(DerivedVariable(name="cycle_var_a", dependencies=["cycle_var_b"], func=func_a, description="Cycle A"))
    registry.register(DerivedVariable(name="cycle_var_b", dependencies=["cycle_var_a"], func=func_b, description="Cycle B"))
    
    print("Accessor repr with cyclic variables registered:")
    print(ds.derived) # Should show them as unavailable due to cycle
    # Using internal _is_computable for direct check, normally use get_status
    print(f"Is cycle_var_a computable (direct check)? {registry._is_computable('cycle_var_a', ds, resolving_stack=set())}") 

    try:
        print("Attempting to compute cycle_var_a...")
        _ = ds.derived.cycle_var_a
    except Exception as e:
        print(f"Successfully caught error for cycle_var_a: {e}")
        assert "Circular dependency detected" in str(e)

    # Clean up test cycle variables
    registry.unregister("cycle_var_a")
    registry.unregister("cycle_var_b")
    print("--- Example Script Finished ---")

