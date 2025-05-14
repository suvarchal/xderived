# /home/ubuntu/xarray-derived-variables/examples/test_notebook_repr_example.py

"""Example script to test the custom HTML representation in a simulated notebook environment."""

import xarray as xr
import numpy as np
import dask.array as da

# Ensure the plugin is imported and patches are applied
import xarray_derived # This should apply the patch
from xarray_derived import registry, DerivedVariable, config

def main():
    print("--- Testing Notebook HTML Representation for Derived Variables ---")

    # Clear any existing registrations for a clean test
    registry.clear()

    # Define some derived variables
    # 1. Simple one, always computable if T is present
    registry.register(DerivedVariable(
        name="temp_kelvin_plus_10",
        dependencies=["air_temperature"],
        func=lambda ds: ds["air_temperature"] + 10,
        description="Air temperature in Kelvin + 10K.",
        output_dims_hint=("lat", "lon"),
        output_dtype_hint=np.float32
    ))

    # 2. One that depends on a missing variable
    registry.register(DerivedVariable(
        name="derived_with_missing_base",
        dependencies=["non_existent_var"],
        func=lambda ds: ds["non_existent_var"] * 2, # func won_t be called
        description="Depends on a missing base variable.",
        output_dims_hint=("lat",), 
        output_dtype_hint=np.float64
    ))

    # 3. One that depends on another derived variable (chaining)
    registry.register(DerivedVariable(
        name="temp_kelvin_plus_20",
        dependencies=["temp_kelvin_plus_10"],
        func=lambda ds: ds["temp_kelvin_plus_10"] + 10,
        description="Air temperature in Kelvin + 20K (chained).",
        output_dims_hint=("lat", "lon"),
        output_dtype_hint=np.float32
    ))
    
    # 4. A derived variable that might be missing a derived dependency
    registry.register(DerivedVariable(
        name="derived_missing_derived_dep",
        dependencies=["derived_with_missing_base"], # This one is not computable
        func=lambda ds: ds["derived_with_missing_base"] + 5,
        description="Depends on an uncomputable derived variable.",
        output_dims_hint=("lat",), 
        output_dtype_hint=np.float64
    ))

    # --- Test Case 1: Eager (NumPy-backed) Dataset ---
    print("\n--- Test Case 1: Eager (NumPy-backed) Dataset ---")
    ds_numpy = xr.Dataset(
        {
            "air_temperature": (("lat", "lon"), np.array([[280, 285], [290, 295]], dtype=np.float32)),
        },
        coords={
            "lat": [0, 1],
            "lon": [10, 20]
        }
    )
    print("Dataset created. Generating HTML repr...")
    html_output_numpy = ds_numpy._repr_html_()
    print("\nHTML Output (NumPy ds, show_computable_only=False - default):")
    print(html_output_numpy)
    # Check if derived variables are computed (they should not be in cache yet)
    print(f"Cache after NumPy HTML repr: {ds_numpy.derived._cache}")
    assert not ds_numpy.derived._cache, "Cache should be empty after HTML repr generation for NumPy ds"

    # Test with config option
    config.config["repr_show_computable_only"] = True
    print("\nHTML Output (NumPy ds, show_computable_only=True):")
    html_output_numpy_computable_only = ds_numpy._repr_html_()
    print(html_output_numpy_computable_only)
    config.config["repr_show_computable_only"] = False # Reset config
    print(f"Cache after NumPy HTML repr (computable_only=True): {ds_numpy.derived._cache}")
    assert not ds_numpy.derived._cache, "Cache should be empty after HTML repr generation for NumPy ds (computable_only=True)"


    # --- Test Case 2: Lazy (Dask-backed) Dataset ---
    print("\n\n--- Test Case 2: Lazy (Dask-backed) Dataset ---")
    ds_dask = xr.Dataset(
        {
            "air_temperature": (("lat", "lon"), da.from_array(np.array([[280, 285], [290, 295]], dtype=np.float32), chunks=(1, 1))),
        },
        coords={
            "lat": [0, 1],
            "lon": [10, 20]
        }
    )
    print("Dask Dataset created. Generating HTML repr...")
    html_output_dask = ds_dask._repr_html_()
    print("\nHTML Output (Dask ds, show_computable_only=False - default):")
    print(html_output_dask)
    print(f"Cache after Dask HTML repr: {ds_dask.derived._cache}")
    assert not ds_dask.derived._cache, "Cache should be empty after HTML repr generation for Dask ds"

    # Test with config option for Dask
    config.config["repr_show_computable_only"] = True
    print("\nHTML Output (Dask ds, show_computable_only=True):")
    html_output_dask_computable_only = ds_dask._repr_html_()
    print(html_output_dask_computable_only)
    config.config["repr_show_computable_only"] = False # Reset config
    print(f"Cache after Dask HTML repr (computable_only=True): {ds_dask.derived._cache}")
    assert not ds_dask.derived._cache, "Cache should be empty after HTML repr generation for Dask ds (computable_only=True)"

    # --- Test Case 3: Accessing a derived variable to see if it computes ---
    print("\n\n--- Test Case 3: Accessing a derived variable ---")
    print("Accessing ds_dask.derived.temp_kelvin_plus_10...")
    computed_var = ds_dask.derived.temp_kelvin_plus_10
    print(f"Computed var:\n{computed_var}")
    assert "temp_kelvin_plus_10" in ds_dask.derived._cache, "temp_kelvin_plus_10 should be in cache after access"
    assert isinstance(computed_var.data, da.Array), "Computed variable from Dask input should be Dask array"
    print("Accessing ds_dask.derived.temp_kelvin_plus_20 (chained)...")
    computed_var_chained = ds_dask.derived.temp_kelvin_plus_20
    print(f"Computed chained var:\n{computed_var_chained}")
    assert "temp_kelvin_plus_20" in ds_dask.derived._cache, "temp_kelvin_plus_20 should be in cache after access"
    assert isinstance(computed_var_chained.data, da.Array), "Chained computed variable from Dask input should be Dask array"

    print("\n\n--- Test completed. Review the HTML output above. ---")

if __name__ == "__main__":
    main()


