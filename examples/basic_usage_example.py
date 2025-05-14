# /home/ubuntu/xderived_project/examples/basic_usage_example.py

"""Basic usage example for the xderived plugin."""

import xarray as xr
import numpy as np
import sys
import os

# Add the parent directory of xderived to the Python path
# This allows importing xderived when running the script from the examples directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # This should be xderived_project/
sys.path.insert(0, project_root)

import xderived # This will register the accessor and standard variables
from xderived.core import DerivedVariable, registry
from xderived.utils import MissingDependencyError

def main():
    print("--- xderived Plugin: Basic Usage Example ---")

    # 1. Load a sample dataset
    print("\n1. Loading sample dataset (eraint_uvz)...")
    try:
        ds = xr.tutorial.load_dataset("eraint_uvz")
    except Exception as e:
        print(f"Could not load eraint_uvz dataset. Do you have pooch installed and network access? Error: {e}")
        print("Attempting to create a minimal dummy dataset for demonstration.")
        ds = xr.Dataset(
            {
                "air_temperature": (("level", "latitude", "longitude"), np.random.rand(2, 3, 4) * 30 + 273.15),
                "air_pressure": (("level", "latitude", "longitude"), (np.arange(1000, 500, -250)[:, np.newaxis, np.newaxis] + np.random.rand(2,3,4)*10) * 100),
                "eastward_wind": (("level", "latitude", "longitude"), np.random.rand(2, 3, 4) * 20 - 10),
                "northward_wind": (("level", "latitude", "longitude"), np.random.rand(2, 3, 4) * 20 - 10),
            },
            coords={
                "level": [750, 850],
                "latitude": [10, 20, 30],
                "longitude": [100, 110, 120, 130],
            },
        )
        ds["air_temperature"].attrs["units"] = "K"
        ds["air_pressure"].attrs["units"] = "Pa"
        ds["eastward_wind"].attrs["units"] = "m s-1"
        ds["northward_wind"].attrs["units"] = "m s-1"
        print("Using dummy dataset.")

    # Prepare the dataset: rename variables and convert units if necessary
    # For eraint_uvz:
    # 't' is temperature in K, 'level' is pressure level in hPa
    # 'u' is eastward_wind in m/s, 'v' is northward_wind in m/s
    if "t" in ds and "level" in ds.coords:
        print("Preparing eraint_uvz dataset for derived variable calculations...")
        ds = ds.rename({"t": "air_temperature", "u": "eastward_wind", "v": "northward_wind"})
        # Convert pressure levels (hPa) to pressure in Pascals (Pa)
        # This creates a DataArray for pressure that matches the dimensions of temperature
        ds["air_pressure"] = ds["level"] * 100.0
        ds["air_pressure"].attrs["units"] = "Pa"
        ds["air_pressure"].attrs["long_name"] = "Air Pressure"
        # Ensure air_temperature has units attribute if not already present
        if "units" not in ds["air_temperature"].attrs:
            ds["air_temperature"].attrs["units"] = "K"
        if "units" not in ds["eastward_wind"].attrs:
            ds["eastward_wind"].attrs["units"] = "m s-1"
        if "units" not in ds["northward_wind"].attrs:
            ds["northward_wind"].attrs["units"] = "m s-1"
        
        # For this example, let's take a slice to make it smaller
        ds = ds.isel(month=0, level=slice(0,2)) # Use first two levels
        print("Dataset prepared.")
    
    print("\nOriginal Dataset:")
    print(ds)

    # 2. Show the repr of the derived accessor for discoverability
    print("\n2. Discovering available derived variables via accessor repr:")
    print(ds.derived)

    # 3. Access a pre-defined derived variable: Potential Temperature
    print("\n3. Accessing Potential Temperature (pre-defined):")
    # Check if computable using the accessor's own logic (which get_status provides)
    if ds.derived.get_status("potential_temperature")["computable"]:
        potential_temp = ds.derived.potential_temperature
        print("Potential Temperature DataArray:")
        print(potential_temp)
        print(f"Potential Temperature computed values (sample):\n{potential_temp.isel(level=0, latitude=0, longitude=0).values}")
        if hasattr(potential_temp.data, "dask"): # Check if it's a Dask array
            print("Potential Temperature is Dask-backed, demonstrating lazy computation.")
            # print("Dask graph for potential_temperature:", potential_temp.data.dask) # This can be very verbose
    else:
        print("Potential Temperature is not available for this dataset according to get_status().")
        print(f"Status: {ds.derived.get_status('potential_temperature')}")
        print("Check dependencies: requires 'air_temperature' (K) and 'air_pressure' (Pa).")

    # 4. Access other pre-defined derived variables: Wind Speed and Direction
    print("\n4. Accessing Wind Speed and Direction (pre-defined):")
    if ds.derived.get_status("wind_speed")["computable"]:
        wind_s = ds.derived.wind_speed
        print("Wind Speed DataArray:")
        print(wind_s)
        print(f"Wind Speed computed values (sample):\n{wind_s.isel(level=0, latitude=0, longitude=0).values}")

        wind_d = ds.derived.wind_from_direction
        print("\nWind Direction DataArray:")
        print(wind_d)
        print(f"Wind Direction computed values (sample):\n{wind_d.isel(level=0, latitude=0, longitude=0).values}")
    else:
        print("Wind Speed/Direction are not available. Check dependencies: 'eastward_wind', 'northward_wind'.")
        print(f"Wind Speed Status: {ds.derived.get_status('wind_speed')}")

    # 5. Demonstrate error handling for missing dependencies
    print("\n5. Demonstrating error for missing dependencies:")
    missing_dep_var = DerivedVariable(
        name="test_missing_dep",
        dependencies=["non_existent_var", "air_temperature"],
        func=lambda ds_local: ds_local["air_temperature"], # Dummy func
        description="Test variable with a missing dependency."
    )
    is_temp_registered = False
    if registry.get_variable("test_missing_dep") is None:
        registry.register(missing_dep_var)
        is_temp_registered = True
    
    print("ds.derived repr after registering 'test_missing_dep':")
    print(ds.derived) # Should show test_missing_dep as unavailable

    try:
        print("Attempting to access 'test_missing_dep'...")
        _ = ds.derived.test_missing_dep
    except MissingDependencyError as e:
        print(f"Successfully caught MissingDependencyError: {e}")
    except AttributeError as e:
        print(f"Caught AttributeError (likely because it was filtered by __dir__ or not in available_vars): {e}")
    finally:
        if is_temp_registered:
            registry.unregister("test_missing_dep") # Clean up

    # 6. Demonstrate defining and registering a custom derived variable
    print("\n6. Defining and using a custom derived variable: Temperature in Celsius")

    def kelvin_to_celsius(ds_custom: xr.Dataset) -> xr.DataArray:
        temp_k = ds_custom["air_temperature"]
        temp_c = temp_k - 273.15
        temp_c.attrs = temp_k.attrs.copy()
        temp_c.attrs["units"] = "Celsius"
        temp_c.attrs["long_name"] = "Air Temperature in Celsius"
        temp_c.name = "air_temperature_celsius" # Important for the DataArray itself
        return temp_c

    temp_c_var = DerivedVariable(
        name="air_temperature_celsius",
        dependencies=["air_temperature"],
        func=kelvin_to_celsius,
        description="Air Temperature in Celsius from Kelvin.",
        attrs={"units": "Celsius", "long_name": "Air Temperature in Celsius"}
    )
    
    if registry.get_variable("air_temperature_celsius") is None:
        registry.register(temp_c_var)
        print("'air_temperature_celsius' custom variable registered.")
    else:
        print("'air_temperature_celsius' custom variable was already registered.")

    print("ds.derived repr after registering custom 'air_temperature_celsius':")
    print(ds.derived)

    if ds.derived.get_status("air_temperature_celsius")["computable"]:
        temp_c_da = ds.derived.air_temperature_celsius
        print("Custom Air Temperature in Celsius DataArray:")
        print(temp_c_da)
        print(f"Custom Air Temperature in Celsius (sample):\n{temp_c_da.isel(level=0, latitude=0, longitude=0).values}")
    else:
        print("Custom 'air_temperature_celsius' is not available. This shouldn't happen if 'air_temperature' exists.")

    print("\n--- Example Finished ---")

if __name__ == "__main__":
    main()

