import xarray as xr
import numpy as np
import dask.array as da
import xderived # This should register the .derived accessor and standard variables

# Create a sample dataset (Dask-backed)
ds_dask = xr.Dataset(
    {
        "air_temperature": (("time", "lat"), da.from_array(np.array([[280., 285.], [290., 295.]]), chunks=(1,1)), {"units": "K"}),
        "air_pressure": (("time",), np.array([100000., 95000.]), {"units": "Pa"}),
        "specific_humidity": (("time", "lat"), da.from_array(np.array([[0.005, 0.006], [0.007, 0.008]]), chunks=(1,1)), {"units": "kg/kg"}),
        "eastward_wind": (("time", "lat"), da.from_array(np.array([[5., 6.], [7., 8.]]), chunks=(1,1)), {"units": "m/s"}),
        # Missing northward_wind for wind_from_direction
    },
    coords={"time": np.arange(2), "lat": [30, 40]}
)

# Create a sample dataset (NumPy-backed)
ds_numpy = xr.Dataset(
    {
        "air_temperature": (("time", "lat"), np.array([[280., 285.], [290., 295.]]), {"units": "K"}),
        "air_pressure": (("time",), np.array([100000., 95000.]), {"units": "Pa"}),
        # Missing specific_humidity for relative_humidity_from_mixing_ratios
        "eastward_wind": (("time", "lat"), np.array([[5., 6.], [7., 8.]]), {"units": "m/s"}),
        "northward_wind": (("time", "lat"), np.array([[1., 2.], [3., 4.]]), {"units": "m/s"}),
    },
    coords={"time": np.arange(2), "lat": [30, 40]}
)

print("--- HTML representation for ds_dask.derived ---")
html_output_dask = ds_dask.derived._repr_html_()
print(html_output_dask)

print("\n--- HTML representation for ds_numpy.derived ---")
html_output_numpy = ds_numpy.derived._repr_html_()
print(html_output_numpy)

print("\n--- Testing config option: repr_show_computable_only = True ---")
xderived.config.config["repr_show_computable_only"] = True
html_output_numpy_computable_only = ds_numpy.derived._repr_html_()
print(html_output_numpy_computable_only)

# Reset config for other tests/examples
xderived.config.config["repr_show_computable_only"] = False

print("\n--- Text representation for ds_dask.derived (for comparison) ---")
print(ds_dask.derived)

print("\n--- Text representation for ds_numpy.derived (for comparison) ---")
print(ds_numpy.derived)

