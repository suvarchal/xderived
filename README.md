# xderived: Xarray Plugin for Derived Variables

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`xderived` is an [xarray](http://xarray.pydata.org/) plugin that allows users to dynamically register, discover, and lazily compute derived scientific variables based on the contents of their xarray Datasets. This functionality is inspired by concepts from VisAD and Unidata's Integrated Data Viewer (IDV), aiming to make common (and custom) derived quantity calculations more accessible and integrated within the xarray ecosystem.
![derived_data_demo](https://github.com/user-attachments/assets/9ffbf70f-7e17-49e2-8551-cf8257b85416)

## Features

- **Dynamic Registration:** Define derived variables (e.g., potential temperature, wind speed) with their dependencies, computation functions, and rich metadata (description, units, formula string, standard name, long name, output dimension/dtype hints).
- **Chainable Derived Variables:** Define derived variables that depend on other derived variables, allowing for complex, multi-step computations. The plugin automatically resolves and computes the full dependency chain.
- **Lazy Computation:** Derived variables are only computed when their data is explicitly accessed, leveraging xarray and Dask for efficiency.
- **Enhanced Discoverability:**
    - Use the `dataset.derived` accessor to see which derived variables are computable from a given dataset directly in its `repr`.
    - `dataset.derived.list_computable()`: Get a list of all derived variables that can be computed.
    - `dataset.derived.get_metadata("variable_name")`: Retrieve detailed metadata for a specific derived variable.
    - `dataset.derived.get_dependencies("variable_name", recursive=True)`: Explore the full dependency tree of a derived variable.
    - `dataset.derived.search_variables("keyword")`: Search for derived variables based on keywords in their metadata.
- **Jupyter Notebook Integration:** When a Dataset is displayed in a Jupyter Notebook, a new "Derived variables" section is automatically added to its HTML representation. This section lists registered derived variables, their computability status (including missing dependencies), and expected output signature (dimensions, dtype, Dask status) without triggering computation. This feature can be configured (e.g., to show only computable variables).
- **Cycle Detection:** Automatically detects and prevents errors from circular dependencies in chained computations.
- **Extensibility:** Easily register your own custom derived variables tailored to your specific scientific domain or dataset.
- **Standard Variables:** Comes with a set of pre-defined common meteorological variables, including examples of chained computations (e.g., Relative Humidity from saturation mixing ratio, Equivalent Potential Temperature).
- **Configuration:** Plugin behavior, such as the notebook representation, can be tweaked via `xderived.config`.

## Installation

Currently, you can install `xderived` directly from the source code:

```bash
# Clone the repository (replace with your actual URL if you host it)
# git clone https://github.com/yourusername/xderived.git
# cd xderived

# Install using pip
pip install .
# Or for development mode:
# pip install -e .
```

(Once published to PyPI, installation will be as simple as `pip install xderived`.)

## Quick Start

Here's a brief example of how to use `xderived` with its new features:

```python
import xarray as xr
import numpy as np
import dask.array as da
import xderived # Registers the .derived accessor, standard variables, and HTML repr patch

# --- Create a sample dataset (can be Dask-backed) ---
ds = xr.Dataset(
    {
        "air_temperature": (("time", "lat"), da.from_array(np.array([[280., 285.], [290., 295.]]), chunks=(1,1)), {"units": "K"}),
        "air_pressure": (("time",), np.array([100000., 95000.]), {"units": "Pa"}),
        # specific_humidity is missing for some derived variables
    },
    coords={"time": np.arange(2), "lat": [30, 40]}
)

# --- Jupyter Notebook Display ---
# If you are in a Jupyter Notebook and display `ds`, you will see a new "Derived variables" section.
# This section will list variables like 'potential_temperature', 'relative_humidity_from_mixing_ratios',
# indicating their computability status (e.g., 'potential_temperature' might be computable, 
# while 'relative_humidity_from_mixing_ratios' might be 'unavailable' due to missing 'specific_humidity').
# The display is lazy and respects Dask arrays.

# Example: To see the HTML that would be generated (outside a notebook):
# html_output = ds._repr_html_()
# print(html_output) # This would show the full HTML including the derived variables section.

# --- Configure Notebook Display (Optional) ---
# from xderived import config
# config.config["repr_show_computable_only"] = True # Now notebook repr only shows computable ones
# display(ds) # Re-display in notebook to see the change
# config.config["repr_show_computable_only"] = False # Reset


# --- Discoverability Features (remain the same) ---
print("--- Accessor Representation (for console) ---")
print(ds.derived)

print("\n--- List Computable Variables ---")
print(ds.derived.list_computable())

print("\n--- Get Metadata for 'potential_temperature' ---")
pt_meta = ds.derived.get_metadata("potential_temperature")
if pt_meta:
    for k, v in pt_meta.items():
        print(f"  {k}: {v}")

# --- Computation (remains the same) ---
if "potential_temperature" in ds.derived.list_computable():
    pt = ds.derived.potential_temperature
    print("\n--- Computed Potential Temperature (Dask array) ---")
    print(pt)
    # To actually compute and get numpy values if it's a Dask array:
    # print(pt.compute())
else:
    print("\nPotential temperature is not computable from this dataset.")

```

(See the `examples/` directory for more detailed scripts, including `test_notebook_repr_example.py` which, while a script, simulates the HTML generation.)

## How it Works

1.  **`DerivedVariable` Class:** Stores the definition of a derived quantity, including its name, dependencies, calculation function, and rich metadata (description, units, formula string, standard name, long name, output dimension/dtype hints for display).
2.  **`DerivedVariableRegistry`:** A global registry holds all `DerivedVariable` definitions. Standard variables are registered upon import. Users can register their own.
3.  **`@xr.register_dataset_accessor("derived")`:** Adds the `.derived` accessor to all xarray Datasets.
4.  **`DerivedAccessor`:**
    *   Provides methods for discovery and computation.
    *   Its `__repr__` (for console) lists registered variables and their computability status.
    *   When a specific derived variable is accessed (e.g., `ds.derived.potential_temperature`), it resolves dependencies (recursively if needed, with cycle detection) and calls the calculation function. Results are cached.
    *   If input DataArrays are Dask-backed, the output derived DataArray will also be Dask-backed.
5.  **HTML Representation Patch (`html_repr.py`):**
    *   Upon import of `xderived`, the plugin attempts to "monkey-patch" xarray's internal HTML formatting function for Datasets.
    *   The patched function injects a new "Derived variables" section into the HTML output.
    *   This section lists registered derived variables, their computability status (based on `ds.derived.get_status()`), and an expected signature (dims, dtype, Dask status from `ds.derived.get_expected_signature()`).
    *   **Crucially, this display does not trigger any computation of the derived variables.** If a variable is Dask-backed, its Dask representation (shape, chunks) is shown if inferable from hints or direct Dask dependencies, otherwise a placeholder is used.
    *   The display respects the `xderived.config["repr_show_computable_only"]` setting.

## Configuration

The plugin offers some global configuration options via the `xderived.config.config` dictionary:

-   `config["repr_show_computable_only"]` (default: `False`): If set to `True`, the "Derived variables" section in the Jupyter Notebook HTML representation will only list variables that are currently computable from the dataset. Otherwise, it lists all registered variables with their status.

## Contributing

Contributions are welcome! If you'd like to contribute, please:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Add your changes, including tests if applicable.
4.  Ensure tests pass.
5.  Submit a pull request.

Possible areas for contribution:

*   Adding more standard derived variables.
*   Enhancing unit handling.
*   Improving documentation and examples.
*   More robust HTML injection for the notebook representation (less reliant on string matching xarray's internal HTML structure).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   The xarray team for creating and maintaining an indispensable tool for scientific data analysis.
*   The concepts of derived quantities in systems like VisAD and Unidata's IDV, which provided inspiration.

