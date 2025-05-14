# xderived/standard_variables.py

"""Standard derived variables for common meteorological calculations."""

import xarray as xr
import numpy as np
from .core import DerivedVariable, registry

# Constants
R_d = 287.058  # J/(kg·K), specific gas constant for dry air
C_p = 1005.0   # J/(kg·K), specific heat capacity of dry air at constant pressure
P0 = 100000.0  # Pa, reference pressure
L_v = 2.501e6  # J/kg, latent heat of vaporization for water
EPSILON = 0.622 # ratio of molar masses of water vapor to dry air

# --- Helper functions (if any, or define within lambdas/functions below) ---

# --- Variable Definitions ---

def calculate_potential_temperature(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Potential Temperature (theta)."""
    temp_k = ds["air_temperature"]
    pressure_pa = ds["air_pressure"]
    theta = temp_k * (P0 / pressure_pa)**(R_d / C_p)
    theta.attrs = {
        "units": "K",
        "long_name": "Air Potential Temperature",
        "standard_name": "air_potential_temperature"
    }
    return theta

def calculate_wind_speed(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Wind Speed from u and v components."""
    u = ds["eastward_wind"]
    v = ds["northward_wind"]
    speed = np.sqrt(u**2 + v**2)
    speed.attrs = {
        "units": getattr(u, "units", "m s-1"), # Preserve units if possible
        "long_name": "Wind Speed",
        "standard_name": "wind_speed"
    }
    return speed

def calculate_wind_from_direction(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Wind From Direction from u and v components."""
    u = ds["eastward_wind"]
    v = ds["northward_wind"]
    direction = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    direction.attrs = {
        "units": "degree",
        "long_name": "Wind From Direction",
        "standard_name": "wind_from_direction"
    }
    return direction

# --- Phase 2: Chainable and Discoverable Variables ---

def calculate_saturation_vapor_pressure_tetens(ds: xr.Dataset) -> xr.DataArray:
    """Calculate saturation vapor pressure using Tetens' formula."""
    temp_k = ds["air_temperature"]
    temp_c = temp_k - 273.15
    temp_c.attrs["units"] = "degC" # For clarity in formula application
    # Tetens' formula: es(T) = 0.61078 * exp((17.27 * T_c) / (T_c + 237.3)) (es in kPa)
    # Convert to Pa: es_Pa = 1000 * 0.61078 * exp((17.27 * T_c) / (T_c + 237.3))
    es_pa = 610.78 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    es_pa.attrs = {
        "units": "Pa",
        "long_name": "Saturation Vapor Pressure (Tetens)",
        "standard_name": "saturation_vapor_pressure"
    }
    return es_pa

def calculate_saturation_mixing_ratio(ds: xr.Dataset) -> xr.DataArray:
    """Calculate saturation mixing ratio."""
    pressure_pa = ds["air_pressure"]
    # Dependency on another derived variable
    es_pa = ds["saturation_vapor_pressure_tetens"] 
    # ws = epsilon * es / (p - es)
    # Ensure p - es is not zero or negative, though physically p should be > es
    denominator = xr.where(pressure_pa - es_pa <= 0, np.nan, pressure_pa - es_pa)
    ws = (EPSILON * es_pa) / denominator
    ws.attrs = {
        "units": "kg kg-1",
        "long_name": "Saturation Mixing Ratio",
        "standard_name": "saturation_mixing_ratio"
    }
    return ws

def calculate_mixing_ratio_from_specific_humidity(ds: xr.Dataset) -> xr.DataArray:
    """Calculate mixing ratio from specific humidity."""
    q = ds["specific_humidity"]
    # w = q / (1 - q)
    # Ensure 1 - q is not zero
    denominator = xr.where(1 - q <= 0, np.nan, 1 - q)
    w = q / denominator
    w.attrs = {
        "units": "kg kg-1",
        "long_name": "Mixing Ratio (from Specific Humidity)",
        "standard_name": "humidity_mixing_ratio"
    }
    return w

def calculate_relative_humidity_from_mixing_ratios(ds: xr.Dataset) -> xr.DataArray:
    """Calculate relative humidity from mixing ratio and saturation mixing ratio."""
    # Dependencies on other derived variables
    w = ds["mixing_ratio_from_specific_humidity"]
    ws = ds["saturation_mixing_ratio"]
    # RH = (w / ws) * 100
    # Ensure ws is not zero
    rh = xr.where(ws <= 0, np.nan, (w / ws) * 100)
    rh = rh.clip(0, 100) # RH should be between 0 and 100
    rh.attrs = {
        "units": "%",
        "long_name": "Relative Humidity (from Mixing Ratios)",
        "standard_name": "relative_humidity"
    }
    return rh

def calculate_equivalent_potential_temperature_approx(ds: xr.Dataset) -> xr.DataArray:
    """Calculate Equivalent Potential Temperature (approximate formula)."""
    # Dependencies on other derived variables and base variables
    theta = ds["potential_temperature"] # Derived
    w = ds["mixing_ratio_from_specific_humidity"] # Derived
    temp_k = ds["air_temperature"] # Base
    
    # θ_e ≈ θ * exp((L_v * w) / (c_p * T_k))
    # Using T_k as the surface temperature for approximation, common in some contexts.
    # More accurate calculations would use temperature at LCL, which is more complex.
    theta_e = theta * np.exp((L_v * w) / (C_p * temp_k))
    theta_e.attrs = {
        "units": "K",
        "long_name": "Equivalent Potential Temperature (Approximate)",
        "standard_name": "equivalent_potential_temperature"
    }
    return theta_e


# List of all standard variable definitions
# Phase 1 variables
PT_DEF = DerivedVariable(
    name="potential_temperature",
    dependencies=["air_temperature", "air_pressure"],
    func=calculate_potential_temperature,
    description="Air Potential Temperature calculated using temperature and pressure.",
    attrs={"units": "K", "long_name": "Air Potential Temperature", "standard_name": "air_potential_temperature"},
    formula_str="T * (P0 / P)^(R_d / C_p)"
)

WS_DEF = DerivedVariable(
    name="wind_speed",
    dependencies=["eastward_wind", "northward_wind"],
    func=calculate_wind_speed,
    description="Wind Speed calculated from u and v components.",
    attrs={"units": "m s-1", "long_name": "Wind Speed", "standard_name": "wind_speed"},
    formula_str="sqrt(u^2 + v^2)"
)

WD_DEF = DerivedVariable(
    name="wind_from_direction",
    dependencies=["eastward_wind", "northward_wind"],
    func=calculate_wind_from_direction,
    description="Wind From Direction calculated from u and v components.",
    attrs={"units": "degree", "long_name": "Wind From Direction", "standard_name": "wind_from_direction"},
    formula_str="(270 - atan2(v, u) * 180/pi) % 360"
)

# Phase 2 variables (demonstrating chaining)
SAT_VP_DEF = DerivedVariable(
    name="saturation_vapor_pressure_tetens",
    dependencies=["air_temperature"],
    func=calculate_saturation_vapor_pressure_tetens,
    description="Saturation vapor pressure using Tetens' formula.",
    attrs={"units": "Pa", "long_name": "Saturation Vapor Pressure (Tetens)", "standard_name": "saturation_vapor_pressure"},
    formula_str="610.78 * exp((17.27 * (T_k - 273.15)) / ((T_k - 273.15) + 237.3))"
)

MIX_RATIO_SPEC_HUM_DEF = DerivedVariable(
    name="mixing_ratio_from_specific_humidity",
    dependencies=["specific_humidity"],
    func=calculate_mixing_ratio_from_specific_humidity,
    description="Mixing ratio calculated from specific humidity.",
    attrs={"units": "kg kg-1", "long_name": "Mixing Ratio (from Specific Humidity)", "standard_name": "humidity_mixing_ratio"},
    formula_str="q / (1 - q)"
)

SAT_MIX_RATIO_DEF = DerivedVariable(
    name="saturation_mixing_ratio",
    dependencies=["air_pressure", "saturation_vapor_pressure_tetens"], # Depends on derived var
    func=calculate_saturation_mixing_ratio,
    description="Saturation mixing ratio.",
    attrs={"units": "kg kg-1", "long_name": "Saturation Mixing Ratio", "standard_name": "saturation_mixing_ratio"},
    formula_str="(0.622 * es_tetens) / (P - es_tetens)"
)

RH_MIX_RATIO_DEF = DerivedVariable(
    name="relative_humidity_from_mixing_ratios",
    dependencies=["mixing_ratio_from_specific_humidity", "saturation_mixing_ratio"], # Depends on two derived vars
    func=calculate_relative_humidity_from_mixing_ratios,
    description="Relative humidity from mixing ratio and saturation mixing ratio.",
    attrs={"units": "%", "long_name": "Relative Humidity (from Mixing Ratios)", "standard_name": "relative_humidity"},
    formula_str="(w / ws) * 100"
)

THETA_E_APPROX_DEF = DerivedVariable(
    name="equivalent_potential_temperature_approx",
    dependencies=["potential_temperature", "mixing_ratio_from_specific_humidity", "air_temperature"], # Depends on two derived and one base
    func=calculate_equivalent_potential_temperature_approx,
    description="Equivalent Potential Temperature (approximate).",
    attrs={"units": "K", "long_name": "Equivalent Potential Temperature (Approximate)", "standard_name": "equivalent_potential_temperature"},
    formula_str="theta * exp((Lv * w) / (Cp * T_k))"
)


# Function to register all standard variables
def register_standard_variables():
    """Registers all predefined standard derived variables."""
    # Clear existing ones first to allow re-registration during development/testing
    # In a production setting, you might not want to clear if extending dynamically.
    # For this project structure, clearing and re-adding is fine for initialization.
    # registry.clear() # Decided against clearing here, __init__ can handle if needed for a full reset.
    
    standard_vars = [
        PT_DEF,
        WS_DEF,
        WD_DEF,
        # Phase 2 variables
        SAT_VP_DEF,
        MIX_RATIO_SPEC_HUM_DEF,
        SAT_MIX_RATIO_DEF,
        RH_MIX_RATIO_DEF,
        THETA_E_APPROX_DEF
    ]
    for var_def in standard_vars:
        try:
            registry.register(var_def)
        except registry.RegistrationError: # Use fully qualified name for the exception type
            # Variable might already be registered (e.g., if this function is called multiple times without clearing)
            # For now, we can choose to ignore this or print a warning.
            # print(f"Warning: Variable {var_def.name} already registered. Skipping.")
            pass # Silently skip if already registered

# Call registration when this module is imported (optional, depends on desired behavior)
# register_standard_variables() 
# It's better to call this explicitly from __init__.py to control registration timing.

