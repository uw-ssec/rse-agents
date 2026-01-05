# Real-World Examples

## Example 1: Climate Data Analysis

**Load and analyze global temperature data:**
```python
import xarray as xr
import matplotlib.pyplot as plt

# Load climate data
ds = xr.open_dataset("global_temperature.nc", chunks={"time": 365})

# Calculate annual mean temperature
annual_mean = ds["temperature"].resample(time="1Y").mean()

# Calculate global mean (area-weighted)
weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"
global_mean = (ds["temperature"] * weights).sum(dim=["lat", "lon"]) / weights.sum()

# Plot time series
global_mean.plot()
plt.title("Global Mean Temperature")
plt.ylabel("Temperature (°C)")
plt.show()

# Calculate temperature anomaly
climatology = ds["temperature"].groupby("time.month").mean(dim="time")
anomaly = ds["temperature"].groupby("time.month") - climatology

# Save results
anomaly.to_netcdf("temperature_anomaly.nc")
```

## Example 2: Satellite Data Processing

**Process multi-temporal satellite imagery:**
```python
# Load satellite data
ds = xr.open_mfdataset("satellite_*.nc", combine="by_coords")

# Calculate NDVI (Normalized Difference Vegetation Index)
ndvi = (ds["nir"] - ds["red"]) / (ds["nir"] + ds["red"])
ds["ndvi"] = ndvi

# Calculate temporal statistics
ndvi_mean = ds["ndvi"].mean(dim="time")
ndvi_std = ds["ndvi"].std(dim="time")
ndvi_trend = ds["ndvi"].polyfit(dim="time", deg=1)

# Identify areas with significant vegetation change
change_mask = np.abs(ndvi_trend.polyfit_coefficients.sel(degree=1)) > 0.01

# Export results
result = xr.Dataset({
    "ndvi_mean": ndvi_mean,
    "ndvi_std": ndvi_std,
    "change_mask": change_mask
})
result.to_netcdf("ndvi_analysis.nc")
```

## Example 3: Oceanographic Data Analysis

**Analyze ocean temperature and salinity profiles:**
```python
# Load ocean data with depth dimension
ds = xr.open_dataset("ocean_profiles.nc")

# Calculate mixed layer depth (where temp drops by 0.5°C from surface)
surface_temp = ds["temperature"].isel(depth=0)
temp_diff = surface_temp - ds["temperature"]
mld = ds["depth"].where(temp_diff > 0.5).min(dim="depth")

# Calculate heat content
heat_capacity = 4186  # J/(kg·K)
density = 1025  # kg/m³
heat_content = (ds["temperature"] * heat_capacity * density).integrate("depth")

# Seasonal analysis
seasonal_temp = ds["temperature"].groupby("time.season").mean()

# Plot vertical profile
ds["temperature"].sel(lat=0, lon=180, method="nearest").plot(y="depth")
plt.gca().invert_yaxis()  # Depth increases downward
plt.title("Temperature Profile at Equator")
```

## Example 4: Multi-Model Ensemble Analysis

**Compare and analyze multiple climate model outputs:**
```python
import xarray as xr
import glob

# Load multiple model outputs
model_files = glob.glob("models/model_*.nc")
models = [xr.open_dataset(f, chunks={"time": 365}) for f in model_files]

# Add model dimension
for i, ds in enumerate(models):
    ds["model"] = i

# Concatenate into single dataset
ensemble = xr.concat(models, dim="model")

# Calculate ensemble mean and spread
ensemble_mean = ensemble.mean(dim="model")
ensemble_std = ensemble.std(dim="model")

# Calculate model agreement (fraction of models agreeing on sign of change)
future_change = ensemble.sel(time=slice("2080", "2100")).mean(dim="time")
historical = ensemble.sel(time=slice("1980", "2000")).mean(dim="time")
change = future_change - historical

# Agreement: fraction of models with same sign as ensemble mean
agreement = (np.sign(change) == np.sign(change.mean(dim="model"))).sum(dim="model") / len(models)

# Identify robust changes (high agreement and large magnitude)
robust_change = change.mean(dim="model").where(
    (agreement > 0.8) & (np.abs(change.mean(dim="model")) > ensemble_std.mean(dim="model"))
)

# Save results
result = xr.Dataset({
    "ensemble_mean": ensemble_mean,
    "ensemble_std": ensemble_std,
    "agreement": agreement,
    "robust_change": robust_change
})
result.to_netcdf("ensemble_analysis.nc")
```

## Example 5: Time Series Decomposition

**Decompose time series into trend, seasonal, and residual components:**
```python
import xarray as xr
from scipy import signal

# Load time series data
ds = xr.open_dataset("timeseries.nc")

# Calculate long-term trend (annual rolling mean)
trend = ds["temperature"].rolling(time=365, center=True).mean()

# Remove trend to get anomaly
anomaly = ds["temperature"] - trend

# Calculate seasonal cycle (monthly climatology)
seasonal = anomaly.groupby("time.month").mean(dim="time")

# Remove seasonal cycle to get residual
residual = anomaly.groupby("time.month") - seasonal

# Combine into single dataset
decomposition = xr.Dataset({
    "original": ds["temperature"],
    "trend": trend,
    "seasonal": seasonal.rename({"month": "time"}),
    "residual": residual
})

# Calculate variance explained by each component
total_var = ds["temperature"].var()
trend_var = trend.var()
seasonal_var = seasonal.var()
residual_var = residual.var()

print(f"Trend explains {100*trend_var/total_var:.1f}% of variance")
print(f"Seasonal explains {100*seasonal_var/total_var:.1f}% of variance")
print(f"Residual explains {100*residual_var/total_var:.1f}% of variance")

# Plot decomposition
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition["original"].plot(ax=axes[0])
axes[0].set_title("Original")
decomposition["trend"].plot(ax=axes[1])
axes[1].set_title("Trend")
decomposition["seasonal"].plot(ax=axes[2])
axes[2].set_title("Seasonal")
decomposition["residual"].plot(ax=axes[3])
axes[3].set_title("Residual")
plt.tight_layout()
```

