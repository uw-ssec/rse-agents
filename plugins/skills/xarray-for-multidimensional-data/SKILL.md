---
name: xarray-for-multidimensional-data
description: Labeled multidimensional arrays for scientific data analysis with Xarray
---

# Xarray for Multidimensional Data

Master **Xarray**, the powerful library for working with labeled multidimensional arrays in scientific Python. Learn how to efficiently handle complex datasets with multiple dimensions, coordinates, and metadata - from climate data and satellite imagery to experimental measurements and simulations.

**Official Documentation**: https://docs.xarray.dev/

**GitHub**: https://github.com/pydata/xarray

## Quick Reference Card

### Installation & Setup
```bash
# Using pixi (recommended for scientific projects)
pixi add xarray netcdf4 dask

# Using pip
pip install xarray[complete]

# Optional dependencies for specific formats
pixi add zarr h5netcdf scipy bottleneck
```

### Essential Xarray Concepts
```python
import xarray as xr
import numpy as np

# DataArray: Single labeled array
temperature = xr.DataArray(
    data=np.random.randn(3, 4),
    dims=["time", "location"],
    coords={
        "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "location": ["A", "B", "C", "D"]
    },
    name="temperature"
)

# Dataset: Collection of DataArrays
ds = xr.Dataset({
    "temperature": temperature,
    "pressure": (["time", "location"], np.random.randn(3, 4))
})
```

### Essential Operations
```python
# Selection by label
ds.sel(time="2024-01-01")
ds.sel(location="A")

# Selection by index
ds.isel(time=0)

# Slicing
ds.sel(time=slice("2024-01-01", "2024-01-02"))

# Aggregation
ds.mean(dim="time")
ds.sum(dim="location")

# Computation
ds["temperature"] + 273.15  # Celsius to Kelvin
ds.groupby("time.month").mean()

# I/O operations
ds.to_netcdf("data.nc")
ds = xr.open_dataset("data.nc")
```

### Quick Decision Tree

```
Working with multidimensional scientific data?
├─ YES → Use Xarray for labeled dimensions
└─ NO → NumPy/Pandas sufficient

Need to track coordinates and metadata?
├─ YES → Xarray keeps everything aligned
└─ NO → Plain NumPy arrays work

Data too large for memory?
├─ YES → Use Xarray with Dask backend
└─ NO → Standard Xarray is fine

Need to save/load scientific data formats?
├─ NetCDF/HDF5 → Xarray native support
├─ Zarr → Use Xarray with zarr backend
└─ CSV/Excel → Pandas then convert to Xarray

Working with time series data?
├─ Multi-dimensional → Xarray
└─ Tabular → Pandas

Need to align data from different sources?
├─ YES → Xarray handles alignment automatically
└─ NO → Manual alignment with NumPy
```

## When to Use This Skill

Use Xarray when working with:

- **Climate and weather data** with dimensions like time, latitude, longitude, and altitude
- **Satellite and remote sensing imagery** with spatial and temporal dimensions
- **Oceanographic data** with depth, time, and spatial coordinates
- **Experimental measurements** with multiple parameters and conditions
- **Simulation outputs** with complex dimensional structures
- **Time series data** that varies across multiple spatial locations
- **Any data** where keeping track of dimensions and coordinates is critical
- **Large datasets** that benefit from lazy loading and Dask integration

## Core Concepts

### 1. DataArray: Labeled Multidimensional Arrays

A **DataArray** is Xarray's fundamental data structure - think of it as a NumPy array with labels and metadata.

**Anatomy of a DataArray:**
```python
import xarray as xr
import numpy as np

# Create a DataArray
temperature = xr.DataArray(
    data=np.array([[15.2, 16.1, 14.8],
                   [16.5, 17.2, 15.9],
                   [17.1, 18.0, 16.5]]),
    dims=["time", "location"],
    coords={
        "time": pd.date_range("2024-01-01", periods=3),
        "location": ["Station_A", "Station_B", "Station_C"],
        "lat": ("location", [40.7, 34.0, 41.8]),
        "lon": ("location", [-74.0, -118.2, -87.6])
    },
    attrs={
        "units": "Celsius",
        "description": "Daily average temperature"
    }
)
```

**Key components:**
- **data**: The actual NumPy array
- **dims**: Dimension names (like column names in Pandas)
- **coords**: Coordinate labels for each dimension
- **attrs**: Metadata dictionary

### 2. Dataset: Collection of DataArrays

A **Dataset** is like a dict of DataArrays that share dimensions - similar to a Pandas DataFrame but for N-dimensional data.

**Example:**
```python
# Create a Dataset
ds = xr.Dataset({
    "temperature": (["time", "location"], np.random.randn(3, 4)),
    "humidity": (["time", "location"], np.random.rand(3, 4) * 100),
    "pressure": (["time", "location"], 1013 + np.random.randn(3, 4) * 10)
},
coords={
    "time": pd.date_range("2024-01-01", periods=3),
    "location": ["A", "B", "C", "D"]
})
```

### 3. Coordinates: Dimension Labels

Coordinates provide meaningful labels for array dimensions and enable label-based indexing.

**Types of coordinates:**

**Dimension coordinates** (1D, same name as dimension):
```python
time_coord = pd.date_range("2024-01-01", periods=365)
```

**Non-dimension coordinates** (auxiliary information):
```python
# Latitude/longitude for each station
coords = {
    "time": time_coord,
    "station": ["A", "B", "C"],
    "lat": ("station", [40.7, 34.0, 41.8]),
    "lon": ("station", [-74.0, -118.2, -87.6])
}
```

### 4. Indexing and Selection

Xarray provides powerful label-based and position-based indexing.

**Label-based selection (.sel):**
```python
# Select by coordinate value
ds.sel(time="2024-01-15")
ds.sel(location="Station_A")

# Nearest neighbor selection
ds.sel(time="2024-01-15", method="nearest")

# Range selection
ds.sel(time=slice("2024-01-01", "2024-01-31"))
```

**Position-based selection (.isel):**
```python
# Select by integer position
ds.isel(time=0)
ds.isel(location=[0, 2])
```

**Boolean indexing (.where):**
```python
# Keep only values meeting condition
ds.where(ds["temperature"] > 15, drop=True)
```

## Decision Trees

### Choosing Between Xarray, Pandas, and NumPy

```
What's your data structure?
├─ Tabular (2D with rows/columns) → Pandas
├─ Simple arrays without labels → NumPy
└─ Multidimensional with coordinates → Xarray

Need to track dimension names?
├─ YES → Xarray or Pandas
└─ NO → NumPy sufficient

More than 2 dimensions?
├─ YES → Xarray (Pandas limited to 2D)
└─ NO → Pandas or Xarray

Need automatic alignment?
├─ YES → Xarray or Pandas
└─ NO → NumPy with manual alignment

Working with NetCDF/HDF5?
├─ YES → Xarray (native support)
└─ NO → Any tool works
```

### File Format Selection

```
What's your use case?
├─ Climate/weather data → NetCDF4
├─ Cloud-optimized access → Zarr
├─ General scientific data → NetCDF4 or HDF5
├─ Interoperability with other tools → NetCDF4
├─ Maximum compression → Zarr with compression
└─ Streaming/incremental writes → Zarr

Need cloud storage support?
├─ YES → Zarr (cloud-native)
└─ NO → NetCDF4 works fine

Need parallel writes?
├─ YES → Zarr
└─ NO → NetCDF4 or Zarr

File size concerns?
├─ CRITICAL → Zarr with compression
└─ NOT CRITICAL → NetCDF4 default
```

## Patterns and Examples

### Pattern 1: Creating DataArrays and Datasets

**From NumPy arrays:**
```python
import xarray as xr
import numpy as np
import pandas as pd

# Create sample data
data = np.random.randn(365, 10, 20)  # time, lat, lon

# Create DataArray with coordinates
temperature = xr.DataArray(
    data=data,
    dims=["time", "lat", "lon"],
    coords={
        "time": pd.date_range("2024-01-01", periods=365),
        "lat": np.linspace(-90, 90, 10),
        "lon": np.linspace(-180, 180, 20)
    },
    attrs={
        "units": "Celsius",
        "long_name": "Air Temperature"
    }
)

# Create Dataset with multiple variables
ds = xr.Dataset({
    "temperature": temperature,
    "precipitation": (["time", "lat", "lon"], 
                     np.random.rand(365, 10, 20) * 100)
})
```

**From Pandas DataFrame:**
```python
import pandas as pd

# Tabular data
df = pd.DataFrame({
    "time": pd.date_range("2024-01-01", periods=100),
    "station": ["A"] * 50 + ["B"] * 50,
    "temperature": np.random.randn(100),
    "humidity": np.random.rand(100) * 100
})

# Convert to Xarray
ds = df.set_index(["time", "station"]).to_xarray()
```

### Pattern 2: Reading and Writing Data

**NetCDF files:**
```python
# Write to NetCDF
ds.to_netcdf("climate_data.nc")

# Read from NetCDF
ds = xr.open_dataset("climate_data.nc")

# Read multiple files as single dataset
ds = xr.open_mfdataset("data_*.nc", combine="by_coords")

# Lazy loading (doesn't load data into memory)
ds = xr.open_dataset("large_file.nc", chunks={"time": 100})
```

**Zarr format (cloud-optimized):**
```python
# Write to Zarr
ds.to_zarr("climate_data.zarr")

# Read from Zarr
ds = xr.open_zarr("climate_data.zarr")

# Write to cloud storage (S3, GCS)
ds.to_zarr("s3://bucket/climate_data.zarr")
```

**Other formats:**
```python
# From CSV (via Pandas)
df = pd.read_csv("data.csv")
ds = df.to_xarray()

# To CSV (flatten first)
ds.to_dataframe().to_csv("output.csv")
```

### Pattern 3: Selection and Indexing

**Label-based selection:**
```python
# Select single time point
ds.sel(time="2024-01-15")

# Select multiple coordinates
ds.sel(time="2024-01-15", lat=40.7, lon=-74.0)

# Nearest neighbor (useful for inexact matches)
ds.sel(lat=40.5, lon=-74.2, method="nearest")

# Range selection
ds.sel(time=slice("2024-01-01", "2024-01-31"))
ds.sel(lat=slice(30, 50))

# Select multiple discrete values
ds.sel(time=["2024-01-01", "2024-01-15", "2024-01-31"])
```

**Position-based selection:**
```python
# Select by integer index
ds.isel(time=0)
ds.isel(lat=slice(0, 5), lon=slice(0, 10))

# Select every nth element
ds.isel(time=slice(None, None, 7))  # Every 7th time point
```

**Conditional selection:**
```python
# Keep only values meeting condition
warm_days = ds.where(ds["temperature"] > 20, drop=True)

# Replace values not meeting condition
ds_filled = ds.where(ds["temperature"] > 0, 0)

# Boolean mask
mask = (ds["temperature"] > 15) & (ds["temperature"] < 25)
comfortable_temps = ds.where(mask)
```

### Pattern 4: Computation and Aggregation

**Basic operations:**
```python
# Arithmetic operations
ds["temp_kelvin"] = ds["temperature"] + 273.15
ds["temp_fahrenheit"] = ds["temperature"] * 9/5 + 32

# Statistical operations
mean_temp = ds["temperature"].mean()
std_temp = ds["temperature"].std()
max_temp = ds["temperature"].max()

# Aggregation along dimensions
daily_mean = ds.mean(dim="time")
spatial_mean = ds.mean(dim=["lat", "lon"])
```

**GroupBy operations:**
```python
# Group by time components
monthly_mean = ds.groupby("time.month").mean()
seasonal_mean = ds.groupby("time.season").mean()
hourly_mean = ds.groupby("time.hour").mean()

# Custom grouping
ds["region"] = (["lat", "lon"], region_mask)
regional_mean = ds.groupby("region").mean()
```

**Rolling window operations:**
```python
# 7-day rolling mean
rolling_mean = ds.rolling(time=7, center=True).mean()

# 30-day rolling sum
rolling_sum = ds.rolling(time=30).sum()
```

**Resampling (time series):**
```python
# Resample to monthly
monthly = ds.resample(time="1M").mean()

# Resample to weekly
weekly = ds.resample(time="1W").sum()

# Upsample and interpolate
daily = ds.resample(time="1D").interpolate("linear")
```

### Pattern 5: Combining Datasets

**Concatenation:**
```python
# Concatenate along existing dimension
combined = xr.concat([ds1, ds2, ds3], dim="time")

# Concatenate along new dimension
ensemble = xr.concat([run1, run2, run3], 
                     dim=pd.Index([1, 2, 3], name="run"))
```

**Merging:**
```python
# Merge datasets with different variables
merged = xr.merge([temp_ds, precip_ds, pressure_ds])

# Merge with alignment
merged = xr.merge([ds1, ds2], join="inner")  # or "outer", "left", "right"
```

**Alignment:**
```python
# Automatic alignment in operations
result = ds1 + ds2  # Automatically aligns coordinates

# Manual alignment
ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="inner")
```

### Pattern 6: Dask Integration for Large Data

**Chunked operations:**
```python
import dask

# Open with chunks (lazy loading)
ds = xr.open_dataset("large_file.nc", chunks={"time": 100, "lat": 50})

# Operations remain lazy
result = ds["temperature"].mean(dim="time")

# Trigger computation
computed_result = result.compute()

# Parallel computation
with dask.config.set(scheduler="threads", num_workers=4):
    result = ds.mean(dim="time").compute()
```

**Chunking strategies:**
```python
# Chunk by time (good for time series operations)
ds = ds.chunk({"time": 365})

# Chunk by space (good for spatial operations)
ds = ds.chunk({"lat": 50, "lon": 50})

# Auto-chunking
ds = ds.chunk("auto")

# Rechunk for different operations
ds_rechunked = ds.chunk({"time": -1, "lat": 10, "lon": 10})
```

### Pattern 7: Interpolation and Regridding

**Interpolation:**
```python
# Interpolate to new coordinates
new_lats = np.linspace(-90, 90, 180)
new_lons = np.linspace(-180, 180, 360)

ds_interp = ds.interp(lat=new_lats, lon=new_lons, method="linear")

# Interpolate missing values
ds_filled = ds.interpolate_na(dim="time", method="linear")
```

**Reindexing:**
```python
# Reindex to new coordinates
new_time = pd.date_range("2024-01-01", "2024-12-31", freq="1D")
ds_reindexed = ds.reindex(time=new_time, method="nearest")

# Fill missing values during reindex
ds_reindexed = ds.reindex(time=new_time, fill_value=0)
```

### Pattern 8: Working with Coordinates

**Adding coordinates:**
```python
# Add non-dimension coordinate
ds = ds.assign_coords(station_name=(["location"], ["NYC", "LA", "CHI"]))

# Add computed coordinate
ds = ds.assign_coords(day_of_year=ds["time"].dt.dayofyear)
```

**Swapping dimensions:**
```python
# Use non-dimension coordinate as dimension
ds_swapped = ds.swap_dims({"location": "station_name"})
```

**Multi-index coordinates:**
```python
# Create multi-index
ds = ds.set_index(location=["lat", "lon"])

# Reset index
ds = ds.reset_index("location")
```

### Pattern 9: Plotting and Visualization

**Basic plotting:**
```python
import matplotlib.pyplot as plt

# Line plot (1D)
ds["temperature"].sel(location="Station_A").plot()

# Heatmap (2D)
ds["temperature"].sel(time="2024-01-15").plot()

# Faceted plots
ds["temperature"].plot(col="time", col_wrap=4)
```

**Advanced plotting:**
```python
# Contour plot
ds["temperature"].plot.contourf(levels=20)

# Time series with error bands
mean = ds["temperature"].mean(dim="location")
std = ds["temperature"].std(dim="location")

fig, ax = plt.subplots()
mean.plot(ax=ax, label="Mean")
ax.fill_between(mean.time, mean - std, mean + std, alpha=0.3)
```

### Pattern 10: Custom Functions with apply_ufunc

**Apply NumPy functions:**
```python
# Apply custom function element-wise
def custom_transform(x):
    return np.log(x + 1)

result = xr.apply_ufunc(
    custom_transform,
    ds["temperature"],
    dask="parallelized",
    output_dtypes=[float]
)
```

**Vectorized operations:**
```python
from scipy import stats

def detrend(data, axis):
    return stats.detrend(data, axis=axis)

# Apply along specific dimension
detrended = xr.apply_ufunc(
    detrend,
    ds["temperature"],
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    kwargs={"axis": -1},
    dask="parallelized",
    output_dtypes=[float]
)
```

## Best Practices Checklist

### Data Organization
- Use meaningful dimension and coordinate names
- Include units and descriptions in attrs
- Use standard dimension names (time, lat, lon, etc.) when applicable
- Keep coordinates sorted for better performance
- Use appropriate data types (float32 vs float64)

### Performance
- Chunk large datasets appropriately for your operations
- Use lazy loading with open_dataset(chunks=...)
- Avoid loading entire dataset into memory unnecessarily
- Use vectorized operations instead of loops
- Consider using float32 instead of float64 for large datasets

### File I/O
- Use NetCDF4 for general scientific data
- Use Zarr for cloud storage and parallel writes
- Include metadata (attrs) when saving
- Use compression for large datasets
- Document coordinate reference systems for geospatial data

### Code Quality
- Use .sel() for label-based indexing (more readable)
- Chain operations for clarity
- Use meaningful variable names
- Add type hints for function parameters
- Document expected dimensions in docstrings

### Computation
- Use built-in methods (.mean(), .sum()) over manual loops
- Leverage groupby for categorical aggregations
- Use .compute() explicitly with Dask
- Monitor memory usage with large datasets
- Use .persist() to cache intermediate results

## Common Issues and Solutions

### Issue 1: Memory Errors with Large Datasets

**Problem:** `MemoryError` when loading large NetCDF files.

**Solution:** Use chunking and lazy loading:
```python
# Don't do this
ds = xr.open_dataset("large_file.nc")  # Loads everything

# Do this instead
ds = xr.open_dataset("large_file.nc", chunks={"time": 100})
result = ds.mean(dim="time").compute()  # Lazy evaluation
```

### Issue 2: Misaligned Coordinates

**Problem:** Operations fail due to coordinate mismatch.

**Solution:** Use alignment or reindexing:
```python
# Automatic alignment
result = ds1 + ds2  # Xarray aligns automatically

# Manual alignment with specific join
ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="inner")

# Reindex to match
ds2_reindexed = ds2.reindex_like(ds1, method="nearest")
```

### Issue 3: Slow Operations on Chunked Data

**Problem:** Operations are slower with Dask than expected.

**Solution:** Optimize chunking strategy:
```python
# Bad: chunks too small
ds = ds.chunk({"time": 1})  # Too much overhead

# Good: reasonable chunk size
ds = ds.chunk({"time": 100})  # Better balance

# Rechunk for specific operation
ds_rechunked = ds.chunk({"time": -1, "lat": 50})  # All time, chunked space
```

### Issue 4: Coordinate Precision Issues

**Problem:** `.sel()` doesn't find exact coordinate values.

**Solution:** Use nearest neighbor or tolerance:
```python
# Fails if exact match not found
ds.sel(lat=40.7128)  # Might fail

# Use nearest neighbor
ds.sel(lat=40.7128, method="nearest")

# Use tolerance
ds.sel(lat=40.7128, method="nearest", tolerance=0.01)
```

### Issue 5: Dimension Order Confusion

**Problem:** Operations produce unexpected results due to dimension order.

**Solution:** Explicitly specify dimensions:
```python
# Ambiguous
result = ds.mean()  # Means over all dimensions

# Clear
result = ds.mean(dim=["lat", "lon"])  # Spatial mean
result = ds.mean(dim="time")  # Temporal mean
```

### Issue 6: Broadcasting Errors

**Problem:** Operations fail with dimension mismatch errors.

**Solution:** Use broadcasting or alignment:
```python
# Error: dimensions don't match
weights = xr.DataArray([1, 2, 3], dims="location")
result = ds * weights  # Fails if ds has different dims

# Solution: broadcast explicitly
weights_broadcast = weights.broadcast_like(ds)
result = ds * weights_broadcast

# Or use align
ds_aligned, weights_aligned = xr.align(ds, weights, join="outer")
```

### Issue 7: Encoding Issues When Saving

**Problem:** Data types or attributes cause errors when saving to NetCDF.

**Solution:** Set encoding explicitly:
```python
# Specify encoding for each variable
encoding = {
    "temperature": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": -999.0
    }
}

ds.to_netcdf("data.nc", encoding=encoding)
```

### Issue 8: Time Coordinate Parsing Issues

**Problem:** Time coordinates not recognized or parsed incorrectly.

**Solution:** Use pandas datetime and set calendar:
```python
# Ensure proper datetime format
import pandas as pd

time = pd.date_range("2024-01-01", periods=365, freq="D")
ds = ds.assign_coords(time=time)

# For non-standard calendars (climate models)
import cftime
time_noleap = xr.cftime_range("2024-01-01", periods=365, calendar="noleap")
```

## Performance Optimization

### Memory Management

**Strategies for working with large datasets:**

```python
# 1. Use appropriate data types
ds["temperature"] = ds["temperature"].astype("float32")  # Half the memory

# 2. Drop unnecessary variables early
ds = ds[["temperature", "pressure"]]  # Keep only what you need

# 3. Select spatial/temporal subset first
ds_subset = ds.sel(lat=slice(30, 50), lon=slice(-120, -80))

# 4. Use chunks that fit in memory
ds = ds.chunk({"time": 100, "lat": 50, "lon": 50})

# 5. Persist intermediate results
ds_processed = ds.mean(dim="time").persist()  # Cache in memory
```

### Computation Optimization

**Optimize operations for performance:**

```python
# Use vectorized operations
# Bad: loop over coordinates
results = []
for lat in ds.lat:
    results.append(ds.sel(lat=lat).mean())

# Good: vectorized operation
result = ds.mean(dim="lon")

# Use built-in methods over apply_ufunc when possible
# Good: built-in
rolling_mean = ds.rolling(time=7).mean()

# Avoid: custom function
# def rolling_mean_custom(x):
#     return np.convolve(x, np.ones(7)/7, mode='same')

# Combine operations to reduce passes over data
# Bad: multiple passes
result = ds.mean(dim="time").std(dim="lat").max(dim="lon")

# Better: combine where possible
result = ds.mean(dim="time").reduce(np.std, dim="lat").max(dim="lon")
```

### Parallel Computing with Dask

**Optimize Dask operations:**

```python
import dask

# Configure Dask scheduler
with dask.config.set(scheduler="threads", num_workers=4):
    result = ds.mean(dim="time").compute()

# Use persist for reused intermediate results
ds_annual = ds.resample(time="1Y").mean()
ds_annual = ds_annual.persist()  # Keep in memory

# Multiple operations on same data
result1 = ds_annual.mean()
result2 = ds_annual.std()

# Batch computations
results = dask.compute(
    ds.mean(dim="time"),
    ds.std(dim="time"),
    ds.max(dim="time")
)

# Monitor progress
from dask.diagnostics import ProgressBar

with ProgressBar():
    result = ds.mean(dim="time").compute()
```

## Integration with Other Tools

### Pandas Integration

**Convert between Xarray and Pandas:**
```python
# Xarray to Pandas
df = ds.to_dataframe()

# Pandas to Xarray
ds = df.to_xarray()

# Multi-index DataFrame to Xarray
ds = df.set_index(["time", "location"]).to_xarray()

# Preserve metadata during conversion
ds_with_attrs = df.to_xarray()
ds_with_attrs.attrs = {"description": "Converted from DataFrame"}
```

### Dask Integration

**Parallel computation:**
```python
import dask

# Open with Dask
ds = xr.open_dataset("file.nc", chunks={"time": 100})

# Parallel operations
result = ds.mean(dim="time").compute()

# Configure Dask
with dask.config.set(scheduler="threads", num_workers=4):
    result = ds.mean().compute()
```

### Matplotlib/Cartopy Integration

**Geospatial plotting:**
```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

ds["temperature"].sel(time="2024-01-15").plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs={"label": "Temperature (°C)"}
)
ax.coastlines()
```

### SciPy Integration

**Apply SciPy functions:**
```python
from scipy import signal
from scipy import stats

# Apply filter along time dimension
filtered = xr.apply_ufunc(
    signal.detrend,
    ds["temperature"],
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    dask="parallelized"
)

# Statistical tests
def correlation_test(x, y):
    """Calculate correlation and p-value."""
    corr, pval = stats.pearsonr(x, y)
    return corr, pval

# Apply along specific dimensions
corr = xr.apply_ufunc(
    lambda x, y: stats.pearsonr(x, y)[0],
    ds["temperature"],
    ds["pressure"],
    input_core_dims=[["time"], ["time"]],
    vectorize=True,
    dask="parallelized"
)
```

### NumPy Integration

**Seamless NumPy operations:**
```python
# Xarray operations return Xarray objects
result = np.sqrt(ds["temperature"])  # Returns DataArray
result = np.exp(ds["temperature"])   # Returns DataArray

# Access underlying NumPy array
numpy_array = ds["temperature"].values

# Create DataArray from NumPy result
numpy_result = np.fft.fft(ds["temperature"].values, axis=0)
fft_result = xr.DataArray(
    numpy_result,
    dims=ds["temperature"].dims,
    coords=ds["temperature"].coords
)
```

### Scikit-learn Integration

**Machine learning with Xarray:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reshape for sklearn (samples, features)
X = ds["temperature"].stack(sample=("time", "lat", "lon"))
X_array = X.values.T  # Transpose to (samples, features)

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_array)

# Convert back to Xarray
pca_result = xr.DataArray(
    X_pca.T,
    dims=["component", "sample"],
    coords={"component": [1, 2, 3], "sample": X.sample}
)

# Unstack back to original dimensions
pca_unstacked = pca_result.unstack("sample")
```

## Real-World Examples

### Example 1: Climate Data Analysis

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

### Example 2: Satellite Data Processing

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

### Example 3: Oceanographic Data Analysis

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

### Example 4: Multi-Model Ensemble Analysis

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

### Example 5: Time Series Decomposition

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

## Resources and References

### Official Documentation
- **Xarray Documentation**: https://docs.xarray.dev/
- **Xarray Tutorial**: https://tutorial.xarray.dev/
- **API Reference**: https://docs.xarray.dev/en/stable/api.html

### File Formats
- **NetCDF**: https://www.unidata.ucar.edu/software/netcdf/
- **Zarr**: https://zarr.readthedocs.io/
- **HDF5**: https://www.hdfgroup.org/solutions/hdf5/

### Related Libraries
- **Dask**: https://docs.dask.org/ (parallel computing)
- **Pandas**: https://pandas.pydata.org/ (tabular data)
- **NumPy**: https://numpy.org/ (array operations)

### Domain-Specific Resources
- **Climate Data Operators (CDO)**: https://code.mpimet.mpg.de/projects/cdo
- **Pangeo**: https://pangeo.io/ (big data geoscience)
- **Xarray-spatial**: https://xarray-spatial.org/ (spatial analytics)

### Tutorials and Examples
- **Xarray Examples Gallery**: https://docs.xarray.dev/en/stable/gallery.html
- **Pangeo Gallery**: https://gallery.pangeo.io/
- **Earth and Environmental Data Science**: https://earth-env-data-science.github.io/

## Quick Start Template

**Basic Xarray workflow:**
```python
import xarray as xr
import numpy as np
import pandas as pd

# Create sample dataset
ds = xr.Dataset(
    {
        "temperature": (["time", "lat", "lon"], 
                       np.random.randn(365, 10, 20)),
        "precipitation": (["time", "lat", "lon"], 
                         np.random.rand(365, 10, 20) * 100)
    },
    coords={
        "time": pd.date_range("2024-01-01", periods=365),
        "lat": np.linspace(-90, 90, 10),
        "lon": np.linspace(-180, 180, 20)
    }
)

# Basic operations
monthly_mean = ds.resample(time="1M").mean()
spatial_mean = ds.mean(dim=["lat", "lon"])
warm_days = ds.where(ds["temperature"] > 20)

# Save and load
ds.to_netcdf("data.nc")
ds_loaded = xr.open_dataset("data.nc")

# Plot
ds["temperature"].sel(time="2024-01-15").plot()
```

## Summary

Xarray is the go-to library for working with labeled multidimensional arrays in scientific Python. It combines the power of NumPy arrays with the convenience of Pandas labels, making it ideal for climate data, satellite imagery, experimental measurements, and any data with multiple dimensions.

**Key takeaways:**

- Use DataArrays for single variables, Datasets for collections
- Label-based indexing (.sel) is more readable than position-based
- Leverage automatic alignment for operations between datasets
- Use chunking and Dask for datasets larger than memory
- NetCDF and Zarr are the preferred formats for scientific data
- GroupBy and resample enable powerful temporal aggregations
- Xarray integrates seamlessly with NumPy, Pandas, and Dask

**Next steps:**

- Start with small datasets to learn the API
- Use .sel() and .isel() for intuitive data selection
- Explore groupby operations for categorical analysis
- Learn chunking strategies for your specific use case
- Integrate with domain-specific tools (Cartopy, Dask, etc.)

Xarray transforms complex multidimensional data analysis into intuitive, readable code while maintaining high performance and scalability.
