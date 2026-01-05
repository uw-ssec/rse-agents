# Xarray Patterns

## Pattern 1: Creating DataArrays and Datasets

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

## Pattern 2: Reading and Writing Data

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

## Pattern 3: Selection and Indexing

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

## Pattern 4: Computation and Aggregation

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

## Pattern 5: Combining Datasets

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

## Pattern 6: Dask Integration for Large Data

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

## Pattern 7: Interpolation and Regridding

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

## Pattern 8: Custom Functions with apply_ufunc

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

