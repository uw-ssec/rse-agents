# Scientific Domain Applications Plugin

Domain-specific scientific computing skills for geospatial analysis, astronomy, climate science, and interactive visualization.

## Overview

This plugin provides expert guidance for domain-specific scientific computing applications, covering specialized libraries and tools used across various scientific disciplines.

**Version:** 0.1.0

**Contents:**
- 0 Agents (coming soon)
- 4 Skills (xarray, geospatial, astropy, holoviz)

## Available Skills

### Xarray for Multidimensional Data

**File:** [skills/xarray-for-multidimensional-data/SKILL.md](skills/xarray-for-multidimensional-data/SKILL.md)

**Description:** Master labeled multidimensional arrays for scientific data analysis with Xarray. Work efficiently with complex datasets having multiple dimensions, coordinates, and metadata.

**Key topics:**
- Xarray DataArray and Dataset structures
- Coordinates and dimensions
- Dimension alignment
- Computation and aggregation
- I/O operations (NetCDF, Zarr, etc.)
- Integration with Dask for large data
- Plotting and visualization
- Time series operations

**When to use:**
- Climate and weather data analysis
- Satellite and remote sensing imagery
- Oceanographic data processing
- Experimental measurements with multiple parameters
- Simulation outputs with complex dimensional structures
- Any multidimensional scientific data

**Includes:** Quick reference card, decision trees, 10 detailed patterns, best practices checklist, real-world examples

### Geospatial Data Processing

**File:** [skills/geospatial-data-processing/SKILL.md](skills/geospatial-data-processing/SKILL.md)

**Description:** Process and analyze geospatial data using GDAL, Rasterio, GeoPandas, and related tools for GIS and remote sensing applications.

**Key topics:**
- GDAL/Rasterio for raster processing
- GeoPandas for vector data
- CRS (Coordinate Reference System) handling
- Reprojection and transformation
- Cloud-optimized GeoTIFF (COG)
- Spatial indexing (R-tree)
- Zonal statistics
- Raster-vector operations

**When to use:**
- GIS data processing and analysis
- Remote sensing and satellite imagery
- Spatial data transformation
- Geographic data visualization
- Environmental and ecological studies

**Status:** Coming soon

### Astropy Fundamentals

**File:** [skills/astropy-fundamentals/SKILL.md](skills/astropy-fundamentals/SKILL.md)

**Description:** Master astronomical computing with Astropy, covering FITS I/O, coordinate systems, units, time handling, and astronomical calculations.

**Key topics:**
- FITS I/O and header manipulation
- Coordinate systems and transformations
- Units and quantities
- Time handling in astronomy
- Tables and datasets
- Photometry basics with Photutils
- Spectroscopy basics with Specutils
- WCS (World Coordinate System)

**When to use:**
- Astronomical data analysis
- Observatory data processing
- Coordinate transformations
- Time series analysis in astronomy
- Photometry and spectroscopy

**Status:** Coming soon

### HoloViz Visualization

**File:** [skills/holoviz-visualizations/SKILL.md](skills/holoviz-visualizations/SKILL.md)

**Description:** Create interactive visualizations and dashboards using the HoloViz ecosystem (hvPlot, Panel, HoloViews, GeoViews, Datashader).

**Key topics:**
- hvPlot for quick interactive plots from Pandas, Xarray, Dask
- Panel for creating dashboards and apps
- HoloViews for declarative data visualization
- GeoViews for geographic data visualization
- Datashader for rendering large datasets
- Lumen for data-driven dashboards from YAML
- Integration with Bokeh, Matplotlib, Plotly
- Deploying visualization apps

**When to use:**
- Interactive exploration of scientific data
- Creating dashboards and web applications
- Visualizing large datasets efficiently
- Geographic data visualization
- Sharing interactive results with collaborators

**Status:** Coming soon

## Architecture and Design

### Scientific Domain Focus

This plugin focuses on domain-specific applications that build upon the foundational Python development skills:

1. **Multidimensional Data** - Xarray for labeled arrays across all domains
2. **Geospatial Computing** - GIS, remote sensing, and spatial analysis
3. **Astronomical Computing** - Observatory data and celestial calculations
4. **Interactive Visualization** - Modern tools for data exploration and communication

### Integration with Other Plugins

- **Python Development Plugin** - Foundational skills (pixi, packaging, testing, code quality)
- **Scientific Computing Plugin** - HPC, parallel computing, numerical methods (planned)

## When to Use This Plugin

Use the Scientific Domain Applications plugin when working on:

- **Climate and weather analysis** with multidimensional datasets
- **Geospatial projects** involving GIS, remote sensing, or spatial analysis
- **Astronomical research** with observatory data and celestial calculations
- **Interactive visualization** needs for scientific data exploration
- **Domain-specific workflows** requiring specialized scientific libraries
- **Data-intensive applications** with complex dimensional structures

## Technologies Covered

### Multidimensional Data
- Xarray - Labeled multidimensional arrays
- NetCDF4 - Network Common Data Form
- Zarr - Cloud-optimized array storage
- Dask - Parallel computing for large arrays

### Geospatial Computing
- GDAL - Geospatial Data Abstraction Library
- Rasterio - Raster data I/O
- GeoPandas - Vector data with Pandas interface
- Shapely - Geometric operations
- Fiona - Vector data I/O
- PyProj - Coordinate transformations

### Astronomical Computing
- Astropy - Core astronomy library
- Photutils - Photometry tools
- Specutils - Spectroscopy tools
- FITS - Flexible Image Transport System
- WCS - World Coordinate System

### Interactive Visualization
- hvPlot - High-level plotting
- Panel - Dashboards and apps
- HoloViews - Declarative visualization
- GeoViews - Geographic visualization
- Datashader - Large dataset rendering
- Bokeh - Interactive plotting backend

## Examples and Use Cases

### Example 1: Climate Data Analysis

```python
import xarray as xr

# Load climate data
ds = xr.open_dataset("temperature.nc", chunks={"time": 365})

# Calculate annual mean
annual_mean = ds["temperature"].resample(time="1Y").mean()

# Calculate anomaly
climatology = ds["temperature"].groupby("time.month").mean(dim="time")
anomaly = ds["temperature"].groupby("time.month") - climatology

# Save results
anomaly.to_netcdf("temperature_anomaly.nc")
```

### Example 2: Geospatial Analysis

```python
import geopandas as gpd
import rasterio

# Load vector data
cities = gpd.read_file("cities.geojson")

# Load raster data
with rasterio.open("elevation.tif") as src:
    elevation = src.read(1)
    
# Reproject to common CRS
cities_projected = cities.to_crs("EPSG:3857")

# Spatial operations
buffer = cities_projected.buffer(10000)  # 10km buffer
```

### Example 3: Interactive Visualization

```python
import hvplot.pandas
import panel as pn

# Create interactive plot
plot = df.hvplot.scatter(x="time", y="temperature", 
                         hover_cols=["location"])

# Create dashboard
dashboard = pn.Column(
    "# Temperature Dashboard",
    plot,
    pn.widgets.Select(name="Location", options=locations)
)

dashboard.show()
```

## Resources

### Xarray
- [Xarray Documentation](https://docs.xarray.dev/)
- [Xarray Tutorial](https://tutorial.xarray.dev/)
- [Pangeo](https://pangeo.io/) - Big data geoscience

### Geospatial
- [GDAL Documentation](https://gdal.org/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GeoPandas Documentation](https://geopandas.org/)

### Astronomy
- [Astropy Documentation](https://docs.astropy.org/)
- [Astropy Tutorials](https://learn.astropy.org/)
- [Photutils](https://photutils.readthedocs.io/)

### Visualization
- [HoloViz](https://holoviz.org/)
- [hvPlot](https://hvplot.holoviz.org/)
- [Panel](https://panel.holoviz.org/)

## Contributing

We welcome contributions to this plugin! You can:

- **Add new skills** - Create focused guides on specific domain topics
- **Enhance existing skills** - Add patterns, examples, or clarifications
- **Add agents** - Create specialized agents for domain-specific workflows
- **Report issues** - Let us know if something is unclear or incorrect

See the main [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Potential Additions

Ideas for new skills or improvements:

- Machine learning for science (scikit-learn, PyTorch)
- Bioinformatics tools (BioPython)
- Chemistry and materials science (RDKit, ASE)
- Time series analysis (statsmodels, Prophet)
- Network analysis (NetworkX)

## Questions or Feedback?

Please open an issue on [GitHub](https://github.com/uw-ssec/rse-agents/issues) with the label `scientific-domain-applications`.
