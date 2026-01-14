# Scientific Domain Applications Plugin

Domain-specific scientific computing agents and skills for specialized research areas in astronomy, astrophysics, and multidimensional data analysis.

## Overview

This plugin provides expert guidance for domain-specific scientific computing applications, focusing on astronomical research, astrophysical data analysis, and multidimensional scientific datasets. It combines deep domain knowledge with modern computational tools to support researchers in astronomy, climate science, remote sensing, and related fields.

**Version:** 0.1.0

**Contents:**
- 1 Agent: Astronomy & Astrophysics Expert
- 2 Skills: Astropy Fundamentals, Xarray for Multidimensional Data

## Installation

This plugin is part of the RSE Plugins collection. To use it with Claude Code:

1. Clone the repository:
   ```bash
   git clone https://github.com/uw-ssec/rse-plugins.git
   ```

2. The plugin will be automatically available in the repository's marketplace at:
   ```
   plugins/scientific-domain-applications/
   ```

3. Load the Astronomy & Astrophysics Expert agent or individual skills through Claude Code's plugin interface.

## Plugin Structure

```
scientific-domain-applications/
├── .claude/
│   └── plugin.json                                # Plugin metadata and configuration
├── agents/
│   └── astronomy-astrophysics-expert.md           # Astronomy and astrophysics expert agent
├── skills/
│   ├── astropy-fundamentals/                      # Astropy for astronomical data
│   │   ├── SKILL.md
│   │   └── references/
│   │       ├── COMMON_ISSUES.md
│   │       ├── EXAMPLES.md
│   │       └── PATTERNS.md
│   └── xarray-for-multidimensional-data/         # Xarray for labeled N-D arrays
│       ├── SKILL.md
│       ├── assets/
│       ├── references/
│       │   ├── COMMON_ISSUES.md
│       │   ├── EXAMPLES.md
│       │   └── PATTERNS.md
│       └── scripts/
├── LICENSE -> ../../LICENSE                       # Symlink to main repository license
└── README.md                                     # This file
```

## Available Agents

### Astronomy & Astrophysics Expert

**File:** [agents/astronomy-astrophysics-expert.md](agents/astronomy-astrophysics-expert.md)

**Agent Version:** 2025-11-20

**Description:** Expert astronomer and astrophysicist for observational data analysis, theoretical calculations, and astronomical research. Specializes in AstroPy, FITS data, photometry, spectroscopy, coordinate systems, and time-domain astronomy. Deep knowledge spanning Solar System to cosmology. Use PROACTIVELY for astronomical data processing, telescope observations, celestial mechanics, or astrophysical calculations.

**Integrated Skills:**
- astropy-fundamentals
- xarray-for-multidimensional-data (for spectral cubes and multidimensional astronomical data)

**Capabilities:**

**Astronomical Python Ecosystem:**
- AstroPy Core: Units, time systems (UTC, TAI, TT, TDB, GPS), coordinate systems (ICRS, FK5, Galactic, AltAz), FITS I/O, tables, cosmology, modeling
- Photometry (Photutils): Aperture photometry, PSF photometry, source detection, background estimation, calibration
- Spectroscopy (Specutils): 1D/3D spectral data, line identification, radial velocity, equivalent width, continuum fitting
- High-Precision Timing: Nanosecond-precision timing for pulsar observations, GPS time handling, barycentric corrections, leap second management
- Astrometry: Proper motion, parallax, position matching, coordinate transformations, epoch conversions
- Time-Domain Astronomy: Light curve analysis, period finding, variability metrics, transit detection
- Image Processing: FITS manipulation, cosmic ray rejection, image alignment, flat fielding
- WCS: Pixel-to-sky conversions, projection handling, distortion corrections
- Observing and Planning: Target visibility, airmass calculations, moon separation, twilight duration

**Astronomical Knowledge Domains:**
- Solar System Science: Planetary mechanics, asteroid/comet observations, satellite dynamics, space weather
- Stellar Astrophysics: Spectral classification, HR diagram, binary stars, variable stars, stellar parameters
- Exoplanet Science: Transit photometry, radial velocity, direct imaging, atmospheric characterization
- Galactic Astronomy: Milky Way structure, star clusters, interstellar medium, planetary nebulae
- Extragalactic Astronomy: Galaxy morphology, redshift measurements, AGN identification, gravitational lensing
- Cosmology: Cosmic distance ladder, Hubble diagram, CMB analysis, large-scale structure
- High-Energy Astrophysics: X-ray/gamma-ray analysis, pulsar timing, supernovae, GRBs

**Physical Understanding:**
- Fundamental physics (gravitational dynamics, EM radiation, Doppler shifts, blackbody radiation)
- Observational effects (atmospheric extinction, seeing, detector characteristics, cosmic rays)
- Error analysis (photon counting statistics, Poisson noise, systematic errors, Bayesian estimation)

**Data Sources and Archives:**
- Major sky surveys (SDSS, Pan-STARRS, Gaia, 2MASS, WISE, GALEX, TESS, Kepler)
- Virtual Observatory tools (TAP queries, VO cone searches, SAMP, Aladin, Topcat)
- Archive access (MAST, ESO, IRSA, HEASARC, exoplanet archives)

**When to use:**
- Processing telescope images (FITS files) and performing photometry/astrometry
- Analyzing astronomical spectra and measuring redshifts
- Working with celestial coordinates and coordinate transformations
- Performing high-precision timing analysis for pulsars or transients
- Planning observations and calculating target visibility
- Cross-matching astronomical catalogs
- Building spectral energy distributions from multi-wavelength data
- Analyzing time-series astronomical data (light curves, periodic variables)
- Processing data from space missions or ground-based observatories
- Any astronomical research requiring proper units, coordinates, and time handling

**Decision-Making Framework:**

The agent follows a structured approach for every astronomy task:

1. **Understand Astronomical Context** - Research area, observation type, wavelength regime, data source, scientific goal
2. **Assess Data Characteristics** - What type of observations? What wavelength? What instruments?
3. **Identify Physical Processes** - What astrophysical phenomena are involved? What physics applies?
4. **Choose Analysis Methods** - Which techniques are appropriate (photometry, astrometry, spectroscopy, timing)?
5. **Select Tools** - Which astronomy libraries best fit the need (AstroPy, Photutils, Specutils, etc.)?
6. **Plan Validation** - How to verify correctness (known standards, cross-checks, error propagation)?
7. **Consider Observational Effects** - What corrections are needed (airmass, extinction, instrument response)?

**Quality Assurance:**

Every response includes a self-review checklist covering:
- **Astronomical Correctness** - Units attached to all quantities, coordinate systems specified, time systems properly handled, WCS transformations validated
- **Observational Accuracy** - Calibration steps identified, systematic corrections applied, instrumental effects considered, error propagation included
- **Code Quality** - AstroPy best practices, FITS header handling, proper exception handling, type hints with Quantity types
- **Scientific Rigor** - Assumptions stated, limitations acknowledged, alternative interpretations considered, references provided

## Available Skills

### Astropy Fundamentals

**File:** [skills/astropy-fundamentals/SKILL.md](skills/astropy-fundamentals/SKILL.md)

**Description:** Work with astronomical data using Astropy for FITS file I/O, coordinate transformations, physical units, precise time handling, and catalog operations. Use when processing telescope images, matching celestial catalogs, handling time-series observations, or building photometry/spectroscopy pipelines. Ideal for astronomy research requiring proper unit handling, coordinate frame transformations, and astronomical time scales.

**Key topics:**
- FITS file I/O with headers and WCS
- Physical units and quantities with astronomical constants
- Celestial coordinate systems (ICRS, FK5, Galactic, AltAz) and transformations
- High-precision time handling (UTC, TAI, TT, TDB, GPS) with sub-nanosecond precision
- Astronomical tables and catalogs
- World Coordinate System (WCS) for image astrometry
- Photometry with Photutils (aperture and PSF photometry)
- Spectroscopy with Specutils (1D spectral analysis)
- Coordinate matching and catalog cross-matching
- Observing planning and target visibility

**When to use:**
- Processing FITS files from telescopes, surveys, or simulations
- Performing coordinate transformations between reference frames
- Working with physical quantities requiring dimensional correctness
- Handling astronomical time with multiple time scales and high precision
- Managing catalogs and tables from surveys or observations
- Converting between pixel and sky coordinates using WCS
- Performing aperture or PSF photometry on astronomical images
- Analyzing 1D spectra with wavelength calibration
- Cross-matching sources across multiple catalogs
- Planning observations (rise/set times, airmass, visibility)

**Skill contents:**
- SKILL.md: Main skill guide with quick reference, decision trees, and core concepts
- references/COMMON_ISSUES.md: Troubleshooting guide for FITS I/O, units, coordinates, time, tables, WCS, and performance issues
- references/EXAMPLES.md: Complete workflows for telescope image processing, catalog cross-matching, light curve analysis, multi-wavelength SED construction, spectroscopic redshift measurement, and observability calculation
- references/PATTERNS.md: Advanced patterns for FITS manipulation, units and quantities, coordinates, time, tables, WCS, photometry, and spectroscopy

### Xarray for Multidimensional Data

**File:** [skills/xarray-for-multidimensional-data/SKILL.md](skills/xarray-for-multidimensional-data/SKILL.md)

**Description:** Work with labeled multidimensional arrays for scientific data analysis using Xarray. Use when handling climate data, satellite imagery, oceanographic data, or any multidimensional datasets with coordinates and metadata. Ideal for NetCDF/HDF5 files, time series analysis, and large datasets requiring lazy loading with Dask.

**Key topics:**
- DataArray and Dataset structures for labeled N-dimensional data
- Coordinate systems and dimension labels
- Label-based and position-based indexing
- NetCDF, Zarr, and HDF5 file formats
- Lazy loading with Dask for large datasets
- GroupBy operations and resampling for time series
- Combining datasets (concat, merge, align)
- Interpolation and regridding
- Custom functions with apply_ufunc
- DataTree for hierarchical data organization
- Geospatial operations with rioxarray (CRS-aware raster operations)

**When to use:**
- Working with climate and weather data (time, lat, lon, altitude dimensions)
- Processing satellite and remote sensing imagery
- Analyzing oceanographic data with depth profiles
- Handling experimental measurements with multiple parameters
- Managing simulation outputs with complex dimensional structures
- Building time series that vary across spatial locations
- Working with datasets where tracking dimensions and coordinates is critical
- Processing large datasets that don't fit in memory (using Dask)
- Performing geospatial raster operations with proper CRS handling

**Skill contents:**
- SKILL.md: Main skill guide with quick reference, decision trees, and core concepts covering DataArray, Dataset, coordinates, indexing, DataTree, and ecosystem extensions
- assets/: (Directory for future configuration examples)
- references/COMMON_ISSUES.md: Solutions for memory errors, coordinate misalignment, chunking issues, dimension order confusion, and encoding problems
- references/EXAMPLES.md: Real-world examples including climate data analysis, satellite data processing, oceanographic analysis, multi-model ensemble analysis, time series decomposition, hierarchical climate model data with DataTree, and geospatial satellite processing with rioxarray
- references/PATTERNS.md: Detailed patterns for creating DataArrays/Datasets, reading/writing data, selection/indexing, computation/aggregation, combining datasets, Dask integration, interpolation/regridding, custom functions, working with DataTree, and geospatial operations with rioxarray
- scripts/: (Directory for example scripts)

## Architecture and Design

### Domain-Specific Focus

This plugin focuses on specialized scientific computing domains:

1. **Astronomy and Astrophysics** - Complete toolkit for observational astronomy, from raw telescope data to astrophysical interpretation
2. **Multidimensional Scientific Data** - Labeled array operations for climate, remote sensing, and experimental data
3. **Physical Correctness** - Proper handling of units, coordinates, time systems, and metadata throughout all operations

### Scientific Computing Principles

The plugin follows scientific computing best practices:

1. **Physical Units** - Always attach and propagate physical units to prevent dimensional errors
2. **Coordinate Systems** - Explicit specification of reference frames and coordinate systems
3. **Time Handling** - Proper management of time scales (UTC, TAI, TDB, GPS) with appropriate precision
4. **Metadata Preservation** - Maintain provenance and metadata throughout analysis pipelines
5. **Reproducibility** - Document dependencies, parameters, and processing steps for reproducible research

### Integration Approach

The skills are designed to work together:

- **Astropy + Xarray** - Use Xarray for multidimensional astronomical data (IFU spectral cubes, time-series imaging)
- **Xarray + rioxarray** - Geospatial operations with proper CRS handling for satellite/remote sensing data
- **Cross-Domain** - Skills can be combined for complex workflows (e.g., processing 3D spectral cubes with Xarray, then extracting spectra with Specutils)

## How to Use This Plugin

### Using the Astronomy & Astrophysics Expert Agent

The agent is designed to be used **proactively** for astronomical research and data analysis. It automatically loads relevant skills and provides comprehensive guidance.

Load the agent through Claude Code's interface and use it for:
- End-to-end astronomical data analysis pipelines
- Complex workflows combining photometry, spectroscopy, and time-domain analysis
- Architectural decisions for astronomy software projects
- Learning astronomical data analysis best practices
- Interpreting observations and connecting to physical theory

### Using Individual Skills

Skills can be loaded independently when you need focused expertise:

- **Load astropy-fundamentals** when working with FITS files, astronomical coordinates, time systems, or photometry/spectroscopy
- **Load xarray-for-multidimensional-data** when working with climate data, satellite imagery, or any labeled multidimensional arrays

Skills provide:
- Quick reference cards for common tasks
- Decision trees for choosing approaches
- Configuration templates and examples
- Troubleshooting guides
- Best practices and patterns
- Real-world complete examples

### Skill Usage Pattern

```bash
# Example: Loading a skill for a specific task
# In Claude Code, reference the skill:
# "Load the astropy-fundamentals skill"

# Or directly invoke skill guidance:
# "Using astropy-fundamentals skill, help me process this FITS image"

# For multidimensional data:
# "Load the xarray-for-multidimensional-data skill"
# "Using xarray skill, help me analyze this NetCDF climate dataset"
```

## When to Use This Plugin

Use the Scientific Domain Applications plugin when working on:

- **Astronomical research** requiring FITS data processing, photometry, spectroscopy, or astrometry
- **Observational astronomy** with telescope data from ground-based or space missions
- **Time-domain astronomy** including variable stars, exoplanet transits, or transient events
- **High-precision timing** for pulsars, VLBI, or multi-telescope coordination
- **Climate science** with multidimensional climate model outputs or reanalysis data
- **Remote sensing** processing satellite imagery with multiple spectral bands and time series
- **Oceanographic data** with depth profiles and spatiotemporal measurements
- **Geospatial analysis** requiring CRS-aware raster operations
- **Multi-model ensemble analysis** combining outputs from different models or resolutions
- **Any research** requiring proper handling of physical units, coordinates, and metadata

## Technologies Covered

### Astronomy and Astrophysics

**Core Libraries:**
- **Astropy** - Fundamental astronomy library (units, coordinates, time, FITS, tables, WCS, cosmology)
- **Photutils** - Source detection and photometry
- **Specutils** - Spectroscopic data analysis
- **Astroquery** - Accessing online astronomical databases

**Data Formats:**
- **FITS** - Flexible Image Transport System (astronomy standard)
- **ASDF** - Advanced Scientific Data Format (next-generation astronomy format)

**Applications:**
- Observational astronomy
- Photometry and spectroscopy
- Astrometry and celestial mechanics
- Time-domain astronomy
- Multi-wavelength analysis

### Multidimensional Scientific Data

**Core Libraries:**
- **Xarray** - Labeled multidimensional arrays
- **Dask** - Parallel computing and lazy evaluation
- **rioxarray** - Geospatial raster operations with CRS awareness
- **xESMF** - Universal regridder for geospatial data
- **Geocube** - Vector to raster conversion

**Data Formats:**
- **NetCDF** - Network Common Data Form (climate/ocean standard)
- **Zarr** - Cloud-optimized chunked storage
- **HDF5** - Hierarchical Data Format
- **GeoTIFF** - Georeferenced raster imagery

**Applications:**
- Climate and weather modeling
- Satellite and remote sensing
- Oceanography
- Experimental data analysis
- Geospatial analysis

### Supporting Technologies

**Scientific Python Stack:**
- NumPy - Numerical computing
- Pandas - Tabular data
- Matplotlib - Visualization
- SciPy - Scientific algorithms

**Performance:**
- Dask - Parallel and distributed computing
- Numba - JIT compilation

**Geospatial:**
- GDAL/Rasterio - Geospatial data abstraction
- Geopandas - Vector geospatial data
- Cartopy - Cartographic projections

## Integration with Other Plugins

This plugin is part of the RSE Plugins ecosystem and complements other scientific computing plugins:

**Current RSE Plugins:**
- **scientific-python-development** - Modern Scientific Python development practices (pixi, pytest, packaging, code quality)
- **holoviz-visualization** - Interactive visualization with Panel, hvPlot, HoloViews

**Potential Integration Patterns:**

1. **With scientific-python-development:**
   - Use pixi to manage astronomy package dependencies (astropy, photutils, etc.)
   - Apply pytest patterns to test astronomical calculations
   - Package astronomy tools for distribution on PyPI

2. **With holoviz-visualization:**
   - Create interactive astronomical image viewers with Panel
   - Build interactive light curve explorers with hvPlot
   - Visualize multidimensional climate data with HoloViews

3. **Cross-Domain Workflows:**
   - Process astronomical spectral cubes with Xarray, visualize with HoloViz
   - Build distributable astronomy packages with modern packaging tools
   - Create interactive climate data exploration tools

## Examples and Use Cases

### Example 1: Complete Astronomical Image Processing

Process a telescope image from raw FITS to source catalog:

```python
from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder, CircularAperture, aperture_photometry
from astropy.stats import sigma_clipped_stats
import astropy.units as u

# Load FITS file
with fits.open('observation.fits') as hdul:
    image_data = hdul[0].data
    header = hdul[0].header
    wcs = WCS(header)

# Background subtraction
mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
image_subtracted = image_data - median

# Source detection
daofind = DAOStarFinder(fwhm=4.0, threshold=5.0 * std)
sources = daofind(image_subtracted)

# Aperture photometry
positions = [(s['xcentroid'], s['ycentroid']) for s in sources]
apertures = CircularAperture(positions, r=5.0)
phot_table = aperture_photometry(image_subtracted, apertures)

# Convert to sky coordinates
sky_coords = wcs.pixel_to_world(phot_table['xcenter'], phot_table['ycenter'])
```

(See astropy-fundamentals skill for complete pipeline)

### Example 2: Multi-Model Climate Ensemble Analysis

Analyze and compare multiple climate model outputs:

```python
import xarray as xr
import glob

# Load multiple model outputs
model_files = glob.glob("models/model_*.nc")
models = [xr.open_dataset(f, chunks={"time": 365}) for f in model_files]

# Combine into ensemble
ensemble = xr.concat(models, dim="model")

# Calculate ensemble statistics
ensemble_mean = ensemble.mean(dim="model")
ensemble_std = ensemble.std(dim="model")

# Identify robust changes (high model agreement)
change = ensemble.sel(time=slice("2080", "2100")).mean() - \
         ensemble.sel(time=slice("1980", "2000")).mean()
agreement = (np.sign(change) == np.sign(change.mean(dim="model"))).sum(dim="model") / len(models)
robust_change = change.mean(dim="model").where(agreement > 0.8)
```

(See xarray-for-multidimensional-data skill for complete workflow)

### Example 3: High-Precision Pulsar Timing

Process pulsar observations with nanosecond timing precision:

```python
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

# Pulsar coordinates
pulsar = SkyCoord(ra='05:34:31.95', dec='+22:00:52.1', unit=(u.hourangle, u.deg))
observatory = EarthLocation.of_site('Arecibo')

# High-precision observation times
obs_times = Time(['2024-01-15T12:00:00.000000000',
                  '2024-01-15T12:00:01.000000000'],
                 format='isot', scale='utc', precision=9)

# Convert to TDB at Solar System barycenter
ltt_bary = obs_times.light_travel_time(pulsar, 'barycentric', location=observatory)
times_bary = obs_times.tdb + ltt_bary

# Apply dispersion measure correction
dm = 56.7  # pc cm^-3
freq_mhz = 1400.0
dispersion_delay = (dm / (0.000241 * freq_mhz**2)) * u.second
times_corrected = times_bary + dispersion_delay
```

(See astronomy-astrophysics-expert agent for complete timing analysis)

### Example 4: Geospatial Satellite Time Series

Process Landsat time series with CRS-aware operations:

```python
import rioxarray
import xarray as xr

# Load multi-temporal satellite data
scenes = [rioxarray.open_rasterio(f"landsat_{year}.tif") for year in [2020, 2021, 2022]]

# Calculate NDVI for each scene
def calc_ndvi(scene):
    red = scene.sel(band=3).astype(float)
    nir = scene.sel(band=4).astype(float)
    return ((nir - red) / (nir + red)).rio.write_crs(scene.rio.crs)

ndvi_series = [calc_ndvi(s) for s in scenes]
ndvi_ts = xr.concat(ndvi_series, dim="time")
ndvi_ts["time"] = pd.date_range("2020-01-01", periods=3, freq="1Y")

# Reproject to WGS84 for analysis
ndvi_wgs84 = ndvi_ts.rio.reproject("EPSG:4326")

# Calculate vegetation change
change = ndvi_ts.isel(time=-1) - ndvi_ts.isel(time=0)
change = change.rio.write_crs(ndvi_ts.rio.crs)

# Clip to study area
aoi = ndvi_wgs84.rio.clip_box(minx=-122, miny=37, maxx=-121, maxy=38)
```

(See xarray-for-multidimensional-data skill for geospatial patterns)

## Resources

### Astronomy and Astrophysics

**Official Documentation:**
- [Astropy Documentation](https://docs.astropy.org/en/stable/)
- [Astropy Tutorials](https://learn.astropy.org/)
- [Photutils](https://photutils.readthedocs.io/)
- [Specutils](https://specutils.readthedocs.io/)
- [Astroquery](https://astroquery.readthedocs.io/)

**Community:**
- [Astropy Discourse](https://community.openastronomy.org/c/astropy)
- [GitHub Issues](https://github.com/astropy/astropy/issues)

**Data Archives:**
- [MAST](https://mast.stsci.edu/) - Mikulski Archive for Space Telescopes
- [ESO Archive](https://archive.eso.org/)
- [IRSA](https://irsa.ipac.caltech.edu/) - Infrared Science Archive
- [VizieR](https://vizier.cds.unistra.fr/) - Astronomical Catalogs
- [SIMBAD](http://simbad.cds.unistra.fr/) - Astronomical Database

### Multidimensional Data Analysis

**Official Documentation:**
- [Xarray Documentation](https://docs.xarray.dev/)
- [Xarray Tutorial](https://tutorial.xarray.dev/)
- [Dask Documentation](https://docs.dask.org/)
- [rioxarray Documentation](https://corteva.github.io/rioxarray/)

**Community:**
- [Xarray GitHub Discussions](https://github.com/pydata/xarray/discussions)
- [Pangeo Community](https://pangeo.io/)

**Geospatial Extensions:**
- [xESMF](https://xesmf.readthedocs.io/) - Universal regridder
- [Geocube](https://corteva.github.io/geocube/) - Vector to raster
- [xarray-spatial](https://xarray-spatial.readthedocs.io/) - Spatial analytics
- [Salem](https://salem.readthedocs.io/) - Geolocation operations

**Domain-Specific:**
- [Pangeo Gallery](https://gallery.pangeo.io/) - Geoscience examples
- [Earth and Environmental Data Science](https://earth-env-data-science.github.io/)

## Contributing

We welcome contributions to this plugin! You can:

- **Add new agents** - Create specialized agents for other scientific domains (biology, materials science, etc.)
- **Add new skills** - Develop skills for related technologies (e.g., SunPy for solar physics, yt for volumetric data)
- **Enhance existing skills** - Add patterns, examples, troubleshooting tips, or clarifications
- **Improve the agent** - Suggest enhancements to the Astronomy & Astrophysics Expert agent
- **Report issues** - Let us know if something is unclear, incorrect, or outdated
- **Share examples** - Contribute real-world usage examples and case studies

See the main repository [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

### Skill Structure Guidelines

When contributing a new skill, follow this structure:

```
skills/your-skill-name/
├── SKILL.md                    # Main skill guide (required)
├── assets/                     # Configuration files, templates, examples
│   └── example-config.yml
├── scripts/                    # Runnable example scripts (optional)
│   └── example.py
└── references/                 # Deep-dive documentation (optional)
    ├── PATTERNS.md
    ├── EXAMPLES.md
    └── COMMON_ISSUES.md
```

### Ideas for New Skills

Potential additions that would enhance this plugin:

**Astronomy-Related:**
- **sunpy-solar-physics** - Solar physics data analysis with SunPy
- **yt-volumetric-data** - 3D volumetric data visualization (simulation data, spectral cubes)
- **radio-astronomy** - Radio interferometry data (CASA integration, visibility data)
- **gaia-astrometry** - Gaia mission data analysis and proper motion studies
- **time-series-photometry** - Advanced light curve analysis (TESS, Kepler pipelines)

**Geospatial and Climate:**
- **climate-data-operators** - Climate Data Operators (CDO) integration
- **geospatial-vector-analysis** - Geopandas for vector geospatial data
- **atmospheric-science** - Atmospheric data analysis (vertical profiles, radiosonde data)
- **ocean-modeling** - Ocean model output analysis

**Cross-Domain:**
- **hdf5-advanced** - Advanced HDF5 file handling for large scientific datasets
- **scientific-visualization-3d** - 3D visualization for scientific data (Mayavi, PyVista)

## Questions or Feedback?

- **Issues**: Open an issue on [GitHub](https://github.com/uw-ssec/rse-plugins/issues) with the label `scientific-domain-applications`
- **Discussions**: Start a discussion on [GitHub Discussions](https://github.com/uw-ssec/rse-plugins/discussions)
- **Pull Requests**: Submit improvements via [pull requests](https://github.com/uw-ssec/rse-plugins/pulls)

## License

This plugin is part of the RSE Plugins project and is licensed under the BSD-3-Clause License. See [LICENSE](../../LICENSE) for details.
