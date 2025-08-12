# Data Setup Guide

## Required Data Files

This repository does not include large data files due to GitHub size limits. To run the pipeline, you need to set up the following data:

### 1. Radiation Data
**Location**: `data_3years_2018_2020_final/`
**Source**: Copernicus Climate Data Store (CDS)
**Format**: Parquet files with 15-minute resolution
**Coverage**: Germany, 2018-2020

### 2. PV Locations Data
**Location**: `data_pv_lookup_comprehensive/`
**Source**: OpenClimateFix Quartz (OSM-derived)
**Format**: Parquet lookup tables
**Coverage**: 14,861 German PV assets

### 3. Generation Data
**Note**: Synthetic generation is created automatically from radiation data.

## Data Sources

### ERA5 Radiation Data
1. Register at https://cds.climate.copernicus.eu/
2. Download SSRD (Surface Solar Radiation Downwards) data
3. Process using the provided scripts in `src/`

### OSM PV Data
1. Access OpenClimateFix Quartz dataset
2. Extract German PV assets
3. Process using `src/build_comprehensive_pv_lookup.py`

## Directory Structure
```
data_3years_2018_2020_final/
├── monthly_15min_results/
│   └── ssrd_germany_2018_*_15min/
└── quarterly_15min_results/
    └── ssrd_germany_*_*_15min/

data_pv_lookup_comprehensive/
├── primary_lookup.parquet
└── compact_lookup.parquet
```

## Data Processing Scripts

### Radiation Data Processing
The pipeline expects radiation data in the following format:
- **File Structure**: Partitioned Parquet files by year/month
- **Columns**: `time`, `ssrd` (or `ssrd_w_m2`)
- **Resolution**: 15-minute intervals
- **Coverage**: Germany (latitude: 47-55°, longitude: 6-15°)

### PV Location Processing
Use the provided script to build lookup tables:
```bash
python src/build_comprehensive_pv_lookup.py
```

This will:
1. Load OSM PV data from `data_german_pv_scaled/german_pv_osm_scaled.parquet`
2. Apply H3 spatial indexing
3. Match with irradiance grid
4. Create lookup tables in `data_pv_lookup_comprehensive/`

## Data Size Estimates

| **Data Type** | **Size** | **Records** | **Notes** |
|---------------|----------|-------------|-----------|
| Radiation (raw) | ~50 GB | ~90M | Full 2018-2020 dataset |
| Radiation (sampled) | ~2.5 GB | ~4.5M | 5% sampling used in pipeline |
| PV Locations | ~50 MB | 14,861 | OSM-derived assets |
| Predictions | ~100 MB | 3.36M | Pipeline output |

## Alternative Data Sources

### For Testing/Development
If you don't have access to the full datasets, you can:

1. **Use synthetic data**: Modify `baseline_pipeline_profiled.py` to generate synthetic radiation data
2. **Use sample data**: Create smaller sample datasets for testing
3. **Use public datasets**: Find alternative solar radiation datasets

### Public Datasets
- **NASA POWER**: Global solar radiation data
- **ERA5-Land**: Higher resolution reanalysis data
- **Meteosat**: Satellite-based solar radiation
- **PVGIS**: Photovoltaic Geographical Information System

## Troubleshooting

### Common Data Issues

1. **Missing Data Directories**
   ```
   Error: No radiation data found!
   Solution: Ensure data_3years_2018_2020_final/ exists with proper structure
   ```

2. **Incorrect File Format**
   ```
   Error: No irradiance column found in data
   Solution: Check column names (should be 'ssrd' or 'ssrd_w_m2')
   ```

3. **Memory Issues**
   ```
   Error: zsh: killed
   Solution: Reduce sampling rate in _load_radiation_data()
   ```

### Data Validation

Run data validation checks:
```bash
# Check radiation data
python -c "
import pandas as pd
from pathlib import Path
data_path = Path('data_3years_2018_2020_final')
if data_path.exists():
    print('✅ Radiation data directory exists')
    files = list(data_path.rglob('*.parquet'))
    print(f'Found {len(files)} parquet files')
else:
    print('❌ Radiation data directory missing')
"

# Check PV data
python -c "
import pandas as pd
from pathlib import Path
pv_path = Path('data_pv_lookup_comprehensive/primary_lookup.parquet')
if pv_path.exists():
    df = pd.read_parquet(pv_path)
    print(f'✅ PV data: {len(df):,} locations')
else:
    print('❌ PV data missing')
"
```

## Data Pipeline Flow

```
Raw ERA5 Data (NetCDF)
         ↓
    Convert to Parquet
         ↓
   Resample to 15-min
         ↓
   Apply 5% Sampling
         ↓
   Pipeline Processing
         ↓
   Baseline Model Training
         ↓
   Generate Predictions
```

## Performance Considerations

- **Sampling Rate**: 5% provides good balance between speed and accuracy
- **Memory Usage**: ~4.5M records requires ~2GB RAM
- **Processing Time**: ~14 seconds total execution
- **Storage**: ~100MB output file

For production use, consider:
- Increasing sampling rate for better accuracy
- Using full dataset for final training
- Implementing distributed processing for large datasets
