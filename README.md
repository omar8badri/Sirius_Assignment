# Solar PV Prediction Pipeline

A comprehensive solar photovoltaic (PV) power prediction pipeline using real radiation data and machine learning models. This project implements a baseline linear model for solar generation forecasting with comprehensive profiling and logging.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- Docker (optional, for containerized execution)

### Single Command Execution

**Run the complete baseline pipeline with one command:**

```bash
make full-run
```

This will:
- ✅ Load radiation data (5% sampling for efficiency)
- ✅ Load comprehensive PV locations (14,861 real OSM assets)
- ✅ Create synthetic generation data based on real radiation
- ✅ Apply feature engineering (6 features)
- ✅ Train baseline linear model on 3.36M samples
- ✅ Generate predictions with excellent performance
- ✅ Save `predictions.parquet` and comprehensive profiling logs

### Output Files

After running `make full-run`, you'll get:

| **File** | **Size** | **Purpose** |
|----------|----------|-------------|
| `predictions.parquet` | ~100 MB | **Main output** (3.36M prediction records) |
| `runtime.log` | ~7 KB | **Detailed execution log** with timestamps |
| `pipeline_output/pipeline_summary.json` | ~500 B | **Profiling summary** with step timings |
| `pipeline_output/profiling_report.txt` | ~3 KB | **Detailed profiling** (cProfile output) |
| `pipeline_output/baseline_metrics.json` | ~100 B | **Model performance metrics** |
| `pipeline_output/baseline_model_info.json` | ~500 B | **Model details and coefficients** |

### Performance Results

**Baseline Model Performance:**
- **MAE**: 137.39 MW
- **RMSE**: 229.85 MW  
- **R²**: 0.978 (excellent fit)

**Execution Profiling:**
- **Total Time**: ~14 seconds
- **Data Loading**: ~10s (70%) - largest bottleneck
- **Model Training**: ~2.4s (17%)
- **Feature Engineering**: ~0.4s (3%)

## 📁 Project Structure

```
Sirius_Assignment/
├── src/
│   ├── baseline_pipeline_profiled.py    # Main baseline pipeline with profiling
│   ├── baseline_pipeline_extended.py    # Extended baseline with more data
│   ├── main_pipeline.py                 # Full pipeline (baseline + candidate models)
│   ├── baseline_model_fixed.py          # Baseline model implementation
│   ├── candidate_models_simple.py       # Random Forest and Neural Network models
│   ├── validation_pipeline.py           # Cross-validation framework
│   ├── scalability_analysis.py          # Performance analysis
│   └── pv_locations/                    # PV data processing modules
├── config/                              # Configuration files
├── tests/                               # Test files
├── notebooks/                           # Jupyter notebooks
├── Makefile                             # Single command interface
├── Dockerfile                           # Container configuration
├── pyproject.toml                       # Poetry configuration
└── README.md                            # This file
```

## 🔧 Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sirius_Assignment
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

### Docker Development

1. **Build and run with Docker**
   ```bash
   docker build -t solar-pv-pipeline .
   docker run --rm -v $(PWD):/workspace solar-pv-pipeline make full-run
   ```

## 📊 Data Sources

### Radiation Data
- **Source**: ERA5 SSRD (Surface Solar Radiation Downwards) from Copernicus CDS
- **Coverage**: Germany, 2018-2020 (limited due to computational constraints)
- **Resolution**: 15-minute intervals (resampled from hourly)
- **Format**: NetCDF → Parquet with solar-geometry-aware interpolation

### PV Locations
- **Source**: OpenStreetMap (OSM) via OpenClimateFix Quartz
- **Coverage**: 14,861 real German PV assets
- **Features**: Latitude, longitude, capacity, installation dates
- **Processing**: H3 spatial indexing and irradiance grid matching

### Solar Generation
- **Source**: Synthetic generation based on real radiation patterns
- **Method**: Physics-based model with realistic parameters
- **Parameters**: 15% system efficiency, 12% capacity factor
- **Note**: Real ENTSO-E data limited to 7 weeks, synthetic data used for full coverage

## 🧠 Model Architecture

### Baseline Model
- **Algorithm**: Linear Regression (`P_t = α + βI_t + γX_t`)
- **Features**: 
  - Irradiance (W/m²)
  - Cyclical time encoding (hour, month)
  - Daytime indicator
- **Training**: 3.36M samples, 6 features
- **Performance**: R² = 0.978

### Feature Engineering
- **Time Features**: Hour, month, day-of-year
- **Cyclical Encoding**: sin/cos transformations for temporal features
- **Daytime Filtering**: Remove night-time data (irradiance < 10 W/m²)
- **Normalization**: Standard scaling for numerical features

### Performance Optimization

- **Data Sampling**: Adjust sampling rate in `_load_radiation_data()`
- **Chunked Processing**: Large datasets are processed in chunks
- **Memory Management**: Pipeline includes garbage collection and memory monitoring

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- **ERA5**: Hersbach et al. (2020) - The ERA5 global reanalysis
- **OSM PV Data**: OpenClimateFix Quartz dataset
- **Solar Modeling**: Standard PV system modeling approaches
