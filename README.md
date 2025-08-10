# Sirius Assignment - Data Engineering Project

A reproducible data engineering project with Poetry for dependency management and Docker for containerization.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- Docker (optional, for containerized development)

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

4. **Start Jupyter Lab**
   ```bash
   poetry run jupyter lab
   ```

### Docker Development

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access Jupyter Lab**
   - Open your browser and go to `http://localhost:8888`
   - No token or password required in development mode

3. **Stop the containers**
   ```bash
   docker-compose down
   ```

## 📁 Project Structure

```
Sirius_Assignment/
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Poetry configuration
├── poetry.lock            # Locked dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── README.md              # This file
├── notebooks/             # Jupyter notebooks
│   └── data_engineering.ipynb
├── src/                   # Source code
│   └── __init__.py
└── tests/                 # Test files
    └── __init__.py
```

## 📦 Dependencies

### Core Data Science
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning
- **scipy**: Scientific computing

### Visualization
- **plotly**: Interactive plotting
- **bokeh**: Interactive web plotting

### Database Connectivity
- **sqlalchemy**: SQL toolkit and ORM
- **psycopg2-binary**: PostgreSQL adapter
- **pymongo**: MongoDB driver

### Development
- **jupyter**: Jupyter notebooks
- **ipykernel**: IPython kernel for Jupyter
- **requests**: HTTP library

## 🔧 Development

### Adding New Dependencies
```bash
# Add a production dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name
```

### Running Tests
```bash
poetry run pytest
```

### Code Formatting
```bash
# Install black for code formatting
poetry add --group dev black

# Format code
poetry run black src/ tests/
```

## 🐳 Docker Commands

### Build Image
```bash
docker build -t sirius-assignment .
```

### Run Container
```bash
docker run -p 8888:8888 -v $(pwd):/app sirius-assignment
```

### Interactive Shell
```bash
docker run -it sirius-assignment /bin/bash
```

## 📝 Notes

- The `csdapi` module mentioned in the notebook is not available on PyPI. You may need to:
  - Install it from a different source
  - Create a custom implementation
  - Replace it with an alternative library

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
