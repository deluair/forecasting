# Examples

This directory contains example scripts demonstrating the forecasting framework's capabilities.

## Available Examples

### Basic Examples

- **basic_forecasting.py** - Introduction to basic forecasting with ARIMA and Prophet
- **competition_workflow.py** - Complete workflow for prediction competitions (Metaculus, GJ Open)

### Advanced Examples

- **advanced_ensemble.py** - Advanced ensemble methods with stacking and weighted averaging
- **econometric_forecasting.py** - Econometric models (VAR, VECM, State Space)
- **bangladesh_economy_forecast.py** - Real-world economic forecasting example

## Running Examples

Each example can be run independently:

```bash
python examples/basic_forecasting.py
python examples/competition_workflow.py
python examples/advanced_ensemble.py
```

## Prerequisites

Make sure you have installed the package with all dependencies:

```bash
pip install -e ".[dev]"
```

Some examples may require additional data files or API keys. Check the comments at the top of each file for specific requirements.

## Example Structure

Each example follows this general structure:

1. **Import required modules**
2. **Load or generate data**
3. **Create and configure forecaster**
4. **Fit model to training data**
5. **Generate forecasts**
6. **Evaluate predictions**
7. **Visualize results**

## Contributing Examples

If you have an interesting use case or example, please contribute! See [../docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines.
