# Energy Uncertainty Index: A Time Series & Forecasting Project

## Authors
* **Alessandro Macchi** - github.com/alessandro-macchi
* **Simona Wang** - github.com/simona-wang

## Overview
In an increasingly interconnected world, understanding the ripple effects of global uncertainty is paramount. The Macchi-Wang Time Series project is a comprehensive analytical pipeline designed to forecast complex macroeconomic trends. By pitting traditional statistical foundations against modern machine learning techniques, this project evaluates how effectively different models can anticipate market dynamics over time.

## The Data Ecosystem
Our analysis is anchored in a robust dataset spanning nearly three decades, from January 1996 to November 2023. We focus on four critical global indicators:

* **Global Economic Uncertainty Index** (GDP-weighted)
* **Geopolitical Risk Index**
* **CPU Index**
* **Europe Brent Spot Price** (Oil)

To expose the true underlying signals rather than transient noise, the data undergoes rigorous preprocessing. This includes addressing missing values and applying logarithmic and differenced transformations. The refined dataset is then evaluated sequentially, preserving the chronological integrity required for time series analysis with an 80/20 train-test split.

## Project Architecture
The repository is structured to provide a clear, reproducible workflow from raw data to final predictions:

* **Core Analysis**: A central Jupyter Notebook that orchestrates the end-to-end pipeline, seamlessly weaving together data processing, visual exploration, and model validation.
* **Preprocessing Engine**: Dedicated modules that handle the heavy lifting of data cleaning, feature engineering, and proper sequential splitting.
* **Visualization Suite**: Custom analytical tools that generate cross-correlation matrices and temporal visualizations, offering a clear narrative of the data's historical behavior.
* **Modeling Framework**: A dedicated environment housing our training algorithms, feature lagging tools, and forecasting functions.

## Modeling Approach
We benchmark a spectrum of predictive models to capture both linear trends and intricate, non-linear complexities within the data:

1. **The Baseline**: Random Walk with Drift, establishing a foundational metric for baseline market behavior.
2. **Statistical Modeling**: ARIMA (AutoRegressive Integrated Moving Average), featuring both algorithmically optimized and manually calibrated configurations.
3. **Machine Learning**: A Random Forest Regressor, deployed to uncover deep, non-linear relationships between our macroeconomic indicators.

## Performance and Insights
To ensure our models are evaluated under realistic forecasting conditions, we utilize an expanding window forecast methodology. Model accuracy is rigorously measured across standard error metrics, including Mean Squared Forecast Error (MSFE) and Mean Absolute Percentage Error (MAPE).

**The Verdict**: The machine learning approach yields the most compelling results. The Random Forest model successfully captures the complex, hidden patterns within geopolitical and economic data. It consistently minimizes forecast error across all tracked metrics, decisively outperforming both the standard Random Walk baseline and the traditional parametric ARIMA models.
