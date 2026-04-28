# Frequentist and Bayesian Approaches in Modeling the Energy Uncertainty Index

**Author:** Alessandro Macchi

## 📌 Project Overview
This project evaluates whether a Bayesian inferential framework delivers better forecasting performance relative to the frequentist approach when modeling the Energy Uncertainty Index (EUI). The analysis is contextualized by recent major global disputes affecting energy supply, such as the Russian invasion of Ukraine and the conflict between the United States and Iran to control the Strait of Hormuz.

The project explores both univariate and multivariate specifications to determine under what conditions the Bayesian approach provides a tangible advantage over traditional frequentist estimators.

## 📂 Project Structure
The final comprehensive academic report containing:
  1. Introduction
  2. Data Exploration and Preprocessing
  3. Models Implementation
  4. Diagnostics
  5. Results Evaluation
  6. Conclusion

**Diebold-Mariano Results Summary:**
* **Absolute Loss (MAE):** The BVAR(2) model is significantly more accurate than the frequentist VAR(2) at the 10% significance level (DM statistic: `1.7960`, p-value: `0.0725`). This suggests that the Bayesian framework's primary gain is concentrated in reducing moderate-sized errors.
* **Squared Loss (RMSE):** No significant predictive difference was detected between the two models (DM statistic: `0.5442`, p-value: `0.5863`), meaning both models are similarly affected by large, extreme outliers.

**Conclusion:**
The Bayesian approach does not dominate by construction. Its advantage materializes when the model is more complex. In the sparse univariate case, frequentist and Bayesian estimators converge to the same answer. Future extensions could explore introducing informed priors in place of a diffuse specification or implementing a time-varying parameter VAR to capture evolving cross-variable relationships.

## 📚 References
1. Litterman, R. B. (1986). *Forecasting with bayesian vector autoregressions five years of experience*. Journal of Business & Economic Statistics, 4(1):25-38.
2. Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy*. Journal of Business & Economic Statistics, 13(3):253-263.
3. Koop, G., & Korobilis, D. (2010). *Bayesian multivariate time series methods for empirical macroeconomics*. Foundations and Trends in Econometrics, 3(4):267-358.
