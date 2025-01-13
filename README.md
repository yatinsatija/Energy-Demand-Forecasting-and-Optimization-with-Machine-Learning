# Energy-Demand-Forecasting-and-Optimization-with-Machine-Learning

## Overview
This project explores advanced machine learning and optimization techniques to forecast electricity demand and optimize grid resource allocation. It leverages time-series data and external variables such as weather and calendar effects to enhance predictive accuracy. The work demonstrates the application of supervised learning models, deep learning architectures, and optimization algorithms for energy management.

## Key Features
- **Time-Series Analysis:** Seasonal decomposition and exponential smoothing to identify demand trends and cycles.
- **Machine Learning Models:** Classification models (Naive Bayes, Decision Tree, SVM) and regression models (Linear Regression, Neural Network) for electricity demand forecasting.
- **Optimization:** Linear programming to optimize energy resource allocation during high-demand periods.
- **Visualization:** Time-series plots, seasonal decomposition, and model performance metrics for effective reporting.

## Dataset
The dataset used for this project is sourced from [Kaggle's Electricity Load Forecasting dataset](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting). It contains hourly electricity demand data from 2015 to 2020, with features such as:
- **nat_demand:** Natural electricity demand (target variable)
- **temp, humidity, wind_speed:** Weather conditions
- **hour, weekday, holiday:** Temporal variables

## Tools and Libraries
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Time-Series Analysis:** Statsmodels
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow/Keras
- **Optimization:** SciPy

## Methodology
### 1. Data Preprocessing
- Cleaned and imputed missing values.
- Extracted temporal features and normalized data for specific models.

### 2. Time-Series Analysis
- Explored trends, seasonality, and residuals using seasonal decomposition.
- Applied exponential smoothing for noise reduction.

### 3. Model Development
- **Classification Models:** Evaluated Naive Bayes, Decision Tree, and SVM using accuracy, precision, recall, and ROC-AUC.
- **Regression Models:** Trained Linear Regression and Neural Networks, assessed using MSE and R².

### 4. Optimization
- Used linear programming to manage grid overload scenarios, minimizing electricity shedding while maintaining stability.

## Results
- **Best Classification Model:** SVM with 88.7% accuracy.
- **Best Regression Model:** Neural Network with an MSE of 17,784.
- **Optimization Outcome:** Successfully minimized electricity shedding during 589 high-demand periods.

## Key Insights
- External variables (e.g., weather, holidays) significantly enhance forecasting.
- SVM and Neural Networks are the most effective models for this task.
- Optimization methods ensure grid stability during peak loads.

## Future Work
- Incorporate renewable energy data to account for variability in sustainable resources.
- Explore advanced deep learning architectures like LSTMs for sequential forecasting.
- Integrate real-time data streams for dynamic predictions and optimizations.