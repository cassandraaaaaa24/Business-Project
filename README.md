# SALESCAST — Sales Prediction & Analytics

A modern Python application for predicting sales using machine learning. Built with Tkinter for the GUI, Pandas for data handling, Scikit-learn for modeling, and Matplotlib for visualization.

## Features

**Core Functionality:**
- Load and analyze CSV files containing sales data
- Select feature columns and define target column (sales)
- Choose from multiple regression algorithms (Linear Regression, Ridge, Lasso, RandomForest, SVR)
- Train models with optional feature scaling
- Generate batch predictions on loaded data or custom single-value predictions
- Visualize sales trends over time (requires a 'Date' column)

**Data & Analysis:**
- Display data summary statistics (row/column counts, missing values, mean, std dev)
- Apply feature scaling (StandardScaler) for improved model performance
- Export predictions with original data to CSV file
- Compare different regression algorithms
- MAE, MSE, RMSE, and R² metrics for model evaluation

**User Experience:**
- Modern dark-themed interface with intuitive layout
- Real-time status updates
- Organized panels for data source, features, preprocessing, actions, and output
- Quick-access exit button

## Requirements

- Python 3.6+
- Pandas
- Scikit-learn
- Matplotlib
- Tkinter (usually included with Python)

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main2.py
   ```

2. **Load Data:** Click "Load CSV" to select your sales data file. A summary of the data (rows, columns, statistics) will display automatically.

3. **Select Features:** In the "Feature Columns" panel, select the columns you believe influence sales (CTRL+click for multiple selections).

4. **Configure Target:** Enter the target column name (typically "Sales").

5. **Choose Algorithm:** In the "Algorithm" section, select a regression model:
   - **LinearRegression** — Fast, interpretable, works well for linear relationships
   - **Ridge** — Adds regularization to prevent overfitting
   - **Lasso** — Feature selection through regularization
   - **RandomForest** — Handles non-linear patterns, more robust
   - **SVR** — Support Vector Regression for complex boundaries

6. **Preprocessing (Optional):** Check "Apply Feature Scaling" if your features have different scales (recommended for better accuracy).

7. **Train Model:** Click "▶ Train Model" to train the selected model. Results show MAE, MSE, RMSE, and R² score.

8. **Generate Batch Predictions:** Click "◆ Run Predictions" to predict sales values for all rows in your data. Predictions are stored as "Predicted_Sales" column.

9. **Custom Predictions:** In the "Custom Prediction" card, enter values for each feature and click "📊 Predict" to get a quick prediction without modifying your data.

10. **Export Results:** Click "💾 Export CSV" to save your data with predictions to a new CSV file.

11. **View Trends:** Click "⟜ Sales Trend" to visualize sales trends over time (requires a 'Date' column in YYYY-MM-DD format).

12. **Exit:** Click "✕ Exit" to close the application.

## Data Format

Your CSV should include:
- **Numeric feature columns** — The input variables that influence sales (e.g., advertising budget, store size, employees)
- **Numeric target column** — The sales value you want to predict
- **Date column (optional)** — In YYYY-MM-DD format for trend visualization
- **Missing values allowed** — The app automatically fills missing numeric values with column means

Example structure:
```
Date,Marketing_Budget,Employees,Store_Size,Sales
2023-01-01,5000,10,1500,25000
2023-01-02,5500,10,1500,26500
2023-01-03,6000,12,1600,28000
```

## Model Details

- **Available Algorithms:**
  - **LinearRegression** — Classical linear model, fast and interpretable
  - **Ridge** — L2 regularization to handle multicollinearity
  - **Lasso** — L1 regularization for automatic feature selection
  - **RandomForest** — Ensemble method, handles non-linearity well
  - **SVR** — Support Vector Regression, effective for complex patterns
- **Train/Test Split:** 80% training, 20% testing
- **Metrics Reported:**
  - **MAE (Mean Absolute Error)** — Average absolute prediction error (in original units)
  - **MSE (Mean Squared Error)** — Average of squared prediction errors (lower is better)
  - **RMSE (Root Mean Squared Error)** — Square root of MSE in original units
  - **R² Score** — Proportion of variance explained (0-1, higher is better)
- **Feature Scaling:** Optional StandardScaler normalization (applied to all algorithms when enabled)

## Notes

- The app uses only **numeric columns** for features. Categorical data (text) is not automatically encoded.
- Predictions are stored as the **"Predicted_Sales"** column in the data.
- For trend visualization, ensure your data has a **'Date'** column in YYYY-MM-DD format.
- Missing values are filled with **column means** before training and prediction.
- Feature scaling is **optional but recommended** when features have different ranges (e.g., budget in thousands vs. employee count in tens).