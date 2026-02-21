![Ames, Iowa](NorthridgeHeights.png)
<sup>[Northridge Heights, Ames, Iowa, source: homes.com](https://www.homes.com/local-guide/ames-ia/northridge-heights-neighborhood/?dk=3b98p63qzdh6s)</sup>

# House Prices (Ames) – Ridge Regression Pipeline

This project trains a Ridge regression model to predict house prices on the Ames Housing dataset[^1]

## The notebook focuses on:

* Researching the dataset and finding a possible correlation between the characteristics of houses and their prices
* Feature selection by correlation[^2]
* Cross-validated tuning of Ridge regularization[^3] with GridSearchCV

---

## Files

* Datasets in folder Datasets:
  * train.csv
  * test.csv
  * test_with_predictions.csv (original test columns + SalePrice)
* house_prices.ipynb
* README.md
* requirements.txt
* images

---

## How the model works

1. Target transform
  * y = log1p(SalePrice) to reduce skew and stabilize error.
2. Feature engineering (inside the pipeline)
  * TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
  * HouseAge = YrSold - YearBuilt
  * RemodAge = YrSold - YearRemodAdd
  * TotalBathrooms = FullBath + 0.5HalfBath + BsmtFullBath + 0.5BsmtHalfBath
  * HasPool, HasGarage as simple binary indicators
3. Skew handling (inside the pipeline)
  * Automatically applies log1p to highly skewed numeric columns (CV-safe)
4. Preprocessing
  * Numeric: median imputation + standard scaling
  * Categorical: most-frequent imputation + one-hot encoding
5. Model
  * Ridge regression with alpha tuned via GridSearchCV using 5-fold KFold CV and RMSE.

---

## Environment

Python 3.9+ recommended.

---

## How to work with jupyter notebook house_price_prediction.ipynb:
1. Clone the repository with:\
 bash
 ```
 git clone https://github.com/lmacakova/prices.git
 cd prices
 ```
2. Create a virtual environment:\
 bash
 ```
 python -m venv .venv
 ```
     macOS/Linux
 ```
 ource .venv/bin/activate
 ```
     Windows PowerShell
 ```
 .venv\Scripts\Activate.ps1
 ```
3. Install requirements:\
 bash
 ```
 pip install --upgrade pip
 pip install -r requirements.txt
 ```
4. Open the notebook in VS Code or Visual Studio Code and select Python (applied) as the kernel.

---

## Resources:

[^1]:    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
[^2]:    https://en.wikipedia.org/wiki/Correlation
[^3]:    https://www.geeksforgeeks.org/machine-learning/what-is-ridge-regression  
