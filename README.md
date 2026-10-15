# RegressIQ Multiple Linear Regression App

1. Open this folder in PyCharm.
2. In the terminal, create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install packages:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Run:
   ```powershell
   python app.py
   ```
5. Open http://127.0.0.1:5000 in your browser.
6. Upload a CSV, choose the numeric target and input features, train, inspect R²/MSE/RMSE, and try a prediction.

Categorical columns are automatically one-hot encoded. Numeric missing feature values are filled with the median; missing target values are excluded. This is a local learning app; current data/model stay in memory while Flask runs.
