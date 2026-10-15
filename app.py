from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)
df = model = None
features = []
target = None
feature_types = {}
metrics = None
comparison = None

def page(**kwargs):
    return render_template("index.html", **kwargs)

@app.route("/")
def home():
    return page()

@app.route("/upload", methods=["POST"])
def upload():
    global df, model, features, target, metrics, comparison
    file = request.files.get("dataset")
    if not file or not file.filename:
        return page(error="Choose a CSV file first.")
    try:
        df = pd.read_csv(file)
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty or len(df.columns) < 2:
            return page(error="Your CSV must contain data and at least two columns.")
        model = None
        metrics = comparison = None
        preview = df.head(8).to_html(index=False, classes="data-table", border=0)
        return page(columns=list(df.columns), preview=preview, rows=len(df), uploaded=True,
                    filename=file.filename)
    except Exception as e:
        return page(error=f"Could not read CSV: {e}")

@app.route("/train", methods=["POST"])
def train():
    global model, features, target, feature_types, metrics, comparison
    if df is None:
        return page(error="Upload a CSV first.")
    target = request.form.get("target")
    features = [f for f in request.form.getlist("features") if f in df.columns and f != target]
    if target not in df.columns or not features:
        return page(error="Choose one target and at least one feature.")
    try:
        data = df[features + [target]].copy()
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target])
        X, y = data[features], data[target]
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = [c for c in features if c not in numeric]
        transformers = []
        if numeric:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric))
        if categorical:
            cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
            transformers.append(("cat", cat_pipe, categorical))
        prep = ColumnTransformer(transformers)
        model = Pipeline([("prep", prep), ("lr", LinearRegression())])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        metrics = {"r2": r2_score(y_test, pred) if len(y_test) > 1 else float("nan"),
                   "mse": mse, "rmse": np.sqrt(mse),
                   "train": len(X_train), "test": len(X_test)}
        feature_types = {c: ("numeric" if c in numeric else "categorical") for c in features}
        comparison = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": pred}).round(2).to_html(
            index=False, classes="data-table", border=0)
        return page(columns=list(df.columns), preview=df.head(8).to_html(index=False, classes="data-table", border=0),
                    rows=len(df), uploaded=True, trained=True, metrics=metrics,
                    comparison=comparison, feature_types=feature_types)
    except Exception as e:
        return page(error=f"Training failed: {e}. Make sure the target column is numeric.")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return page(error="Train the model first.")
    try:
        row = {}
        for col, kind in feature_types.items():
            val = request.form.get(col, "").strip()
            if not val:
                return page(error=f"Enter a value for {col}.")
            row[col] = float(val) if kind == "numeric" else val
        value = float(model.predict(pd.DataFrame([row], columns=features))[0])
        return page(trained=True, metrics=metrics, comparison=comparison,
                    feature_types=feature_types, prediction=value, prediction_target=target,
                    columns=list(df.columns), preview=df.head(8).to_html(index=False, classes="data-table", border=0),
                    rows=len(df), uploaded=True)
    except Exception as e:
        return page(error=f"Prediction failed: {e}")

if __name__ == "__main__":
    app.run(debug=True)
