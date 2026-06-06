import os
import sys
import math
import pandas as pd
from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from data_analyzer import load_data, clean_data, analyze_data

app = Flask(__name__)

CSV_INPUT  = os.path.join(os.path.dirname(__file__), "sample_data.csv")
CSV_OUTPUT = os.path.join(os.path.dirname(__file__), "output", "dados_limpos.csv")


def _safe(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def run_pipeline():
    df_raw = load_data(CSV_INPUT)
    df_clean, report = clean_data(df_raw.copy())
    stats = analyze_data(df_clean)

    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    df_clean.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    outlier_idx = set(stats.get("_outlier_indices", []))
    records = []
    for i, row in df_clean.iterrows():
        rec = {"_outlier": i in outlier_idx}
        for col in df_clean.columns:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif hasattr(val, "isoformat"):
                rec[col] = val.strftime("%Y-%m-%d")
            else:
                rec[col] = val
        records.append(rec)

    clean_stats = {
        k: {sk: _safe(sv) for sk, sv in v.items()}
        for k, v in stats.items()
        if not k.startswith("_") and isinstance(v, dict)
    }

    dtypes = {
        col: {"dtype": str(dtype), "nulos": int(df_clean[col].isnull().sum())}
        for col, dtype in df_clean.dtypes.items()
    }

    return {
        "report":         report,
        "stats":          clean_stats,
        "dtypes":         dtypes,
        "columns":        df_clean.columns.tolist(),
        "records":        records,
        "total_outliers": len(outlier_idx),
        "pandas_version": pd.__version__,
    }


@app.route("/")
def index():
    return render_template("index.html", **run_pipeline())


@app.route("/api/data")
def api_data():
    return jsonify(run_pipeline())


if __name__ == "__main__":
    print("\n  Rodando em: http://localhost:5000\n")
    app.run(debug=True, port=5000)
