import pandas as pd
import joblib


def predict_new(csv_path, model_path="src/model/model.pkl", id_col="id"):
    df = pd.read_csv(csv_path)

    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in CSV")

    ids = df[id_col].copy()
    df_features = df.drop(columns=[id_col])

    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(df_features)

    return pd.DataFrame({id_col: ids, "prediction": predictions})


if __name__ == "__main__":
    preds_df = predict_new(csv_path="data/test/test.csv")
    print(preds_df.head())
