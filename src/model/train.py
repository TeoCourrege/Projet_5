import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def merge_raw_data(
    csv_output="data/processed/df_final.csv",
    eval_path="data/raw/extrait_eval.csv",
    sirh_path="data/raw/extrait_sirh.csv",
    sondage_path="data/raw/extrait_sondage.csv",
):
    df_eval = pd.read_csv(eval_path)
    df_sirh = pd.read_csv(sirh_path)
    df_sondage = pd.read_csv(sondage_path)

    df_eval["augementation_salaire_precedente"] = (
        df_eval["augementation_salaire_precedente"]
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )

    df_eval["eval_number"] = df_eval["eval_number"].str.replace("E_", "").astype(int)
    df_merged = df_sirh.merge(df_sondage, left_on="id_employee", right_on="code_sondage", how="inner")
    df_merged = df_merged.merge(df_eval, left_on="id_employee", right_on="eval_number", how="inner")
    df_merged["id"] = df_merged["id_employee"]
    df_merged.drop(columns=["id_employee", "code_sondage", "eval_number"], inplace=True)
    df_merged.to_csv(csv_output, index=False)
    return df_merged


def train_pipeline(csv_path="data/processed/df_final.csv", model_path="src/model/model.pkl"):
    df = pd.read_csv(csv_path)

    target_col = "a_quitte_l_entreprise"
    X = df.drop(columns=[target_col, "id"])
    y = df[target_col].map({"Oui": 1, "Non": 0})

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    label_features = ["genre", "frequence_deplacement", "heure_supplementaires"]
    onehot_features = ["statut_marital", "poste", "domaine_etude"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("label", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), label_features),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_features),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            min_samples_split=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    print(f"Pipeline trained and saved: {model_path}")
    return pipeline


if __name__ == "__main__":
    merge_raw_data()
    train_pipeline()
