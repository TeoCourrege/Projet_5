# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer:
    - création de features agrégées
    - flags de risque
    - remplissage des NaN
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Feature: salary_per_age
        if 'revenu_mensuel' in X.columns and 'age' in X.columns:
            X['salary_per_age'] = X['revenu_mensuel'] / X['age']
        
        # Satisfaction générale
        satisfaction_cols = [
            'satisfaction_employee_environnement',
            'satisfaction_employee_nature_travail',
            'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso'
        ]
        present_satisfaction_cols = [c for c in satisfaction_cols if c in X.columns]
        if present_satisfaction_cols:
            X['satisfaction_generale'] = X[present_satisfaction_cols].sum(axis=1)
        
        # High-risk stagnation
        if 'annees_depuis_la_derniere_promotion' in X.columns and 'satisfaction_generale' in X.columns:
            X['high_risk_stagnation'] = (
                (X['annees_depuis_la_derniere_promotion'] >= 2) &
                (X['satisfaction_generale'] < 10)
            ).astype(int)
        
        # Clean inf/-inf
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(X.median(numeric_only=True))
        return X

def encode_dataframe(df: pd.DataFrame):
    """
    Encodage des features catégorielles et binaires
    """
    df_encoded = df.copy()
    
    # Genre
    if 'genre' in df_encoded.columns:
        df_encoded['genre'] = df_encoded['genre'].map({'M': 1, 'F': 0})
    
    # Statut marital
    if 'statut_marital' in df_encoded.columns:
        df_encoded['statut_marital_marie'] = (df_encoded['statut_marital'] == 'Marié(e)').astype(int)
        df_encoded['statut_marital_divorce'] = (df_encoded['statut_marital'] == 'Divorcé(e)').astype(int)
        # Célibataire = 0,0 implicite
        df_encoded.drop(columns=['statut_marital'], inplace=True)
    
    # Poste et domaine d'étude
    if 'poste' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['poste'], prefix='poste', dtype=int)
    if 'domaine_etude' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['domaine_etude'], prefix='domaine_etude', dtype=int)
    
    # Cible
    if 'a_quitte_l_entreprise' in df_encoded.columns:
        df_encoded['a_quitte_l_entreprise'] = df_encoded['a_quitte_l_entreprise'].map({'Oui': 1, 'Non': 0})
    
    # Ordinal encoding
    if 'frequence_deplacement' in df_encoded.columns:
        df_encoded['frequence_deplacement'] = df_encoded['frequence_deplacement'].map({'Aucun': 0, 'Occasionnel': 1, 'Frequent': 2})
    
    # Heures supplémentaires
    if 'heure_supplementaires' in df_encoded.columns:
        df_encoded['heure_supplementaires'] = df_encoded['heure_supplementaires'].map({'Non': 0, 'Oui': 1})
    
    # Remplacer inf/-inf et NA numériques
    df_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    
    return df_encoded