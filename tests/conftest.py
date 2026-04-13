import pytest
import pandas as pd


@pytest.fixture
def sample_row():
    """Single employee row matching the model's expected column names."""
    return {
        "id": 999,
        "age": 30,
        "genre": "M",
        "revenu_mensuel": 4000,
        "statut_marital": "Célibataire",
        "departement": "Commercial",
        "poste": "Cadre Commercial",
        "nombre_experiences_precedentes": 3,
        "nombre_heures_travailless": 80,
        "annee_experience_totale": 5,
        "annees_dans_l_entreprise": 2,
        "annees_dans_le_poste_actuel": 1,
        "a_quitte_l_entreprise": "Non",
        "nombre_participation_pee": 0,
        "nb_formations_suivies": 2,
        "nombre_employee_sous_responsabilite": 0,
        "distance_domicile_travail": 10,
        "niveau_education": 3,
        "domaine_etude": "Infra & Cloud",
        "frequence_deplacement": "Occasionnel",
        "annees_depuis_la_derniere_promotion": 1,
        "annes_sous_responsable_actuel": 0,
        "satisfaction_employee_environnement": 3,
        "note_evaluation_precedente": 3,
        "niveau_hierarchique_poste": 2,
        "satisfaction_employee_nature_travail": 3,
        "satisfaction_employee_equipe": 3,
        "satisfaction_employee_equilibre_pro_perso": 3,
        "note_evaluation_actuelle": 3,
        "heure_supplementaires": "Non",
        "augementation_salaire_precedente": 5.0,
    }


@pytest.fixture
def sample_data(sample_row):
    """DataFrame with one employee for training tests."""
    return pd.DataFrame([sample_row])
