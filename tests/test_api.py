import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock


def test_gradio_app_loads():
    """Verify the Gradio Blocks app object builds without errors."""
    with patch("src.db.database.init_db"), \
         patch("src.db.database.seed_admin"), \
         patch("src.db.database.joblib") as mock_jl:
        mock_jl.load.return_value = MagicMock()
        import importlib
        import app as app_module
        importlib.reload(app_module)
        assert app_module.demo is not None


def test_authenticate_valid_user():
    """Mocked authentication returns True for valid credentials."""
    from src.db.database import hash_password, verify_password

    hashed = hash_password("test123")
    assert verify_password("test123", hashed) is True


def test_authenticate_invalid_password():
    """verify_password rejects wrong password."""
    from src.db.database import hash_password, verify_password

    hashed = hash_password("correct")
    assert verify_password("wrong", hashed) is False


def test_batch_predict_rejects_bad_format():
    """batch_predict returns error message for unsupported file format."""
    from unittest.mock import MagicMock

    fake_file = MagicMock()
    fake_file.name = "data.xml"

    with patch("src.db.database._get_pipeline") as mock_pipe, \
         patch("src.db.database.SessionLocal") as mock_sess:
        mock_sess.return_value = MagicMock()
        from src.db.database import batch_predict
        result = batch_predict(fake_file)
        assert "non supporté" in result


def test_batch_predict_requires_id_column():
    """batch_predict returns error when 'id' column is missing."""
    from unittest.mock import MagicMock

    df = pd.DataFrame({"age": [30], "genre": ["M"]})

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        fake_file = MagicMock()
        fake_file.name = tmp_path

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_sess:
            mock_sess.return_value = MagicMock()
            from src.db.database import batch_predict
            result = batch_predict(fake_file)
            assert "id" in result
    finally:
        os.unlink(tmp_path)

def test_predict_success():
    from src.db.database import predict

    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [1]
    mock_pipeline.predict_proba.return_value = [[0.2, 0.8]]

    with patch("src.db.database._get_pipeline", return_value=mock_pipeline), \
         patch("src.db.database.SessionLocal") as mock_session:

        session = MagicMock()
        mock_session.return_value = session

        result = predict(
            employee_id=1, age=30, genre="M", revenu_mensuel=3000,
            statut_marital="Célibataire", departement="Commercial", poste="Cadre Commercial",
            nombre_experiences_precedentes=2, nombre_heures_travailless=40,
            annee_experience_totale=5, annees_dans_l_entreprise=2,
            annees_dans_le_poste_actuel=1,
            nombre_participation_pee=0, nb_formations_suivies=1,
            nombre_employee_sous_responsabilite=0,
            distance_domicile_travail=10, niveau_education=3,
            domaine_etude="Infra & Cloud", frequence_deplacement="Occasionnel",
            annees_depuis_la_derniere_promotion=1,
            annes_sous_responsable_actuel=0,
            satisfaction_employee_environnement=3,
            note_evaluation_precedente=3,
            niveau_hierarchique_poste=2,
            satisfaction_employee_nature_travail=3,
            satisfaction_employee_equipe=3,
            satisfaction_employee_equilibre_pro_perso=3,
            note_evaluation_actuelle=3,
            heure_supplementaires="Non",
            augementation_salaire_precedente=5.0
        )

        assert result == 1


def test_predict_error():
    from src.db.database import predict

    with patch("src.db.database._get_pipeline", side_effect=Exception("fail")), \
         patch("src.db.database.SessionLocal") as mock_session:

        session = MagicMock()
        mock_session.return_value = session

        result = predict(
            employee_id=1, age=30, genre="M", revenu_mensuel=3000,
            statut_marital="Célibataire", departement="Commercial", poste="Cadre Commercial",
            nombre_experiences_precedentes=2, nombre_heures_travailless=40,
            annee_experience_totale=5, annees_dans_l_entreprise=2,
            annees_dans_le_poste_actuel=1,
            nombre_participation_pee=0, nb_formations_suivies=1,
            nombre_employee_sous_responsabilite=0,
            distance_domicile_travail=10, niveau_education=3,
            domaine_etude="Infra & Cloud", frequence_deplacement="Occasionnel",
            annees_depuis_la_derniere_promotion=1,
            annes_sous_responsable_actuel=0,
            satisfaction_employee_environnement=3,
            note_evaluation_precedente=3,
            niveau_hierarchique_poste=2,
            satisfaction_employee_nature_travail=3,
            satisfaction_employee_equipe=3,
            satisfaction_employee_equilibre_pro_perso=3,
            note_evaluation_actuelle=3,
            heure_supplementaires="Non",
            augementation_salaire_precedente=5.0)

        assert "Erreur" in result
        session.rollback.assert_called()