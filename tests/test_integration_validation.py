import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.api.validation import validate_employee
from pydantic import ValidationError


class TestValidationIntegration:
    """Integration tests for Pydantic validation with database operations."""

    @pytest.fixture
    def valid_employee_dict(self):
        """Valid employee data as dictionary."""
        return {
            "employee_id": 100,
            "age": 35,
            "genre": "F",
            "statut_marital": "Marié(e)",
            "departement": "Consulting",
            "poste": "Consultant",
            "domaine_etude": "Transformation Digitale",
            "niveau_education": 4,
            "niveau_hierarchique_poste": 3,
            "annee_experience_totale": 12.0,
            "annees_dans_l_entreprise": 6.0,
            "annees_dans_le_poste_actuel": 3.0,
            "annees_depuis_la_derniere_promotion": 2.0,
            "annes_sous_responsable_actuel": 3.0,
            "nombre_experiences_precedentes": 4,
            "revenu_mensuel": 6000.0,
            "augementation_salaire_precedente": 8.0,
            "nombre_participation_pee": 3,
            "nombre_heures_travailless": 45.0,
            "heure_supplementaires": "Oui",
            "distance_domicile_travail": 25.0,
            "frequence_deplacement": "Frequent",
            "nb_formations_suivies": 8,
            "nombre_employee_sous_responsabilite": 2,
            "satisfaction_employee_environnement": 4,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 4,
            "satisfaction_employee_equilibre_pro_perso": 2,
            "note_evaluation_precedente": 4,
            "note_evaluation_actuelle": 4,
        }

    def test_predict_with_validation_success(self, valid_employee_dict):
        """Test that predict function validates data successfully."""
        from src.db.database import predict

        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [0]
        mock_pipeline.predict_proba.return_value = [[0.7, 0.3]]

        with patch("src.db.database._get_pipeline", return_value=mock_pipeline), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, int)
            assert result in [0, 1]

    def test_predict_with_invalid_age(self, valid_employee_dict):
        """Test that predict rejects invalid age with validation error."""
        from src.db.database import predict

        valid_employee_dict["age"] = 15

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result
            assert "age" in result.lower()

    def test_predict_with_invalid_genre(self, valid_employee_dict):
        """Test that predict rejects invalid genre."""
        from src.db.database import predict

        valid_employee_dict["genre"] = "X"

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result

    def test_predict_with_experience_exceeding_age(self, valid_employee_dict):
        """Test that predict rejects experience > age."""
        from src.db.database import predict

        valid_employee_dict["age"] = 25
        valid_employee_dict["annee_experience_totale"] = 30.0

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result
            assert "Experience cannot exceed age" in result

    def test_predict_with_unrealistic_distance(self, valid_employee_dict):
        """Test that predict rejects unrealistic distance."""
        from src.db.database import predict

        valid_employee_dict["distance_domicile_travail"] = 1500.0

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result
            assert "Distance unrealistic" in result

    def test_predict_with_too_few_hours(self, valid_employee_dict):
        """Test that predict rejects too few working hours."""
        from src.db.database import predict

        valid_employee_dict["nombre_heures_travailless"] = 3.0

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result
            assert "Too few working hours" in result

    def test_predict_with_negative_salary(self, valid_employee_dict):
        """Test that predict rejects negative salary."""
        from src.db.database import predict

        valid_employee_dict["revenu_mensuel"] = -1000.0

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result

    def test_predict_with_invalid_satisfaction_range(self, valid_employee_dict):
        """Test that predict rejects satisfaction outside 1-4 range."""
        from src.db.database import predict

        valid_employee_dict["satisfaction_employee_environnement"] = 5

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_session:

            session = MagicMock()
            mock_session.return_value = session

            result = predict(**valid_employee_dict)

            assert isinstance(result, str)
            assert "Erreur de validation" in result

    def test_batch_predict_with_validation(self, valid_employee_dict):
        """Test batch_predict with valid data."""
        from src.db.database import batch_predict

        df = pd.DataFrame([valid_employee_dict])
        df["id"] = df["employee_id"]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            fake_file = MagicMock()
            fake_file.name = tmp_path

            mock_pipeline = MagicMock()
            mock_pipeline.predict.return_value = [1]
            mock_pipeline.predict_proba.return_value = [[0.3, 0.7]]

            with patch("src.db.database._get_pipeline", return_value=mock_pipeline), \
                 patch("src.db.database.SessionLocal") as mock_sess:
                mock_sess.return_value = MagicMock()

                result = batch_predict(fake_file)

                assert isinstance(result, pd.DataFrame)
                assert "id" in result.columns
                assert "prediction" in result.columns
                assert len(result) == 1
        finally:
            os.unlink(tmp_path)

    def test_validation_with_all_departements(self):
        """Test validation accepts all valid departements."""
        base_data = {
            "employee_id": 1,
            "age": 30,
            "genre": "M",
            "statut_marital": "Célibataire",
            "poste": "Manager",
            "domaine_etude": "Autre",
            "niveau_education": 3,
            "niveau_hierarchique_poste": 2,
            "annee_experience_totale": 10.0,
            "annees_dans_l_entreprise": 5.0,
            "annees_dans_le_poste_actuel": 2.0,
            "annees_depuis_la_derniere_promotion": 1.0,
            "annes_sous_responsable_actuel": 2.0,
            "nombre_experiences_precedentes": 3,
            "revenu_mensuel": 5000.0,
            "augementation_salaire_precedente": 5.0,
            "nombre_participation_pee": 2,
            "nombre_heures_travailless": 40.0,
            "heure_supplementaires": "Non",
            "distance_domicile_travail": 15.0,
            "frequence_deplacement": "Occasionnel",
            "nb_formations_suivies": 5,
            "nombre_employee_sous_responsabilite": 0,
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
        }

        for dept in ["Commercial", "Consulting", "Ressources Humaines"]:
            test_data = base_data.copy()
            test_data["departement"] = dept
            employee = validate_employee(test_data)
            assert employee.departement == dept

    def test_validation_with_all_statut_marital(self):
        """Test validation accepts all valid marital statuses."""
        base_data = {
            "employee_id": 1,
            "age": 30,
            "genre": "F",
            "departement": "Commercial",
            "poste": "Cadre Commercial",
            "domaine_etude": "Marketing",
            "niveau_education": 3,
            "niveau_hierarchique_poste": 2,
            "annee_experience_totale": 10.0,
            "annees_dans_l_entreprise": 5.0,
            "annees_dans_le_poste_actuel": 2.0,
            "annees_depuis_la_derniere_promotion": 1.0,
            "annes_sous_responsable_actuel": 2.0,
            "nombre_experiences_precedentes": 3,
            "revenu_mensuel": 5000.0,
            "augementation_salaire_precedente": 5.0,
            "nombre_participation_pee": 2,
            "nombre_heures_travailless": 40.0,
            "heure_supplementaires": "Non",
            "distance_domicile_travail": 15.0,
            "frequence_deplacement": "Occasionnel",
            "nb_formations_suivies": 5,
            "nombre_employee_sous_responsabilite": 0,
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
        }

        for status in ["Célibataire", "Marié(e)", "Divorcé(e)"]:
            test_data = base_data.copy()
            test_data["statut_marital"] = status
            employee = validate_employee(test_data)
            assert employee.statut_marital == status

    def test_validation_with_boundary_education_levels(self):
        """Test validation with boundary education levels."""
        base_data = {
            "employee_id": 1,
            "age": 30,
            "genre": "M",
            "statut_marital": "Célibataire",
            "departement": "Commercial",
            "poste": "Manager",
            "domaine_etude": "Autre",
            "niveau_hierarchique_poste": 2,
            "annee_experience_totale": 10.0,
            "annees_dans_l_entreprise": 5.0,
            "annees_dans_le_poste_actuel": 2.0,
            "annees_depuis_la_derniere_promotion": 1.0,
            "annes_sous_responsable_actuel": 2.0,
            "nombre_experiences_precedentes": 3,
            "revenu_mensuel": 5000.0,
            "augementation_salaire_precedente": 5.0,
            "nombre_participation_pee": 2,
            "nombre_heures_travailless": 40.0,
            "heure_supplementaires": "Non",
            "distance_domicile_travail": 15.0,
            "frequence_deplacement": "Occasionnel",
            "nb_formations_suivies": 5,
            "nombre_employee_sous_responsabilite": 0,
            "satisfaction_employee_environnement": 3,
            "satisfaction_employee_nature_travail": 3,
            "satisfaction_employee_equipe": 3,
            "satisfaction_employee_equilibre_pro_perso": 3,
            "note_evaluation_precedente": 3,
            "note_evaluation_actuelle": 3,
        }

        for level in [1, 2, 3, 4, 5]:
            test_data = base_data.copy()
            test_data["niveau_education"] = level
            employee = validate_employee(test_data)
            assert employee.niveau_education == level
