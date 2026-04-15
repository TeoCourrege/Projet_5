import pytest
from pydantic import ValidationError
from src.api.schemas import EmployeeInput
from src.api.validation import validate_employee


class TestEmployeeInputValidation:
    """Test suite for EmployeeInput Pydantic model validation."""

    @pytest.fixture
    def valid_employee_data(self):
        """Fixture providing valid employee data."""
        return {
            "employee_id": 1,
            "age": 30,
            "genre": "M",
            "statut_marital": "Célibataire",
            "departement": "Commercial",
            "poste": "Cadre Commercial",
            "domaine_etude": "Infra & Cloud",
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

    def test_valid_employee_input(self, valid_employee_data):
        """Test that valid employee data passes validation."""
        employee = EmployeeInput(**valid_employee_data)
        assert employee.employee_id == 1
        assert employee.age == 30
        assert employee.genre == "M"

    def test_validate_employee_function(self, valid_employee_data):
        """Test the validate_employee wrapper function."""
        employee = validate_employee(valid_employee_data)
        assert isinstance(employee, EmployeeInput)
        assert employee.age == 30

    # ======================
    # FIELD TYPE TESTS
    # ======================

    def test_employee_id_negative_fails(self, valid_employee_data):
        """Test that negative employee_id is rejected."""
        valid_employee_data["employee_id"] = -1
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "employee_id" in str(exc_info.value)

    def test_age_below_minimum_fails(self, valid_employee_data):
        """Test that age below 16 is rejected."""
        valid_employee_data["age"] = 15
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "age" in str(exc_info.value)

    def test_age_above_maximum_fails(self, valid_employee_data):
        """Test that age above 70 is rejected."""
        valid_employee_data["age"] = 71
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "age" in str(exc_info.value)

    def test_age_below_18_fails_custom_validator(self, valid_employee_data):
        """Test custom validator rejecting age below 18."""
        valid_employee_data["age"] = 17
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "must be adult" in str(exc_info.value)

    # ======================
    # LITERAL FIELD TESTS
    # ======================

    def test_invalid_genre_fails(self, valid_employee_data):
        """Test that invalid genre is rejected."""
        valid_employee_data["genre"] = "X"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "genre" in str(exc_info.value)

    def test_valid_genre_female(self, valid_employee_data):
        """Test that 'F' genre is accepted."""
        valid_employee_data["genre"] = "F"
        employee = EmployeeInput(**valid_employee_data)
        assert employee.genre == "F"

    def test_invalid_statut_marital_fails(self, valid_employee_data):
        """Test that invalid marital status is rejected."""
        valid_employee_data["statut_marital"] = "Pacsé"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "statut_marital" in str(exc_info.value)

    def test_valid_statut_marital_divorced(self, valid_employee_data):
        """Test that 'Divorcé(e)' is accepted."""
        valid_employee_data["statut_marital"] = "Divorcé(e)"
        employee = EmployeeInput(**valid_employee_data)
        assert employee.statut_marital == "Divorcé(e)"

    def test_invalid_departement_fails(self, valid_employee_data):
        """Test that invalid departement is rejected."""
        valid_employee_data["departement"] = "IT"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "departement" in str(exc_info.value)

    def test_valid_departement_consulting(self, valid_employee_data):
        """Test that 'Consulting' departement is accepted."""
        valid_employee_data["departement"] = "Consulting"
        employee = EmployeeInput(**valid_employee_data)
        assert employee.departement == "Consulting"

    def test_invalid_poste_fails(self, valid_employee_data):
        """Test that invalid poste is rejected."""
        valid_employee_data["poste"] = "Developer"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "poste" in str(exc_info.value)

    def test_valid_poste_tech_lead(self, valid_employee_data):
        """Test that 'Tech Lead' is accepted."""
        valid_employee_data["poste"] = "Tech Lead"
        employee = EmployeeInput(**valid_employee_data)
        assert employee.poste == "Tech Lead"

    def test_invalid_domaine_etude_fails(self, valid_employee_data):
        """Test that invalid study domain is rejected."""
        valid_employee_data["domaine_etude"] = "Sciences"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "domaine_etude" in str(exc_info.value)

    def test_invalid_heure_supplementaires_fails(self, valid_employee_data):
        """Test that invalid overtime value is rejected."""
        valid_employee_data["heure_supplementaires"] = "Maybe"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "heure_supplementaires" in str(exc_info.value)

    def test_invalid_frequence_deplacement_fails(self, valid_employee_data):
        """Test that invalid travel frequency is rejected."""
        valid_employee_data["frequence_deplacement"] = "Toujours"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "frequence_deplacement" in str(exc_info.value)

    # ======================
    # RANGE VALIDATION TESTS
    # ======================

    def test_niveau_education_below_minimum_fails(self, valid_employee_data):
        """Test that education level below 1 is rejected."""
        valid_employee_data["niveau_education"] = 0
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "niveau_education" in str(exc_info.value)

    def test_niveau_education_above_maximum_fails(self, valid_employee_data):
        """Test that education level above 5 is rejected."""
        valid_employee_data["niveau_education"] = 6
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "niveau_education" in str(exc_info.value)

    def test_niveau_hierarchique_below_minimum_fails(self, valid_employee_data):
        """Test that hierarchical level below 1 is rejected."""
        valid_employee_data["niveau_hierarchique_poste"] = 0
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "niveau_hierarchique_poste" in str(exc_info.value)

    def test_niveau_hierarchique_above_maximum_fails(self, valid_employee_data):
        """Test that hierarchical level above 10 is rejected."""
        valid_employee_data["niveau_hierarchique_poste"] = 11
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "niveau_hierarchique_poste" in str(exc_info.value)

    def test_satisfaction_below_minimum_fails(self, valid_employee_data):
        """Test that satisfaction below 1 is rejected."""
        valid_employee_data["satisfaction_employee_environnement"] = 0
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "satisfaction_employee_environnement" in str(exc_info.value)

    def test_satisfaction_above_maximum_fails(self, valid_employee_data):
        """Test that satisfaction above 4 is rejected."""
        valid_employee_data["satisfaction_employee_nature_travail"] = 5
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "satisfaction_employee_nature_travail" in str(exc_info.value)

    def test_note_evaluation_below_minimum_fails(self, valid_employee_data):
        """Test that evaluation note below 1 is rejected."""
        valid_employee_data["note_evaluation_precedente"] = 0
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "note_evaluation_precedente" in str(exc_info.value)

    def test_note_evaluation_above_maximum_fails(self, valid_employee_data):
        """Test that evaluation note above 4 is rejected."""
        valid_employee_data["note_evaluation_actuelle"] = 5
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "note_evaluation_actuelle" in str(exc_info.value)

    # ======================
    # NUMERIC FIELD TESTS
    # ======================

    def test_negative_experience_fails(self, valid_employee_data):
        """Test that negative experience is rejected."""
        valid_employee_data["annee_experience_totale"] = -1
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "annee_experience_totale" in str(exc_info.value)

    def test_negative_years_in_company_fails(self, valid_employee_data):
        """Test that negative years in company is rejected."""
        valid_employee_data["annees_dans_l_entreprise"] = -1
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "annees_dans_l_entreprise" in str(exc_info.value)

    def test_negative_revenue_fails(self, valid_employee_data):
        """Test that negative monthly revenue is rejected."""
        valid_employee_data["revenu_mensuel"] = -100
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "revenu_mensuel" in str(exc_info.value)

    def test_negative_salary_increase_fails(self, valid_employee_data):
        """Test that negative salary increase is rejected."""
        valid_employee_data["augementation_salaire_precedente"] = -5
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "augementation_salaire_precedente" in str(exc_info.value)

    def test_working_hours_above_maximum_fails(self, valid_employee_data):
        """Test that working hours above 100 is rejected."""
        valid_employee_data["nombre_heures_travailless"] = 101
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "nombre_heures_travailless" in str(exc_info.value)

    def test_distance_above_maximum_fails(self, valid_employee_data):
        """Test that distance above 300 is rejected."""
        valid_employee_data["distance_domicile_travail"] = 301
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "distance_domicile_travail" in str(exc_info.value)

    # ======================
    # CUSTOM VALIDATOR TESTS
    # ======================

    def test_experience_exceeds_age_fails(self, valid_employee_data):
        """Test custom validator: experience cannot exceed age."""
        valid_employee_data["age"] = 25
        valid_employee_data["annee_experience_totale"] = 30.0
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "Experience cannot exceed age" in str(exc_info.value)

    def test_distance_unrealistic_fails(self, valid_employee_data):
        """Test custom validator: distance above 1000 is unrealistic."""
        valid_employee_data["distance_domicile_travail"] = 1001
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "Distance unrealistic" in str(exc_info.value)

    def test_too_few_working_hours_fails(self, valid_employee_data):
        """Test custom validator: working hours below 5 is rejected."""
        valid_employee_data["nombre_heures_travailless"] = 4
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "Too few working hours" in str(exc_info.value)

    # ======================
    # MISSING FIELD TESTS
    # ======================

    def test_missing_required_field_fails(self, valid_employee_data):
        """Test that missing required field is rejected."""
        del valid_employee_data["age"]
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "age" in str(exc_info.value)

    def test_missing_employee_id_fails(self, valid_employee_data):
        """Test that missing employee_id is rejected."""
        del valid_employee_data["employee_id"]
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "employee_id" in str(exc_info.value)

    # ======================
    # BOUNDARY VALUE TESTS
    # ======================

    def test_age_at_minimum_boundary(self, valid_employee_data):
        """Test that age=18 (minimum valid) is accepted."""
        valid_employee_data["age"] = 18
        valid_employee_data["annee_experience_totale"] = 0.0
        employee = EmployeeInput(**valid_employee_data)
        assert employee.age == 18

    def test_age_at_maximum_boundary(self, valid_employee_data):
        """Test that age=70 (maximum) is accepted."""
        valid_employee_data["age"] = 70
        employee = EmployeeInput(**valid_employee_data)
        assert employee.age == 70

    def test_niveau_education_at_boundaries(self, valid_employee_data):
        """Test education level at boundaries (1 and 5)."""
        valid_employee_data["niveau_education"] = 1
        employee1 = EmployeeInput(**valid_employee_data)
        assert employee1.niveau_education == 1

        valid_employee_data["niveau_education"] = 5
        employee2 = EmployeeInput(**valid_employee_data)
        assert employee2.niveau_education == 5

    def test_zero_values_where_allowed(self, valid_employee_data):
        """Test that zero values are accepted where allowed."""
        valid_employee_data["nombre_experiences_precedentes"] = 0
        valid_employee_data["nb_formations_suivies"] = 0
        valid_employee_data["nombre_employee_sous_responsabilite"] = 0
        valid_employee_data["nombre_participation_pee"] = 0
        employee = EmployeeInput(**valid_employee_data)
        assert employee.nombre_experiences_precedentes == 0
        assert employee.nb_formations_suivies == 0

    def test_working_hours_at_minimum_boundary(self, valid_employee_data):
        """Test that working hours=5 (minimum valid) is accepted."""
        valid_employee_data["nombre_heures_travailless"] = 5.0
        employee = EmployeeInput(**valid_employee_data)
        assert employee.nombre_heures_travailless == 5.0

    def test_working_hours_at_maximum_boundary(self, valid_employee_data):
        """Test that working hours=100 (maximum) is accepted."""
        valid_employee_data["nombre_heures_travailless"] = 100.0
        employee = EmployeeInput(**valid_employee_data)
        assert employee.nombre_heures_travailless == 100.0

    # ======================
    # EDGE CASE TESTS
    # ======================

    def test_all_satisfactions_at_minimum(self, valid_employee_data):
        """Test that all satisfaction fields at minimum (1) are accepted."""
        valid_employee_data["satisfaction_employee_environnement"] = 1
        valid_employee_data["satisfaction_employee_nature_travail"] = 1
        valid_employee_data["satisfaction_employee_equipe"] = 1
        valid_employee_data["satisfaction_employee_equilibre_pro_perso"] = 1
        employee = EmployeeInput(**valid_employee_data)
        assert employee.satisfaction_employee_environnement == 1

    def test_all_satisfactions_at_maximum(self, valid_employee_data):
        """Test that all satisfaction fields at maximum (4) are accepted."""
        valid_employee_data["satisfaction_employee_environnement"] = 4
        valid_employee_data["satisfaction_employee_nature_travail"] = 4
        valid_employee_data["satisfaction_employee_equipe"] = 4
        valid_employee_data["satisfaction_employee_equilibre_pro_perso"] = 4
        employee = EmployeeInput(**valid_employee_data)
        assert employee.satisfaction_employee_environnement == 4

    def test_high_values_accepted(self, valid_employee_data):
        """Test that high but valid values are accepted."""
        valid_employee_data["revenu_mensuel"] = 50000.0
        valid_employee_data["annee_experience_totale"] = 40.0
        valid_employee_data["age"] = 60
        valid_employee_data["nombre_experiences_precedentes"] = 20
        employee = EmployeeInput(**valid_employee_data)
        assert employee.revenu_mensuel == 50000.0
        assert employee.annee_experience_totale == 40.0

    # ======================
    # VALIDATE_EMPLOYEE WRAPPER TESTS
    # ======================

    def test_validate_employee_raises_value_error_on_invalid_data(self, valid_employee_data):
        """Test that validate_employee raises ValueError on invalid data."""
        valid_employee_data["age"] = 15
        with pytest.raises(ValueError) as exc_info:
            validate_employee(valid_employee_data)
        assert "age" in str(exc_info.value)

    def test_validate_employee_success(self, valid_employee_data):
        """Test that validate_employee returns EmployeeInput on valid data."""
        result = validate_employee(valid_employee_data)
        assert isinstance(result, EmployeeInput)
        assert result.employee_id == 1

    # ======================
    # TYPE CONVERSION TESTS
    # ======================

    def test_float_to_int_conversion(self, valid_employee_data):
        """Test that float values for integer fields are converted."""
        valid_employee_data["age"] = 30.0
        valid_employee_data["niveau_education"] = 3.0
        employee = EmployeeInput(**valid_employee_data)
        assert isinstance(employee.age, int)
        assert isinstance(employee.niveau_education, int)

    def test_string_number_fails(self, valid_employee_data):
        """Test that string numbers are rejected."""
        valid_employee_data["age"] = "30"
        with pytest.raises(ValidationError) as exc_info:
            EmployeeInput(**valid_employee_data)
        assert "age" in str(exc_info.value)
