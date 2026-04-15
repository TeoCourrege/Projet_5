from pydantic import BaseModel, Field, field_validator
from typing import Literal


class EmployeeInput(BaseModel):

    # ======================
    # IDENTITÉ
    # ======================
    employee_id: int = Field(..., ge=0)
    age: int = Field(..., ge=16, le=70)
    genre: Literal["M", "F"]

    statut_marital: Literal["Célibataire", "Marié(e)", "Divorcé(e)"]

    # ======================
    # POSTE
    # ======================
    departement: Literal["Commercial", "Consulting", "Ressources Humaines"]

    poste: Literal[
        "Cadre Commercial",
        "Assistant de Direction",
        "Consultant",
        "Tech Lead",
        "Manager",
        "Senior Manager",
        "Représentant Commercial",
        "Directeur Technique",
        "Ressources Humaines"
    ]

    domaine_etude: Literal[
        "Infra & Cloud",
        "Autre",
        "Transformation Digitale",
        "Marketing",
        "Entrepreunariat",
        "Ressources Humaines"
    ]

    niveau_education: int = Field(..., ge=1, le=5)
    niveau_hierarchique_poste: int = Field(..., ge=1, le=10)

    # ======================
    # EXPERIENCE
    # ======================
    annee_experience_totale: float = Field(..., ge=0)
    annees_dans_l_entreprise: float = Field(..., ge=0)
    annees_dans_le_poste_actuel: float = Field(..., ge=0)
    annees_depuis_la_derniere_promotion: float = Field(..., ge=0)
    annes_sous_responsable_actuel: float = Field(..., ge=0)
    nombre_experiences_precedentes: int = Field(..., ge=0)

    # ======================
    # REMUNERATION
    # ======================
    revenu_mensuel: float = Field(..., ge=0)
    augementation_salaire_precedente: float = Field(..., ge=0)
    nombre_participation_pee: int = Field(..., ge=0)

    # ======================
    # CONDITIONS TRAVAIL
    # ======================
    nombre_heures_travailless: float = Field(..., ge=0, le=100)

    heure_supplementaires: Literal["Oui", "Non"]

    distance_domicile_travail: float = Field(..., ge=0, le=300)

    frequence_deplacement: Literal["Aucun", "Occasionnel", "Frequent"]

    # ======================
    # FORMATION / RESPONSABILITE
    # ======================
    nb_formations_suivies: int = Field(..., ge=0)

    nombre_employee_sous_responsabilite: int = Field(..., ge=0)

    # ======================
    # SATISFACTION / EVALUATION
    # ======================
    satisfaction_employee_environnement: int = Field(..., ge=1, le=4)
    satisfaction_employee_nature_travail: int = Field(..., ge=1, le=4)
    satisfaction_employee_equipe: int = Field(..., ge=1, le=4)
    satisfaction_employee_equilibre_pro_perso: int = Field(..., ge=1, le=4)

    note_evaluation_precedente: int = Field(..., ge=1, le=4)
    note_evaluation_actuelle: int = Field(..., ge=1, le=4)

    # ======================
    # VALIDATIONS METIER
    # ======================

    @field_validator("age")
    @classmethod
    def check_age(cls, v):
        if v < 18:
            raise ValueError("Employee must be adult")
        return v

    @field_validator("annee_experience_totale")
    @classmethod
    def experience_logique(cls, v, info):
        if "age" in info.data and v > info.data["age"]:
            raise ValueError("Experience cannot exceed age")
        return v

    @field_validator("distance_domicile_travail")
    @classmethod
    def distance_reasonable(cls, v):
        if v > 1000:
            raise ValueError("Distance unrealistic")
        return v

    @field_validator("nombre_heures_travailless")
    @classmethod
    def hours_check(cls, v):
        if v < 5:
            raise ValueError("Too few working hours")
        return v