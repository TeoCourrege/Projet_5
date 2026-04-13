import os
import logging

import bcrypt
import pandas as pd
import joblib
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)

# ==============================
# DATABASE CONFIG
# ==============================
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# fallback to SQLite if Postgres is not configured
if not POSTGRES_HOST:
    DATABASE_URL = "sqlite:////data/app.db"
    logger.warning("POSTGRES not set → using SQLite fallback")
else:
    DATABASE_URL = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ==============================
# ORM MODELS
# ==============================
class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelInput(Base):
    __tablename__ = "model_inputs"
    input_id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer)
    age = Column(Integer)
    genre = Column(String(10))
    revenu_mensuel = Column(Float)
    statut_marital = Column(String(50))
    departement = Column(String(50))
    poste = Column(String(100))
    nombre_experiences_precedentes = Column(Integer)
    nombre_heures_travailless = Column(Float)
    annee_experience_totale = Column(Integer)
    annees_dans_l_entreprise = Column(Integer)
    annees_dans_le_poste_actuel = Column(Integer)
    nombre_participation_pee = Column(Integer)
    nb_formations_suivies = Column(Integer)
    nombre_employee_sous_responsabilite = Column(Integer)
    distance_domicile_travail = Column(Integer)
    niveau_education = Column(Integer)
    domaine_etude = Column(String(100))
    frequence_deplacement = Column(String(50))
    annees_depuis_la_derniere_promotion = Column(Integer)
    annes_sous_responsable_actuel = Column(Integer)
    satisfaction_employee_environnement = Column(Integer)
    note_evaluation_precedente = Column(Integer)
    niveau_hierarchique_poste = Column(Integer)
    satisfaction_employee_nature_travail = Column(Integer)
    satisfaction_employee_equipe = Column(Integer)
    satisfaction_employee_equilibre_pro_perso = Column(Integer)
    note_evaluation_actuelle = Column(Integer)
    heure_supplementaires = Column(String)
    augementation_salaire_precedente = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    outputs = relationship("ModelOutput", back_populates="input_record")


class ModelOutput(Base):
    __tablename__ = "model_outputs"
    output_id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(Integer, ForeignKey("model_inputs.input_id", ondelete="CASCADE"), nullable=False)
    employee_id = Column(Integer)
    prediction = Column(Integer)
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    input_record = relationship("ModelInput", back_populates="outputs")


Index("idx_inputs_employee", ModelInput.employee_id)
Index("idx_outputs_input", ModelOutput.input_id)
Index("idx_outputs_employee", ModelOutput.employee_id)


# ==============================
# TABLE CREATION
# ==============================
def init_db():
    try:
        Base.metadata.create_all(engine)
        logger.info("Tables created")
    except Exception as e:
        logger.error(f"DB init failed: {e}")


# ==============================
# AUTH HELPERS
# ==============================
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_user(username: str, password: str):
    session = SessionLocal()
    try:
        user = User(username=username, password_hash=hash_password(password))
        session.add(user)
        session.commit()
        logger.info("User '%s' created.", username)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def authenticate(username: str, password: str) -> bool:
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=username, is_active=True).first()
        if user is None:
            return False
        return verify_password(password, user.password_hash)
    finally:
        session.close()


# ==============================
# LAZY MODEL LOADING
# ==============================
_pipeline = None
MODEL_PATH = "src/model/model.pkl"


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


# Feature columns used by the ML pipeline (excludes PK, FK, and metadata)
FEATURE_COLUMNS = [
    "age", "genre", "revenu_mensuel", "statut_marital", "departement", "poste",
    "nombre_experiences_precedentes", "nombre_heures_travailless",
    "annee_experience_totale", "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "nombre_participation_pee", "nb_formations_suivies",
    "nombre_employee_sous_responsabilite", "distance_domicile_travail",
    "niveau_education", "domaine_etude",
    "frequence_deplacement", "annees_depuis_la_derniere_promotion",
    "annes_sous_responsable_actuel", "satisfaction_employee_environnement",
    "note_evaluation_precedente", "niveau_hierarchique_poste",
    "satisfaction_employee_nature_travail", "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso", "note_evaluation_actuelle",
    "heure_supplementaires", "augementation_salaire_precedente",
]


# ==============================
# PREDICTION (single row)
# ==============================
def predict(
    employee_id, age, genre, revenu_mensuel, statut_marital, departement, poste,
    nombre_experiences_precedentes, nombre_heures_travailless, annee_experience_totale,
    annees_dans_l_entreprise, annees_dans_le_poste_actuel,
    nombre_participation_pee, nb_formations_suivies, nombre_employee_sous_responsabilite,
    distance_domicile_travail, niveau_education, domaine_etude,
    frequence_deplacement, annees_depuis_la_derniere_promotion,
    annes_sous_responsable_actuel, satisfaction_employee_environnement,
    note_evaluation_precedente, niveau_hierarchique_poste,
    satisfaction_employee_nature_travail, satisfaction_employee_equipe,
    satisfaction_employee_equilibre_pro_perso, note_evaluation_actuelle,
    heure_supplementaires, augementation_salaire_precedente
):
    session = SessionLocal()
    try:
        input_record = ModelInput(
            employee_id=employee_id, age=age, genre=genre,
            revenu_mensuel=revenu_mensuel, statut_marital=statut_marital,
            departement=departement, poste=poste,
            nombre_experiences_precedentes=nombre_experiences_precedentes,
            nombre_heures_travailless=nombre_heures_travailless,
            annee_experience_totale=annee_experience_totale,
            annees_dans_l_entreprise=annees_dans_l_entreprise,
            annees_dans_le_poste_actuel=annees_dans_le_poste_actuel,
            nombre_participation_pee=nombre_participation_pee,
            nb_formations_suivies=nb_formations_suivies,
            nombre_employee_sous_responsabilite=nombre_employee_sous_responsabilite,
            distance_domicile_travail=distance_domicile_travail,
            niveau_education=niveau_education, domaine_etude=domaine_etude,
            frequence_deplacement=frequence_deplacement,
            annees_depuis_la_derniere_promotion=annees_depuis_la_derniere_promotion,
            annes_sous_responsable_actuel=annes_sous_responsable_actuel,
            satisfaction_employee_environnement=satisfaction_employee_environnement,
            note_evaluation_precedente=note_evaluation_precedente,
            niveau_hierarchique_poste=niveau_hierarchique_poste,
            satisfaction_employee_nature_travail=satisfaction_employee_nature_travail,
            satisfaction_employee_equipe=satisfaction_employee_equipe,
            satisfaction_employee_equilibre_pro_perso=satisfaction_employee_equilibre_pro_perso,
            note_evaluation_actuelle=note_evaluation_actuelle,
            heure_supplementaires=heure_supplementaires,
            augementation_salaire_precedente=augementation_salaire_precedente,
        )
        session.add(input_record)
        session.commit()

        data = pd.DataFrame([{
            col: getattr(input_record, col) for col in FEATURE_COLUMNS
        }])

        pipeline = _get_pipeline()
        prediction = pipeline.predict(data)[0]
        probability = float(pipeline.predict_proba(data)[0][1])

        output_record = ModelOutput(
            input_id=input_record.input_id,
            employee_id=input_record.employee_id,
            prediction=int(prediction),
            probability=probability,
        )
        session.add(output_record)
        session.commit()

        return int(prediction)

    except Exception as e:
        session.rollback()
        return f"Erreur: {str(e)}"
    finally:
        session.close()


# ==============================
# BATCH PREDICTION
# ==============================
def batch_predict(file):
    session = SessionLocal()
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Format non supporté (CSV ou JSON uniquement)"

        if "id" not in df.columns:
            return "Le fichier doit contenir la colonne 'id'"

        results = []
        for _, row in df.iterrows():
            pred = predict(
                employee_id=row.get("id"),
                age=row.get("age"),
                genre=row.get("genre"),
                revenu_mensuel=row.get("revenu_mensuel"),
                statut_marital=row.get("statut_marital"),
                departement=row.get("departement"),
                poste=row.get("poste"),
                nombre_experiences_precedentes=row.get("nombre_experiences_precedentes"),
                nombre_heures_travailless=row.get("nombre_heures_travailless"),
                annee_experience_totale=row.get("annee_experience_totale"),
                annees_dans_l_entreprise=row.get("annees_dans_l_entreprise"),
                annees_dans_le_poste_actuel=row.get("annees_dans_le_poste_actuel"),
                nombre_participation_pee=row.get("nombre_participation_pee"),
                nb_formations_suivies=row.get("nb_formations_suivies"),
                nombre_employee_sous_responsabilite=row.get("nombre_employee_sous_responsabilite"),
                distance_domicile_travail=row.get("distance_domicile_travail"),
                niveau_education=row.get("niveau_education"),
                domaine_etude=row.get("domaine_etude"),
                frequence_deplacement=row.get("frequence_deplacement"),
                annees_depuis_la_derniere_promotion=row.get("annees_depuis_la_derniere_promotion"),
                annes_sous_responsable_actuel=row.get("annes_sous_responsable_actuel"),
                satisfaction_employee_environnement=row.get("satisfaction_employee_environnement"),
                note_evaluation_precedente=row.get("note_evaluation_precedente"),
                niveau_hierarchique_poste=row.get("niveau_hierarchique_poste"),
                satisfaction_employee_nature_travail=row.get("satisfaction_employee_nature_travail"),
                satisfaction_employee_equipe=row.get("satisfaction_employee_equipe"),
                satisfaction_employee_equilibre_pro_perso=row.get("satisfaction_employee_equilibre_pro_perso"),
                note_evaluation_actuelle=row.get("note_evaluation_actuelle"),
                heure_supplementaires=row.get("heure_supplementaires"),
                augementation_salaire_precedente=row.get("augementation_salaire_precedente"),
            )
            results.append({"id": row.get("id"), "prediction": pred})

        return pd.DataFrame(results)

    except Exception as e:
        session.rollback()
        return f"Erreur: {str(e)}"
    finally:
        session.close()


# ==============================
# SEED DEFAULT ADMIN
# ==============================
def seed_admin():
    session = SessionLocal()
    try:
        existing = session.query(User).filter_by(username=os.getenv("ADMIN_USERNAME", "admin")).first()
        if existing:
            return
        create_user(
            username=os.getenv("ADMIN_USERNAME", "admin"),
            password=os.getenv("ADMIN_PASSWORD", "admin"),
        )
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    seed_admin()
    print("Database initialized and admin user created.")
