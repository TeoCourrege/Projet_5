# Employee Churn Prediction

Application de Machine Learning pour prédire le départ d'un employé d'une entreprise.  
Interface Gradio + pipeline scikit-learn + base PostgreSQL pour le logging des prédictions.

---

## Table des matières

1. [Architecture du projet](#architecture-du-projet)
2. [Modèle de données](#modèle-de-données)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Déploiement](#déploiement)
6. [Tests](#tests)
7. [CI/CD](#cicd)
8. [Authentification & Sécurité](#authentification--sécurité)
9. [Processus de traitement des données](#processus-de-traitement-des-données)

---

## Architecture du projet

```
Projet_5/
├── .env.example              # Variables d'environnement (template)
├── .github/workflows/ci.yaml # Pipeline CI/CD GitHub Actions
├── app.py                    # Point d'entrée Gradio
├── pyproject.toml            # Configuration pytest / coverage
├── requirements.txt          # Dépendances Python
├── data/
│   ├── raw/                  # Données brutes (CSV sources)
│   ├── processed/            # Données fusionnées (df_final.csv)
│   └── test/                 # Données de test
├── sql/
│   └── create_tables.sql     # Script SQL de création des tables
├── src/
│   ├── db/
│   │   └── database.py       # ORM, auth, logging prédictions
│   └── model/
│       ├── train.py          # Fusion données + entraînement pipeline
│       ├── predict.py        # Prédiction batch en standalone
│       └── preprocessing.py  # Feature engineering (optionnel)
├── tests/
│   ├── conftest.py           # Fixtures partagées
│   ├── tests.py              # Tests du pipeline ML
│   └── test_api.py           # Tests de l'application Gradio
└── uml.txt                   # Schéma des tables
```

---

## Modèle de données

### Tables PostgreSQL

**`users`** — Comptes d'accès à l'application

| Colonne       | Type         | Contrainte     |
|---------------|--------------|----------------|
| user_id       | SERIAL       | PRIMARY KEY    |
| username      | VARCHAR(100) | UNIQUE NOT NULL|
| password_hash | VARCHAR(255) | NOT NULL       |
| is_active     | BOOLEAN      | DEFAULT TRUE   |
| created_at    | TIMESTAMP    | DEFAULT NOW()  |

**`model_inputs`** — Logging des données envoyées au modèle

| Colonne                       | Type          | Description                    |
|-------------------------------|---------------|--------------------------------|
| input_id                      | SERIAL        | PRIMARY KEY                    |
| employee_id                   | INT           | Identifiant de l'employé       |
| age, genre, revenu_mensuel... | divers        | Features du modèle ML          |
| created_at                    | TIMESTAMP     | Date de la requête             |

**`model_outputs`** — Résultats des prédictions

| Colonne     | Type   | Contrainte                                    |
|-------------|--------|-----------------------------------------------|
| output_id   | SERIAL | PRIMARY KEY                                   |
| input_id    | INT    | FOREIGN KEY → model_inputs (ON DELETE CASCADE) |
| employee_id | INT    | Identifiant de l'employé                      |
| prediction  | INT    | 0 = reste, 1 = part                           |
| probability | FLOAT  | Probabilité de départ                         |
| created_at  | TIMESTAMP | Date de la prédiction                      |

### Relations

- `model_outputs.input_id` → `model_inputs.input_id` (1:N, cascade delete)
- Index sur `employee_id` (inputs et outputs) et `input_id` (outputs) pour les requêtes analytiques.

Le schéma complet est dans `sql/create_tables.sql` et répliqué en ORM dans `src/db/database.py`.

---

## Installation

### Prérequis

- Python 3.11+
- PostgreSQL 14+
- Git

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/<username>/Projet_5.git
cd Projet_5

# 2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos identifiants PostgreSQL

# 5. Créer la base de données PostgreSQL
psql -U postgres -c "CREATE DATABASE projet5;"

# 6. Initialiser les tables et le compte admin
python -m src.db.database
# Ou via le script SQL :
# psql -U postgres -d projet5 -f sql/create_tables.sql

# 7. Entraîner le modèle (si model.pkl n'existe pas)
python -m src.model.train

# 8. Lancer l'application
python app.py
```

L'application sera disponible sur `http://localhost:7860`.

---

## Utilisation

### Interface Gradio

L'application propose deux modes :

1. **Prédiction par fichier** : uploader un CSV/JSON contenant une colonne `id` et toutes les features. Retourne un tableau `id | prediction`.
2. **Prédiction manuelle** : remplir le formulaire pour un seul employé.

Chaque prédiction (unitaire ou batch) est **automatiquement loggée** dans PostgreSQL :
- Les inputs sont enregistrés dans `model_inputs`
- Les outputs (prédiction + probabilité) dans `model_outputs`

Cela permet un suivi analytique et la détection de dérive du modèle.

### Prédiction batch en ligne de commande

```bash
python -m src.model.predict  # utilise data/test/test.csv par défaut
```

---

## Déploiement

### Hugging Face Spaces

Le déploiement est automatisé via GitHub Actions (`.github/workflows/ci.yaml`).

**Secrets GitHub requis :**

| Secret          | Description                     |
|-----------------|---------------------------------|
| `HF_TOKEN`      | Token d'API Hugging Face        |
| `HF_USERNAME`   | Nom d'utilisateur HF            |
| `HF_SPACE_NAME` | Nom du Space HF                 |

Le pipeline CI exécute les tests, puis déploie sur HF Spaces si la branche `main` passe.

### Configuration des environnements

| Variable   | Développement        | Production             | Test (CI)            |
|------------|----------------------|------------------------|----------------------|
| `APP_ENV`  | `development`        | `production`           | `test`               |
| `POSTGRES_HOST` | `localhost`    | URL du service managé  | `localhost` (service) |
| Secrets    | `.env` local         | Secrets GitHub / HF    | Variables CI         |

---

## Tests

### Exécution

```bash
# Tous les tests avec rapport de couverture
pytest

# Tests spécifiques
pytest tests/tests.py       # Pipeline ML
pytest tests/test_api.py    # Application / Auth

# Rapport HTML
pytest --cov-report=html
open htmlcov/index.html
```

### Couverture

La configuration dans `pyproject.toml` génère automatiquement un rapport de couverture.  
Seuil minimum : **60%** (configurable).

### Stratégie de tests

| Fichier         | Portée                                         |
|-----------------|------------------------------------------------|
| `tests.py`      | Entraînement, persistance, shape preprocessing |
| `test_api.py`   | Chargement app, auth, validation batch input   |
| `conftest.py`   | Fixtures partagées (sample_row, sample_data)   |

Les tests vérifient :
- **Intégrité des données** : types, colonnes attendues, shape des outputs
- **Fonctionnalité** : entraînement, prédiction, sérialisation modèle
- **Sécurité** : hachage de mot de passe, rejet de credentials invalides
- **Robustesse** : formats de fichier non supportés, colonnes manquantes

---

## CI/CD

Le fichier `.github/workflows/ci.yaml` automatise :

1. **Job `test`** : 
   - Démarre un service PostgreSQL 16
   - Installe les dépendances
   - Exécute les tests avec couverture
   - Upload le rapport de couverture en artefact

2. **Job `deploy`** (uniquement sur `main`) :
   - Déploie sur Hugging Face Spaces via `git push`

---

## Authentification & Sécurité

### Méthode d'authentification

L'application utilise le système d'authentification intégré de Gradio (`demo.launch(auth=...)`).  
Un compte admin par défaut est créé au premier lancement via les variables d'environnement `ADMIN_USERNAME` / `ADMIN_PASSWORD`.

Dans notre cas, étant donné que ce projet est une démo visant à montrer les capacités du modèles, nous utilisons pour admin les identifiant par défauts suivants:
Username: admin
Password: admin

Les utilisateurs sont stockés dans la table `users` de PostgreSQL.

### Bonnes pratiques de sécurité

| Pratique                     | Implémentation                                    |
|------------------------------|---------------------------------------------------|
| **Hachage des mots de passe** | bcrypt (salt automatique)                        |
| **Secrets**                   | Fichier `.env` (jamais commité, dans `.gitignore`) |
| **Template secrets**          | `.env.example` sans valeurs sensibles             |
| **CI/CD secrets**             | GitHub Secrets (`HF_TOKEN`, etc.)                 |
| **Accès DB**                  | Credentials via variables d'environnement         |
| **Validation des inputs**     | Gradio valide les types via les composants UI     |

### Gestion des accès

```bash
# Créer un nouvel utilisateur (via Python)
python -c "
from src.db.database import init_db, create_user
init_db()
create_user('nom_utilisateur', 'mot_de_passe_fort')
"
```

---

## Processus de traitement des données

### Pipeline de données

1. **Sources** : 3 fichiers CSV bruts (`extrait_eval.csv`, `extrait_sirh.csv`, `extrait_sondage.csv`)
2. **Fusion** : `merge_raw_data()` dans `src/model/train.py` — jointure sur `id_employee` / `code_sondage` / `eval_number`
3. **Nettoyage** : conversion du `%` en float pour `augementation_salaire_precedente`
4. **Stockage** : fichier fusionné `data/processed/df_final.csv`
5. **Entraînement** : `train_pipeline()` — preprocessing (OrdinalEncoder + OneHotEncoder) + RandomForest
6. **Sérialisation** : modèle sauvegardé en `src/model/model.pkl` via joblib

### Pipeline de prédiction (production)

1. L'utilisateur soumet des données via Gradio (formulaire ou fichier)
2. Les inputs sont **loggés** dans `model_inputs`
3. Le pipeline sklearn applique le preprocessing et prédit
4. Le résultat + probabilité sont **loggés** dans `model_outputs`
5. Le résultat est affiché à l'utilisateur

### Besoins analytiques

La base PostgreSQL permet de :
- **Suivre le volume** de prédictions par période
- **Détecter la dérive** en comparant les distributions d'inputs au fil du temps
- **Auditer** les prédictions par employé (`employee_id`)
- **Mesurer la performance** en confrontant les prédictions aux départs réels (via jointure sur `employee_id`)

Les requêtes analytiques sont facilitées par les index sur `employee_id` et `input_id`.
