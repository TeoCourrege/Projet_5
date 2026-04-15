-- ==============================
-- Users (authentication)
-- ==============================
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================
-- Model inputs (logging)
-- ==============================
CREATE TABLE IF NOT EXISTS model_inputs (
    input_id SERIAL PRIMARY KEY,
    employee_id INT,
    age INT,
    genre VARCHAR(10),
    revenu_mensuel FLOAT,
    statut_marital VARCHAR(50),
    departement VARCHAR(50),
    poste VARCHAR(100),
    nombre_experiences_precedentes INT,
    nombre_heures_travailless FLOAT,
    annee_experience_totale INT,
    annees_dans_l_entreprise INT,
    annees_dans_le_poste_actuel INT,
    nombre_participation_pee INT,
    nb_formations_suivies INT,
    nombre_employee_sous_responsabilite INT,
    distance_domicile_travail INT,
    niveau_education INT,
    domaine_etude VARCHAR(100),
    ayant_enfants BOOLEAN,
    frequence_deplacement VARCHAR(50),
    annees_depuis_la_derniere_promotion INT,
    annes_sous_responsable_actuel INT,
    satisfaction_employee_environnement INT,
    note_evaluation_precedente INT,
    niveau_hierarchique_poste INT,
    satisfaction_employee_nature_travail INT,
    satisfaction_employee_equipe INT,
    satisfaction_employee_equilibre_pro_perso INT,
    note_evaluation_actuelle INT,
    heure_supplementaires VARCHAR(50),
    augementation_salaire_precedente FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================
-- Model outputs (logging)
-- ==============================
CREATE TABLE IF NOT EXISTS model_outputs (
    output_id SERIAL PRIMARY KEY,
    input_id INT NOT NULL REFERENCES model_inputs(input_id) ON DELETE CASCADE,
    employee_id INT,
    prediction INT,
    probability FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================
-- Indexes
-- ==============================
CREATE INDEX IF NOT EXISTS idx_inputs_employee ON model_inputs(employee_id);
CREATE INDEX IF NOT EXISTS idx_outputs_input ON model_outputs(input_id);
CREATE INDEX IF NOT EXISTS idx_outputs_employee ON model_outputs(employee_id);
