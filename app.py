import gradio as gr
from src.db.database import (
    predict as db_predict,
    batch_predict,
    authenticate,
    init_db,
    seed_admin,
)

init_db()
seed_admin()

# ===============================
# UI
# ===============================
with gr.Blocks(title="Prédiction départ employé") as demo:

    gr.Markdown("# Prédiction départ employé")
    gr.Markdown("Prédisez si un employé va quitter l'entreprise (0 = reste, 1 = part).")

    with gr.Tab("Prédiction par fichier"):
        file_input = gr.File(label="Upload CSV ou JSON (doit contenir la colonne 'id')")
        batch_btn = gr.Button("Prédire fichier", variant="primary")
        batch_output = gr.Dataframe(
            label="Résultats",
            headers=["id", "prediction"],
            datatype=["number", "number"],
            col_count=(2, "fixed"))

        batch_btn.click(fn=batch_predict, inputs=file_input, outputs=batch_output)

    with gr.Tab("Prédiction manuelle"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Identité")
                inp_id = gr.Number(label="ID employé", precision=0, value=0)
                inp_age = gr.Number(label="Age", precision=0, value=30)
                inp_genre = gr.Radio(["M", "F"], label="Genre", value="M")
                inp_statut = gr.Radio(["Célibataire", "Marié(e)", "Divorcé(e)"], label="Statut marital", value="Célibataire")

            with gr.Column():
                gr.Markdown("### Poste")
                inp_dept = gr.Dropdown(
                    choices=["Commercial", "Consulting", "Ressources Humaines"],
                    label="Département",
                    value="Commercial"
                )
                inp_poste = gr.Dropdown(
                    choices=[
                        'Cadre Commercial',
                        'Assistant de Direction',
                        'Consultant',
                        'Tech Lead',
                        'Manager',
                        'Senior Manager',
                        'Représentant Commercial',
                        'Directeur Technique',
                        'Ressources Humaines'
                    ],
                    label="Poste",
                    value='Consultant'
                )
                inp_domaine = gr.Dropdown(
                    choices=[
                        'Infra & Cloud',
                        'Autre',
                        'Transformation Digitale',
                        'Marketing',
                        'Entrepreunariat',
                        'Ressources Humaines'
                    ],
                    label="Domaine d'étude",
                    value='Autre'
                )
                inp_niveau_edu = gr.Number(label="Niveau éducation (1-5)", precision=0, value=3)
                inp_niveau_hier = gr.Number(label="Niveau hiérarchique", precision=0, value=2)

            with gr.Column():
                gr.Markdown("### Expérience")
                inp_exp_total = gr.Number(label="Expérience totale (années)", precision=0, value=5)
                inp_annees_ent = gr.Number(label="Années dans l'entreprise", precision=0, value=3)
                inp_annees_poste = gr.Number(label="Années dans le poste", precision=0, value=2)
                inp_annees_promo = gr.Number(label="Années depuis dernière promo", precision=0, value=1)
                inp_annees_mgr = gr.Number(label="Années sous manager actuel", precision=0, value=2)
                inp_nb_exp = gr.Number(label="Nb expériences précédentes", precision=0, value=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Rémunération")
                inp_revenu = gr.Number(label="Revenu mensuel", value=5000)
                inp_augment = gr.Number(label="Augmentation salaire (%)", value=3)
                inp_pee = gr.Number(label="Participation PEE", precision=0, value=0)

            with gr.Column():
                gr.Markdown("### Conditions de travail")
                inp_heures = gr.Number(label="Heures travaillées", value=40)
                inp_heures_supp = gr.Radio(["Oui", "Non"], label="Heures supplémentaires", value="Non")
                inp_distance = gr.Number(label="Distance domicile-travail", value=10)
                inp_deplacement = gr.Radio(["Aucun", "Occasionnel", "Frequent"], label="Fréquence déplacement", value="Occasionnel")

            with gr.Column():
                gr.Markdown("### Évaluations & Satisfaction")
                inp_formations = gr.Number(label="Nb formations suivies", precision=0, value=2)
                inp_employes = gr.Number(label="Employés sous responsabilité", precision=0, value=0)
                inp_sat_env = gr.Slider(1, 4, step=1, label="Satisfaction environnement", value=3)
                inp_sat_job = gr.Slider(1, 4, step=1, label="Satisfaction nature travail", value=3)
                inp_sat_equipe = gr.Slider(1, 4, step=1, label="Satisfaction équipe", value=3)
                inp_sat_equil = gr.Slider(1, 4, step=1, label="Équilibre vie pro/perso", value=3)
                inp_note_prec = gr.Slider(1, 4, step=1, label="Note évaluation précédente", value=3)
                inp_note_act = gr.Slider(1, 4, step=1, label="Note évaluation actuelle", value=3)

        inputs = [
            inp_id, inp_age, inp_genre, inp_revenu, inp_statut,
            inp_dept, inp_poste, inp_nb_exp, inp_heures, inp_exp_total,
            inp_annees_ent, inp_annees_poste, inp_pee,
            inp_formations, inp_employes, inp_distance, inp_niveau_edu,
            inp_domaine, inp_deplacement, inp_annees_promo,
            inp_annees_mgr, inp_sat_env, inp_note_prec, inp_niveau_hier,
            inp_sat_job, inp_sat_equipe, inp_sat_equil, inp_note_act,
            inp_heures_supp, inp_augment,
        ]

        with gr.Row():
            predict_btn = gr.Button("Prédire", variant="primary")
            prediction_output = gr.Number(label="Prédiction (0=reste, 1=part)")

        predict_btn.click(fn=db_predict, inputs=inputs, outputs=prediction_output)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    demo.launch(auth=authenticate, server_name="0.0.0.0", server_port=7860)