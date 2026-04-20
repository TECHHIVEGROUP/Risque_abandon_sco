# ============================================
# TABLEAU DE BORD - ANALYSE DU RISQUE D'ABANDON SCOLAIRE
# Dashboard complet avec statistiques, visualisations et PREDICTION
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Abandon Scolaire",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CHARGEMENT DES DONNEES
# ============================================

@st.cache_data
def load_data():
    """Charge le dataset"""
    df = pd.read_csv("./data/student_dropout_dataset.csv")
    
    # Feature engineering
    df["presence_ratio"] = 1 - df["absenteeism_rate"]
    df["effort_score"] = df["study_time_hours"] * df["average_grade"] / 20
    df["global_score"] = (df["average_grade"] * 0.5 + 
                          df["presence_ratio"] * 20 * 0.3 + 
                          df["study_time_hours"] * 2 * 0.2)
    
    # Categorisation de l'age
    df["age_group"] = pd.cut(df["age"], bins=[14, 18, 21, 25], labels=["15-18", "19-21", "22-24"])
    
    # Categorisation de la moyenne
    df["grade_category"] = pd.cut(df["average_grade"], bins=[0, 10, 12, 14, 20], 
                                   labels=["<10 (Faible)", "10-12 (Moyen)", "12-14 (Bon)", ">14 (Excellent)"])
    
    # Categorisation de l'absenteisme
    df["absenteism_category"] = pd.cut(df["absenteeism_rate"], bins=[0, 0.1, 0.2, 0.3, 0.5],
                                        labels=["<10% (Tres faible)", "10-20% (Faible)", "20-30% (Modere)", ">30% (Eleve)"])
    
    return df

@st.cache_resource
def load_model():
    """Charge le modele entraine avec pickle (compatible toutes versions)"""
    import os
    
    # Chemins possibles
    possible_paths = [
        "./models/dropout_model.pkl",
        "models/dropout_model.pkl",
        "dropout_model.pkl",
        "./dropout_model.pkl"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                st.success(f"Modele charge avec succes depuis {model_path}")
                return model_data
            except Exception as e:
                st.warning(f"Erreur avec {model_path}: {e}")
                continue
    
    st.error("Modele non trouve. Veuillez executer generate_model.py pour creer le modele.")
    return None

# ============================================
# FONCTIONS DE VISUALISATION
# ============================================

def create_kpi_cards(df):
    """Cree les cartes KPI"""
    total_students = len(df)
    dropout_count = df["dropout_risk"].sum()
    dropout_rate = (dropout_count / total_students) * 100
    avg_grade = df["average_grade"].mean()
    avg_attendance = (1 - df["absenteeism_rate"].mean()) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total etudiants",
            value=f"{total_students:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Taux d'abandon",
            value=f"{dropout_rate:.1f}%",
            delta=f"{dropout_count} etudiants",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Moyenne generale",
            value=f"{avg_grade:.1f}/20",
            delta="+/- 2.5"
        )
    
    with col4:
        st.metric(
            label="Taux de presence",
            value=f"{avg_attendance:.1f}%",
            delta="Objectif > 90%"
        )

def create_risk_gauge(dropout_rate):
    """Cree une jauge de risque global"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dropout_rate,
        title={"text": "Taux d'abandon global (%)", "font": {"size": 24}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 50], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "darkred", "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 15], "color": "lightgreen"},
                {"range": [15, 30], "color": "yellow"},
                {"range": [30, 50], "color": "salmon"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": dropout_rate
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_dropout_by_variable(df, variable, title):
    """Graphique du taux d'abandon par variable"""
    stats = df.groupby(variable)["dropout_risk"].agg(['count', 'mean']).reset_index()
    stats.columns = [variable, 'count', 'dropout_rate']
    stats['dropout_rate'] = stats['dropout_rate'] * 100
    
    fig = px.bar(
        stats, 
        x=variable, 
        y='dropout_rate',
        text=stats['dropout_rate'].apply(lambda x: f'{x:.1f}%'),
        color='dropout_rate',
        color_continuous_scale='RdYlGn_r',
        title=title
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_correlation_heatmap(df):
    """Matrice de correlation"""
    numeric_cols = ["age", "average_grade", "absenteeism_rate", "study_time_hours", 
                    "presence_ratio", "effort_score", "global_score"]
    corr_matrix = df[numeric_cols + ["dropout_risk"]].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matrice de correlation"
    )
    fig.update_layout(height=500)
    return fig

def create_distribution_comparison(df):
    """Comparaison des distributions risque vs non risque"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Moyenne generale", "Taux d'absenteisme", 
                        "Temps d'etude", "Score global")
    )
    
    # Moyenne generale
    for risk in [0, 1]:
        subset = df[df["dropout_risk"] == risk]
        label = "Non abandon" if risk == 0 else "Abandon"
        color = "blue" if risk == 0 else "red"
        fig.add_trace(
            go.Histogram(x=subset["average_grade"], name=label, 
                        marker_color=color, opacity=0.6),
            row=1, col=1
        )
    
    # Taux d'absenteisme
    for risk in [0, 1]:
        subset = df[df["dropout_risk"] == risk]
        label = "Non abandon" if risk == 0 else "Abandon"
        color = "blue" if risk == 0 else "red"
        fig.add_trace(
            go.Histogram(x=subset["absenteeism_rate"], name=label, 
                        marker_color=color, opacity=0.6),
            row=1, col=2
        )
    
    # Temps d'etude
    for risk in [0, 1]:
        subset = df[df["dropout_risk"] == risk]
        label = "Non abandon" if risk == 0 else "Abandon"
        color = "blue" if risk == 0 else "red"
        fig.add_trace(
            go.Histogram(x=subset["study_time_hours"], name=label, 
                        marker_color=color, opacity=0.6),
            row=2, col=1
        )
    
    # Score global
    for risk in [0, 1]:
        subset = df[df["dropout_risk"] == risk]
        label = "Non abandon" if risk == 0 else "Abandon"
        color = "blue" if risk == 0 else "red"
        fig.add_trace(
            go.Histogram(x=subset["global_score"], name=label, 
                        marker_color=color, opacity=0.6),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True, title_text="Distribution des variables selon le risque")
    fig.update_xaxes(title_text="Moyenne (/20)", row=1, col=1)
    fig.update_xaxes(title_text="Taux d'absenteisme", row=1, col=2)
    fig.update_xaxes(title_text="Temps d'etude (heures)", row=2, col=1)
    fig.update_xaxes(title_text="Score global", row=2, col=2)
    
    return fig

def create_feature_importance(df):
    """Importance des variables (basee sur Random Forest)"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Preparation des donnees
    features = ["age", "average_grade", "absenteeism_rate", "study_time_hours", 
                "presence_ratio", "effort_score", "global_score"]
    X = df[features]
    y = df["dropout_risk"]
    
    # Entrainement rapide
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Creation du graphique
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Importance des variables dans la prediction",
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    return fig

def create_sunburst_chart(df):
    """Graphique sunburst des abandons par categories"""
    # Agregation des donnees
    sunburst_data = df.groupby(['age_group', 'gender', 'dropout_risk']).size().reset_index(name='count')
    sunburst_data['risk_label'] = sunburst_data['dropout_risk'].map({0: 'Non abandon', 1: 'Abandon'})
    
    fig = px.sunburst(
        sunburst_data,
        path=['age_group', 'gender', 'risk_label'],
        values='count',
        color='risk_label',
        color_discrete_map={'Non abandon': 'lightgreen', 'Abandon': 'salmon'},
        title="Hierarchie des abandons par age et sexe"
    )
    fig.update_layout(height=500)
    return fig

def create_treemap(df):
    """Treemap des abandons"""
    treemap_data = df.groupby(['grade_category', 'absenteism_category', 'dropout_risk']).size().reset_index(name='count')
    treemap_data['risk_label'] = treemap_data['dropout_risk'].map({0: 'Non abandon', 1: 'Abandon'})
    
    fig = px.treemap(
        treemap_data,
        path=['grade_category', 'absenteism_category', 'risk_label'],
        values='count',
        color='risk_label',
        color_discrete_map={'Non abandon': '#90EE90', 'Abandon': '#FF6B6B'},
        title="Analyse des abandons par niveau scolaire et absenteisme"
    )
    fig.update_layout(height=500)
    return fig

def create_time_analysis(df):
    """Analyse par age (simulation temporelle)"""
    age_stats = df.groupby('age').agg({
        'dropout_risk': 'mean',
        'average_grade': 'mean',
        'absenteeism_rate': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=age_stats['age'], y=age_stats['dropout_risk'] * 100,
                  name="Taux d'abandon (%)", line=dict(color='red', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=age_stats['age'], y=age_stats['average_grade'],
                  name="Moyenne generale", line=dict(color='green', width=3, dash='dash')),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Age")
    fig.update_yaxes(title_text="Taux d'abandon (%)", secondary_y=False)
    fig.update_yaxes(title_text="Moyenne generale (/20)", secondary_y=True)
    fig.update_layout(title="Evolution du risque d'abandon avec l'age", height=400)
    
    return fig

# ============================================
# FONCTIONS DE PREDICTION
# ============================================

def compute_features(age, average_grade, absenteeism_rate, study_time_hours):
    """Calcule les features engineering"""
    presence_ratio = 1 - absenteeism_rate
    effort_score = study_time_hours * average_grade / 20
    global_score = (average_grade * 0.5 + 
                    presence_ratio * 20 * 0.3 + 
                    study_time_hours * 2 * 0.2)
    return presence_ratio, effort_score, global_score

def predict_risk(model_data, input_df):
    """Effectue la prediction et retourne le resultat"""
    # Extraction des composants du modele
    model = model_data['model']
    scaler = model_data['scaler']
    le_gender = model_data['le_gender']
    le_internet = model_data['le_internet']
    le_activities = model_data['le_activities']
    features = model_data['features']
    
    # Encodage des variables categoriques
    input_encoded = input_df.copy()
    input_encoded['gender'] = le_gender.transform(input_encoded['gender'])
    input_encoded['internet_access'] = le_internet.transform(input_encoded['internet_access'])
    input_encoded['extra_activities'] = le_activities.transform(input_encoded['extra_activities'])
    
    # Selection des features
    X_input = input_encoded[features]
    
    # Normalisation
    X_scaled = scaler.transform(X_input)
    
    # Prediction
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return prediction, probabilities

def get_recommendations(risk_level, features):
    """Genere des recommandations personnalisees"""
    recommendations = []
    
    if risk_level == 1:
        recommendations.append("URGENCE : Plan de soutien pedagogique immediat")
        
        if features["average_grade"] < 10:
            recommendations.append("Moyenne faible : Mettre en place des cours de soutien et tutorat")
        
        if features["absenteeism_rate"] > 0.3:
            recommendations.append("Absenteisme eleve : Contacter la famille et comprendre les causes")
        
        if features["study_time_hours"] < 1:
            recommendations.append("Temps d'etude insuffisant : Encourager une routine d'etude quotidienne")
        
        if features["internet_access"] == "No":
            recommendations.append("Pas d'acces Internet : Proposer un acces au CDI ou des ressources hors ligne")
    else:
        recommendations.append("Situation favorable : Continuer le suivi regulier")
        
        if features["average_grade"] > 14:
            recommendations.append("Excellents resultats academiques - Felicitations")
        
        if features["study_time_hours"] >= 3:
            recommendations.append("Bonne discipline d'etude - Maintenir cette routine")
    
    if not recommendations:
        recommendations.append("Aucune action immediate requise - Suivi standard")
    
    return recommendations

def create_prediction_gauge(risk_proba):
    """Cree une jauge pour la prediction individuelle"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_proba * 100,
        title={"text": "Probabilite d'abandon (%)", "font": {"size": 18}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred", "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "yellow"},
                {"range": [60, 100], "color": "salmon"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": risk_proba * 100
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_radar_chart(features):
    """Cree un graphique radar du profil etudiant"""
    categories = ['Moyenne', 'Presence', 'Temps etude', 'Score global']
    values = [
        features["average_grade"] / 20 * 100,
        (1 - features["absenteeism_rate"]) * 100,
        features["study_time_hours"] / 10 * 100,
        features["global_score"] / 20 * 100
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='blue'),
        name='Profil etudiant'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        height=300,
        title="Profil de l'etudiant (%)"
    )
    return fig

# ============================================
# INTERFACE PRINCIPALE
# ============================================

def main():
    # Titre
    st.title("Tableau de Bord - Analyse du Risque d'Abandon Scolaire")
    st.markdown("---")
    
    # Chargement des donnees et du modele
    df = load_data()
    model_data = load_model()
    
    # Creation des onglets
    tab1, tab2 = st.tabs(["ANALYSE STATISTIQUE", "PREDICTION INDIVIDUELLE"])
    
    # ========================================
    # TAB 1: ANALYSE STATISTIQUE
    # ========================================
    with tab1:
        # Sidebar avec filtres (uniquement pour l'analyse)
        with st.sidebar:
            st.markdown("## Filtres")
            
            # Filtres
            age_range = st.slider("Tranche d'age", 15, 24, (15, 24))
            gender_filter = st.multiselect("Sexe", ["Male", "Female"], default=["Male", "Female"])
            internet_filter = st.multiselect("Acces Internet", ["Yes", "No"], default=["Yes", "No"])
            
            st.markdown("---")
            st.markdown("### Statistiques globales")
            st.markdown(f"- **Total etudiants:** {len(df)}")
            st.markdown(f"- **Abandons:** {df['dropout_risk'].sum()}")
            st.markdown(f"- **Taux global:** {(df['dropout_risk'].mean()*100):.1f}%")
            
            st.markdown("---")
            st.markdown("### Objectif du dashboard")
            st.markdown("""
            Ce dashboard permet de :
            - Visualiser les tendances d'abandon
            - Identifier les facteurs de risque
            - Suivre les indicateurs cles
            - Predire le risque individuel
            """)
        
        # Application des filtres
        filtered_df = df[
            (df['age'].between(age_range[0], age_range[1])) &
            (df['gender'].isin(gender_filter)) &
            (df['internet_access'].isin(internet_filter))
        ]
        
        # KPI Cards
        st.subheader("Indicateurs cles de performance")
        create_kpi_cards(filtered_df)
        
        # Row 1: Jauge et Taux d'abandon par age
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dropout_rate = (filtered_df["dropout_risk"].mean() * 100)
            fig_gauge = create_risk_gauge(dropout_rate)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            fig_age = create_dropout_by_variable(filtered_df, 'age_group', "Taux d'abandon par tranche d'age")
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Row 2: Distribution comparisons
        st.subheader("Distribution des variables selon le risque")
        fig_dist = create_distribution_comparison(filtered_df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Row 3: Deux graphiques cote a cote
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gender = create_dropout_by_variable(filtered_df, 'gender', "Taux d'abandon par sexe")
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            fig_internet = create_dropout_by_variable(filtered_df, 'internet_access', "Taux d'abandon par acces Internet")
            st.plotly_chart(fig_internet, use_container_width=True)
        
        # Row 4: Matrice de correlation
        st.subheader("Relations entre les variables")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_corr = create_correlation_heatmap(filtered_df)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            fig_importance = create_feature_importance(filtered_df)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Row 5: Graphiques avances
        st.subheader("Analyses avancees")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sunburst = create_sunburst_chart(filtered_df)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            fig_treemap = create_treemap(filtered_df)
            st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Row 6: Analyse temporelle et activités extrascolaires
        col1, col2 = st.columns(2)
        
        with col1:
            fig_time = create_time_analysis(filtered_df)
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            fig_activities = create_dropout_by_variable(filtered_df, 'extra_activities', 
                                                         "Taux d'abandon par activités extrascolaires")
            st.plotly_chart(fig_activities, use_container_width=True)
        
        # Row 7: Tableau detaille
        st.subheader("Detail des donnees filtrees")
        
        # Selection des colonnes a afficher
        display_cols = ["age", "gender", "average_grade", "absenteeism_rate", 
                        "study_time_hours", "internet_access", "extra_activities", "dropout_risk"]
        
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            height=400,
            column_config={
                "dropout_risk": st.column_config.SelectboxColumn(
                    "Risque abandon",
                    options=[0, 1],
                    help="0 = Non, 1 = Oui"
                )
            }
        )
        
        # Export des donnees
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Telecharger les donnees filtrees (CSV)",
            data=csv,
            file_name=f"donnees_abandon_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    # ========================================
    # TAB 2: PREDICTION INDIVIDUELLE
    # ========================================
    with tab2:
        st.markdown("## Prediction du risque d'abandon individuel")
        st.markdown("Saisissez les informations de l'etudiant pour obtenir une prediction personnalisee.")
        st.markdown("---")
        
        # Verification du modele
        if model_data is None:
            st.error("Modele non trouve. Veuillez d'abord executer le script generate_model.py")
            
            # Afficher les instructions
            with st.expander("Comment generer le modele ?"):
                st.markdown("""
                1. Creez un fichier `generate_model.py` avec le code ci-dessous
                2. Executez `python generate_model.py`
                3. Relancez cette application
                """)
                
                st.code("""
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.makedirs("./models", exist_ok=True)

df = pd.read_csv("./data/student_dropout_dataset.csv")

df["presence_ratio"] = 1 - df["absenteeism_rate"]
df["effort_score"] = df["study_time_hours"] * df["average_grade"] / 20
df["global_score"] = (df["average_grade"] * 0.5 + df["presence_ratio"] * 20 * 0.3 + df["study_time_hours"] * 2 * 0.2)

num_cols = ["age", "average_grade", "absenteeism_rate", "study_time_hours", "presence_ratio", "effort_score", "global_score"]
cat_cols = ["gender", "internet_access", "extra_activities"]

X = df.drop("dropout_risk", axis=1)
y = df["dropout_risk"]

le_gender = LabelEncoder()
le_internet = LabelEncoder()
le_activities = LabelEncoder()

X['gender'] = le_gender.fit_transform(X['gender'])
X['internet_access'] = le_internet.fit_transform(X['internet_access'])
X['extra_activities'] = le_activities.fit_transform(X['extra_activities'])

features = num_cols + ['gender', 'internet_access', 'extra_activities']
X_final = X[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

model_data = {
    'model': model,
    'scaler': scaler,
    'le_gender': le_gender,
    'le_internet': le_internet,
    'le_activities': le_activities,
    'features': features
}

with open("./models/dropout_model.pkl", 'wb') as f:
    pickle.dump(model_data, f)

print("Modele genere avec succes!")
                """, language="python")
        else:
            # Organisation en deux colonnes
            col_left, col_right = st.columns([1, 1], gap="large")
            
            with col_left:
                st.markdown("### Informations etudiant")
                
                with st.expander("Donnees personnelles", expanded=True):
                    age = st.number_input("Age", min_value=15, max_value=30, value=18, step=1, key="pred_age")
                    gender = st.selectbox("Sexe", ["Male", "Female"], key="pred_gender")
                
                with st.expander("Donnees academiques", expanded=True):
                    average_grade = st.number_input(
                        "Moyenne generale (/20)", 
                        min_value=0.0, 
                        max_value=20.0, 
                        value=12.0, 
                        step=0.1,
                        key="pred_grade"
                    )
                    
                    absenteeism_rate = st.slider(
                        "Taux d'absenteisme", 
                        min_value=0.0, 
                        max_value=0.5, 
                        value=0.2, 
                        step=0.01,
                        format="%.2f",
                        key="pred_absenteism"
                    )
                    
                    study_time_hours = st.number_input(
                        "Temps d'etude (heures/jour)", 
                        min_value=0.0, 
                        max_value=10.0, 
                        value=2.0, 
                        step=0.5,
                        key="pred_study"
                    )
                
                with st.expander("Donnees contextuelles", expanded=True):
                    internet_access = st.selectbox("Acces a Internet", ["Yes", "No"], key="pred_internet")
                    extra_activities = st.selectbox("Activites extrascolaires", ["Yes", "No"], key="pred_activities")
                
                # Bouton de prediction
                st.markdown("---")
                predict_button = st.button("Predire le risque d'abandon", type="primary", use_container_width=True)
            
            with col_right:
                st.markdown("### Resultat de la prediction")
                st.markdown("---")
                
                if predict_button:
                    # Calcul des features engineering
                    presence_ratio, effort_score, global_score = compute_features(
                        age, average_grade, absenteeism_rate, study_time_hours
                    )
                    
                    # Creation du DataFrame d'entree
                    input_data = pd.DataFrame([[
                        age, average_grade, absenteeism_rate, study_time_hours,
                        presence_ratio, effort_score, global_score,
                        gender, internet_access, extra_activities
                    ]], columns=[
                        "age", "average_grade", "absenteeism_rate", "study_time_hours",
                        "presence_ratio", "effort_score", "global_score",
                        "gender", "internet_access", "extra_activities"
                    ])
                    
                    # Prediction (utilisation de la nouvelle fonction)
                    prediction, probabilities = predict_risk(model_data, input_data)
                    risk_proba = probabilities[1]
                    
                    # Affichage du resultat
                    if prediction == 1:
                        st.error("RISQUE ELEVE D'ABANDON")
                        st.markdown(f"""
                        <div style='background-color: #ff4444; padding: 20px; border-radius: 10px; text-align: center;'>
                            <h2 style='color: white;'>Risque critique</h2>
                            <p style='color: white; font-size: 24px;'>Probabilite : {risk_proba:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("RISQUE FAIBLE D'ABANDON")
                        st.markdown(f"""
                        <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;'>
                            <h2 style='color: white;'>Situation favorable</h2>
                            <p style='color: white; font-size: 24px;'>Probabilite d'abandon : {risk_proba:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Jauge de probabilite
                    fig_gauge_pred = create_prediction_gauge(risk_proba)
                    st.plotly_chart(fig_gauge_pred, use_container_width=True)
                    
                    # Graphique radar
                    features_info = {
                        "average_grade": average_grade,
                        "absenteeism_rate": absenteeism_rate,
                        "study_time_hours": study_time_hours,
                        "global_score": global_score
                    }
                    fig_radar = create_radar_chart(features_info)
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Recommandations personnalisees
                    st.markdown("### Recommandations personnalisees")
                    features_rec = {
                        "average_grade": average_grade,
                        "absenteeism_rate": absenteeism_rate,
                        "study_time_hours": study_time_hours,
                        "internet_access": internet_access
                    }
                    recommendations = get_recommendations(prediction, features_rec)
                    
                    for rec in recommendations:
                        st.info(rec)
                    
                    # Details des features calculees
                    with st.expander("Detail des caracteristiques calculees"):
                        st.markdown(f"""
                        - **Ratio presence/absence :** {presence_ratio:.2f}
                        - **Score d'effort :** {effort_score:.2f}
                        - **Score global :** {global_score:.2f}
                        """)
                        
                        # Regle metier
                        st.markdown("### Regle metier du dataset")
                        conditions = []
                        if average_grade < 10:
                            conditions.append("Moyenne < 10")
                        if absenteeism_rate > 0.3:
                            conditions.append("Absenteisme > 30%")
                        if study_time_hours < 1:
                            conditions.append("Temps d'etude < 1h")
                        
                        st.markdown(f"**Conditions verifiees :** {len(conditions)}/3")
                        for c in conditions:
                            st.markdown(f"- {c}")
                        
                        if len(conditions) >= 2:
                            st.warning("Selon la regle metier, cet etudiant serait classe a risque.")
                        else:
                            st.success("Selon la regle metier, cet etudiant ne serait pas classe a risque.")
                else:
                    st.info("Remplissez le formulaire et cliquez sur 'Predire' pour voir le resultat")
                    
                    # Affichage d'un exemple
                    st.markdown("---")
                    st.markdown("### Exemple de profil a risque")
                    st.markdown("""
                    - Age : 18 ans
                    - Moyenne : 8.5/20
                    - Absenteisme : 35%
                    - Temps d'etude : 0.5h/jour
                    - Pas d'acces Internet
                    
                    -> Ce profil correspond a un risque eleve d'abandon.
                    """)
    
    # Footer commun
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Dashboard realise dans le cadre du Master 2 Data Science - Analyse predictive de l'abandon scolaire</p>
        <p>Donnees mises a jour en temps reel | Derniere analyse : {}</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)

# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    main()