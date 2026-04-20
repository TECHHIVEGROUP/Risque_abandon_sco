# generate_model.py
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Creation du dossier models
os.makedirs("./models", exist_ok=True)

print("Chargement des donnees...")
df = pd.read_csv("./data/student_dropout_dataset.csv")

print("Feature engineering...")
df["presence_ratio"] = 1 - df["absenteeism_rate"]
df["effort_score"] = df["study_time_hours"] * df["average_grade"] / 20
df["global_score"] = (df["average_grade"] * 0.5 + 
                      df["presence_ratio"] * 20 * 0.3 + 
                      df["study_time_hours"] * 2 * 0.2)

# Colonnes
num_cols = ["age", "average_grade", "absenteeism_rate", "study_time_hours", 
            "presence_ratio", "effort_score", "global_score"]
cat_cols = ["gender", "internet_access", "extra_activities"]

X = df.drop("dropout_risk", axis=1)
y = df["dropout_risk"]

print("Encodage des variables categoriques...")
le_gender = LabelEncoder()
le_internet = LabelEncoder()
le_activities = LabelEncoder()

X['gender'] = le_gender.fit_transform(X['gender'])
X['internet_access'] = le_internet.fit_transform(X['internet_access'])
X['extra_activities'] = le_activities.fit_transform(X['extra_activities'])

features = num_cols + ['gender', 'internet_access', 'extra_activities']
X_final = X[features]

print("Normalisation...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

print("Entrainement du modele...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Sauvegarde
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

print("Modele genere avec succes dans ./models/dropout_model.pkl")
print(f"Taille du fichier: {os.path.getsize('./models/dropout_model.pkl') / 1024:.2f} KB")