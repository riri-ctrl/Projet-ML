from flask import Flask, render_template, request

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,roc_curve,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# 🔥 charger le modèle
model = joblib.load("voting.pkl")

# 🔥 ordre des colonnes (IMPORTANT)
columns = [
    "Age",
    "Working Professional or Student",
    "Profession",
    "Dietary Habits",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours",
    "Financial Stress",
    "Pressure",
    "Satisfaction"
    ] 

# page principale
@app.route("/form")
def form():
    return render_template("form.html")
@app.route("/graphe")
def graphe():
    return render_template("graphe.html")
# prédiction
@app.route("/predict", methods=["POST"])
def predict():
    
    data = {
        "Age": int(request.form["age"]),
        "Working Professional or Student": int(request.form["professional_or_student"]),
        "Profession": int(request.form["profession"]),
        "Dietary Habits": int(request.form["dietary_habits"]),
        "Have you ever had suicidal thoughts ?": int(request.form["suicidal_thoughts"]),
        "Work/Study Hours": int(request.form["work_hours"]),
        "Financial Stress": int(request.form["financial_stress"]),
        "Pressure": int(request.form["pressure"]),
        "Satisfaction": int(request.form["satisfaction"]),
        
    }

    df = pd.DataFrame([data])
      
    df = df.reindex(columns=columns)

    proba = model.predict_proba(df)[0][1]
    pred = int(proba > 0.5)

    return render_template("form.html", proba=proba, pred=pred)

# lancer le serveur
if __name__ == "__main__":
    app.run(debug=True)