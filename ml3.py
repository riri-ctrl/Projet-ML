# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 23:33:52 2026

@author: grabe et manu
"""



from clean_data import main

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



df = main()

df.plot.box()

#df = df.sample(frac=0.2,random_state=1)

#permet de réduire la taille du dataset lors de la programmation
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html


def corr_matrice(dataframe):
    matr = dataframe.corr()
    plt.figure(figsize=(20,12))
    sns.heatmap(data=matr,annot=True)
    plt.show()
    return matr 
def corr_features(dataframe):
    matr=corr_matrice(dataframe)
    
    corr_matrix = matr["Depression"]
    
    selected_features = corr_matrix[abs(corr_matrix) > 0.1] #Tester plusieur valeur de drop
    return selected_features

select_features=corr_features(df)
df_filtered = df[select_features.index]
select_features.drop("Depression")
new_corr=corr_matrice(df_filtered)


X= df.drop("Depression", axis=1)
X_filtered= df_filtered.drop("Depression", axis=1)
y=df["Depression"]

print(np.mean(X))
print(np.std(X))



def feature_importante(df_feature):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(df_feature, y)
    
    importance_df = pd.DataFrame({
        'Variable': df_feature.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.barh(df_feature.columns, rf.feature_importances_) 
    plt.show()
    return importance_df

importance_df= feature_importante(X)
importance_df_filtered= feature_importante(X_filtered)




#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

def ml (dataframe,result):
    
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe, result, test_size = 0.8, random_state =42, stratify =result)
   
    

    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    dt = DecisionTreeClassifier(max_depth=5)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(max_iter=1500)
    rd = RandomForestClassifier(n_estimators=250)
    classifiers = [('decision',dt),('foret',rd), ('knn', knn), ('lr', lr)]
    vc = VotingClassifier(estimators=classifiers)
    vc.fit(X_train, y_train)
    
    pred = vc.predict(X_test)
    
    print(accuracy_score(y_test,pred))
    
    
    
    
ml(X_filtered,y)

"""
def best_parametre(X_train,y_train):
    
    return grid.best_params_,grid.best_score_
"""


def best_param(df,Y):
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        Y,
                                                        test_size = 0.2,
                                                        random_state =42,
                                                        stratify =y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    dt1 = DecisionTreeClassifier()
    param = {
        "criterion": ['gini', 'entropy'],
        "max_depth": np.arange(2,15),
        "min_samples_split": np.arange(2,15),
        "min_samples_leaf": np.arange(1,10),
        "max_features": [None, "sqrt", "log2"],
        
    }
    
    
    grid = GridSearchCV(dt1,param_grid=param, cv=5)
    grid.fit(X_train,y_train)
    print(grid.best_params_)
    print(grid.best_score_)
    
    #{'criterion': 'gini', 'max_depth': np.int64(9), 'max_features': None, 'min_samples_leaf': np.int64(7), 'min_samples_split': np.int64(2)}
    #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    
    knn1=KNeighborsClassifier()
    param = {
        "n_neighbors": np.arange(1,15),
        "weights": ['uniform', 'distance'],
        "p": [1, 2]  # Manhattan vs Euclidean
    }
    
    grid2 = GridSearchCV(knn1,param_grid=param, cv=5)
    grid2.fit(X_train,y_train)
    print(grid2.best_params_)
    print(grid2.best_score_)
    #{'n_neighbors': np.int64(14), 'p': 1, 'weights': 'uniform'}
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    lr1 = LogisticRegression()
    param = [
        {   
             "max_iter":1500,  #====> max_iter ne change pas les résultats
            "solver": ["lbfgs"],
            "penalty": ["l2"],
            "C": np.logspace(-3, 2, 6)
        },
        {   
            "max_iter":1500,
            "solver": ["saga"],
            "penalty": ["elasticnet"],
            "C": np.logspace(-3, 2, 6),
            "l1_ratio": [0.2, 0.5, 0.8]
        }
    ]
    
    #penalty ne marche pas avec tous les solver
        
    grid3 = GridSearchCV(lr1,param_grid= param, cv=5)
    grid3.fit(X_train,y_train)
    print(grid3.best_params_)
    print(grid3.best_score_)
        
    #{'C': np.float64(0.1), 'penalty': 'l2', 'solver': 'lbfgs'}
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    
    """
    "n_estimators": [10,50, 75],
    "criterion": ['gini', 'entropy'],    #demander la dif entre entropy et log_loss
    "max_depth": [2,8,15],
    "min_samples_split": [2,8,15],
    "min_samples_leaf":[2,8,15],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    0.9333778371161549
    
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 15, 'n_estimators': 75}
    
    "n_estimators": [50, 75,100],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [10,15,20],
    "min_samples_split": [10,15,20],
    "min_samples_leaf":[2,4,8],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    {'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 100}
    0.9336448598130841
    
    "n_estimators": [75,100,150],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [5,10,15,20],
    "min_samples_split": [15,20,25],
    "min_samples_leaf":[2,3,4],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 15, 'n_estimators': 75}
    0.9336893635959056
    
    "n_estimators": [75,85,100],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [10,13,15,17,20],
    "min_samples_split": [15,17,20],
    "min_samples_leaf":[2,3],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    {'class_weight': None, 'criterion': 'gini', 'max_depth': 13, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 15, 'n_estimators': 85}
    0.9338673787271917
    
    "n_estimators": [75,80,85,90,100],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [10,11,12,13,14],
    "min_samples_split": [13,15,17],
    "min_samples_leaf":[2,3,4],
    "max_features": ["log2"],
    "class_weight": [None]
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': 12, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 17, 'n_estimators': 90}
    0.9337338673787272
    
    
    "n_estimators": np.arange(80,100,2),
    "criterion": ['gini', 'entropy'],    
    "max_depth": np.arange(10,15),
    "min_samples_split": np.arange(13,20),
    "min_samples_leaf":np.arange(2,5),
    "max_features": ["log2"],
    "class_weight": [None]
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': np.int64(13), 'max_features': 'log2', 'min_samples_leaf': np.int64(3), 'min_samples_split': np.int64(15), 'n_estimators': np.int64(98)}
    0.9342679127725857
    """
    rd1 = RandomForestClassifier()
    param = {
        
        "n_estimators": np.arange(90,100),
        "criterion": ['gini', 'entropy'],    
        "max_depth": np.arange(10,15),
        "min_samples_split": np.arange(13,20),
        "min_samples_leaf":np.arange(2,5),
        "max_features": ["log2"],
        "class_weight": [None]
        
        
        }
    
    grid4 = GridSearchCV(rd1,param_grid= param, cv=3,n_jobs=-1)
    grid4.fit(X_train,y_train)
    print(grid4.best_params_)
    print(grid4.best_score_)
    #{'class_weight': None, 'criterion': 'gini', 'max_depth': np.int64(12), 'max_features': 'log2', 'min_samples_leaf': np.int64(2), 'min_samples_split': np.int64(14), 'n_estimators': np.int64(99)}
    #0.9337338673787272
    
    
#best_param(X_filtered,y)

def best_param(df,Y):
    X_train, X_test, y_train, y_test = train_test_split(df,
                                                        Y,
                                                        test_size = 0.2,
                                                        random_state =42,
                                                        stratify =y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    dt1 = DecisionTreeClassifier()
    param = {
        "criterion": ['gini', 'entropy'],
        "max_depth": np.arange(2,15),
        "min_samples_split": np.arange(2,15),
        "min_samples_leaf": np.arange(1,10),
        "max_features": [None, "sqrt", "log2"],
        
    }
    
    
    grid = GridSearchCV(dt1,param_grid=param, cv=5)
    grid.fit(X_train,y_train)
    print(grid.best_params_)
    print(grid.best_score_)
    
    #{'criterion': 'gini', 'max_depth': np.int64(9), 'max_features': None, 'min_samples_leaf': np.int64(7), 'min_samples_split': np.int64(2)}
    #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    
    knn1=KNeighborsClassifier()
    param = {
        "n_neighbors": np.arange(1,15),
        "weights": ['uniform', 'distance'],
        "p": [1, 2]  # Manhattan vs Euclidean
    }
    
    grid2 = GridSearchCV(knn1,param_grid=param, cv=5)
    grid2.fit(X_train,y_train)
    print(grid2.best_params_)
    print(grid2.best_score_)
    #{'n_neighbors': np.int64(14), 'p': 1, 'weights': 'uniform'}
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    lr1 = LogisticRegression()
    param = [
        {   
             "max_iter":1500,  #====> max_iter ne change pas les résultats
            "solver": ["lbfgs"],
            "penalty": ["l2"],
            "C": np.logspace(-3, 2, 6)
        },
        {   
            "max_iter":1500,
            "solver": ["saga"],
            "penalty": ["elasticnet"],
            "C": np.logspace(-3, 2, 6),
            "l1_ratio": [0.2, 0.5, 0.8]
        }
    ]
    
    #penalty ne marche pas avec tous les solver
        
    grid3 = GridSearchCV(lr1,param_grid= param, cv=5)
    grid3.fit(X_train,y_train)
    print(grid3.best_params_)
    print(grid3.best_score_)
        
    #{'C': np.float64(0.1), 'penalty': 'l2', 'solver': 'lbfgs'}
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    
    """
    "n_estimators": [10,50, 75],
    "criterion": ['gini', 'entropy'],    #demander la dif entre entropy et log_loss
    "max_depth": [2,8,15],
    "min_samples_split": [2,8,15],
    "min_samples_leaf":[2,8,15],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    0.9333778371161549
    
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 15, 'n_estimators': 75}
    
    "n_estimators": [50, 75,100],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [10,15,20],
    "min_samples_split": [10,15,20],
    "min_samples_leaf":[2,4,8],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    {'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 100}
    0.9336448598130841
    
    "n_estimators": [75,100,150],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [5,10,15,20],
    "min_samples_split": [15,20,25],
    "min_samples_leaf":[2,3,4],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 15, 'n_estimators': 75}
    0.9336893635959056
    
    "n_estimators": [75,85,100],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [10,13,15,17,20],
    "min_samples_split": [15,17,20],
    "min_samples_leaf":[2,3],
    "max_features": [ "sqrt", "log2"],
    "class_weight": [None, "balanced"]
    {'class_weight': None, 'criterion': 'gini', 'max_depth': 13, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 15, 'n_estimators': 85}
    0.9338673787271917
    
    "n_estimators": [75,80,85,90,100],
    "criterion": ['gini', 'entropy'],    
    "max_depth": [10,11,12,13,14],
    "min_samples_split": [13,15,17],
    "min_samples_leaf":[2,3,4],
    "max_features": ["log2"],
    "class_weight": [None]
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': 12, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 17, 'n_estimators': 90}
    0.9337338673787272
    
    
    "n_estimators": np.arange(80,100,2),
    "criterion": ['gini', 'entropy'],    
    "max_depth": np.arange(10,15),
    "min_samples_split": np.arange(13,20),
    "min_samples_leaf":np.arange(2,5),
    "max_features": ["log2"],
    "class_weight": [None]
    {'class_weight': None, 'criterion': 'entropy', 'max_depth': np.int64(13), 'max_features': 'log2', 'min_samples_leaf': np.int64(3), 'min_samples_split': np.int64(15), 'n_estimators': np.int64(98)}
    0.9342679127725857
    """
    rd1 = RandomForestClassifier()
    param = {
        
        "n_estimators": np.arange(90,100),
        "criterion": ['gini', 'entropy'],    
        "max_depth": np.arange(10,15),
        "min_samples_split": np.arange(13,20),
        "min_samples_leaf":np.arange(2,5),
        "max_features": ["log2"],
        "class_weight": [None]
        
        
        }
    
    grid4 = GridSearchCV(rd1,param_grid= param, cv=3,n_jobs=-1)
    grid4.fit(X_train,y_train)
    print(grid4.best_params_)
    print(grid4.best_score_)
    #{'class_weight': None, 'criterion': 'gini', 'max_depth': np.int64(12), 'max_features': 'log2', 'min_samples_leaf': np.int64(2), 'min_samples_split': np.int64(14), 'n_estimators': np.int64(99)}
    #0.9337338673787272
    
    
#best_param(X_filtered,y)

def seuil(proba,y_test,model):
    """
    pred = vc.predict(X_test)
    print("Accuracy Voting :", accuracy_score(y_test, pred))
    """
    
    for t in [0.3, 0.4, 0.5,0.6]:
        
        pred = (proba > t).astype(int)
        
        print(f"\nSeuil = {t}")
        print(f"{model} :", classification_report(y_test, pred))
        cm=confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(f"Vrais négatifs (TN) : {tn}")
        print(f"Faux positifs (FP) : {fp}")
        print(f"Faux négatifs (FN) : {fn}")
        print(f"Vrais positifs (TP) : {tp}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Négatif", "Positif"])
        disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' = entier
        plt.title("Matrice de confusion")
        plt.show()

def init_model_best_para():
    dt = DecisionTreeClassifier(
       criterion='gini',
       max_depth=9,
       min_samples_split=2,
       min_samples_leaf=7,
       max_features= None
       )
    
    rd = RandomForestClassifier(
        class_weight=None,
        criterion='gini',
        max_depth=12,
        max_features='log2',
        min_samples_leaf=2,
        min_samples_split=14,
        n_estimators=99
    )
    
    knn = Pipeline([
       ('scaler', StandardScaler()),
       ('knn', KNeighborsClassifier(
           n_neighbors=14,
           p=1,
           weights='uniform'
       ))
    ])

    lr = Pipeline([
       ('scaler', StandardScaler()),
       ('lr', LogisticRegression(
           max_iter=1500,
           C=0.1,
           penalty='l2',
           solver='lbfgs'
       ))
   ])
    return dt,rd,knn,lr
    

def ROC(models,X_train, X_test, y_train, y_test):
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(accuracy_score(y_test, pred))
        print("Train :", model.score(X_train, y_train))
        print("Test :", model.score(X_test, y_test))
        print(f"${model} :", classification_report(y_test, pred))
        proba = model.predict_proba(X_test)[:,1]
        
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=name)
    
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.show()
    
def ml_best_param (dataframe,result):
    
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe, result, test_size = 0.8, random_state =42, stratify =result)
    
    
    dt,rd,knn,lr=init_model_best_para()
    
    models = {
        "dt":dt,
        "Rd": rd,
        "LR": lr,
        "KNN": knn
    }
    
    ROC(models,X_train, X_test, y_train, y_test)
    
    classifiers = [
       
       ('foret', rd),
       ('knn', knn),
       ('lr', lr)
    ]
    vc = VotingClassifier(
        estimators=classifiers,
        voting='soft'
    )
    
    vc.fit(X_train, y_train)
    
    proba = vc.predict_proba(X_test)[:,1]
    seuil(proba,y_test,vc)
    
        
    joblib.dump(vc, "voting", compress=3)
    return vc
    
    
    
best_models=ml_best_param(X_filtered,y)


def predict_person(model, data,columns, threshold=0.4):
    
    df = pd.DataFrame([data])
    df = df[columns]
    
    proba = model.predict_proba(df)[0][1]
    
    pred = int(proba > threshold)
    
    print("Probabilité :", proba)
    print("Prédiction :", pred)
    
    return pred



nouvelle_donnee = {
    "Age": 25,
    "Working Professional or Student":1,
    "Have you ever had suicidal thoughts ?":1,
    "Work/Study Hours": 8,
    "Pressure": 5,
    "Profession":5,
    "Financial Stress": 6,
    "Satisfaction": 4,
    "Dietary Habits":2,
    
}

print(predict_person(best_models,nouvelle_donnee,X_filtered.columns))
