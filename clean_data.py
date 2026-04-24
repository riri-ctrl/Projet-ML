# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:06:20 2026

@author: grabe
"""

import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np




def drop_col(dataset):
    dataset=dataset.drop(["id","Name","City","CGPA"],axis=1)
    return dataset

# CGPA temporaire ====> moyenne scolaire 



def fusion(dataset):
    liste = ["Academic Pressure","Work Pressure","Study Satisfaction","Job Satisfaction"]
    for col in liste:
        dataset[col] = dataset[col].fillna(0)                   
        
    dataset["Pressure"]=dataset["Academic Pressure"] + dataset["Work Pressure"]
    dataset["Satisfaction"]=dataset["Study Satisfaction"] + dataset["Job Satisfaction"]
    
    dataset=dataset.drop(liste,axis=1)
    return dataset



def gestion_profession_nan(dataset):
    dataset.loc[(dataset["Working Professional or Student"] == "Student"),"Profession"] = "Student" 
    
    dataset["Profession"] = dataset["Profession"].fillna("Unknown") #ne pas indiqué son job alors qu'il travaille peut être un signe de dépression
    return dataset



def plus_frequent(liste,dataset):
    imputer= SimpleImputer(strategy='most_frequent')
    for col in liste:
        dataset[[col]]= imputer.fit_transform(dataset[[col]])
    return dataset
    


"""
def value_col(dataset):
    for col in dataset:
        print(dataset[col].value_counts())

value_col(dataset)
    
    connaitre les valeurs de chaque colonne du dataset ainsi que les plus présente
    
pd.set_option("display.max_rows", None)
print(dataset["Profession"].value_counts())
"""
def gestion_bruit_encoder (dataset):
    top4 = [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    ]
    
    dataset = dataset[dataset["Sleep Duration"].isin(top4)]
    
    mapping = {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }
    
    dataset["Sleep_Duration"] = dataset["Sleep Duration"].map(mapping)
    
    top3 =[
           "Moderate",
           "Unhealthy",
           "Healthy"
    ]
    dataset = dataset[dataset["Dietary Habits"].isin(top3)]
    mapping2={
              "Moderate":1,
              "Unhealthy":0,
              "Healthy":2
    }
    
    
    dataset["Dietary Habits"] = dataset["Dietary Habits"].map(mapping2)
    return dataset



def gestion_bruit(liste,dataset):
    for col in liste :
        counts = dataset[col].value_counts()  
        valid_degree = counts[counts >= 10].index
        dataset = dataset[dataset[col].isin(valid_degree)]
    return dataset





def categorical_features(dataset):
    str_col = dataset.select_dtypes(include='str').columns
    label_encoder = LabelEncoder()
    for col in str_col:
        dataset[col] = label_encoder.fit_transform(dataset[col])
    return dataset




def print_info(dataset):
    print(dataset.info())
    print(dataset.select_dtypes(include=['object', 'string', 'int', 'float', 'bool']).isnull().sum())

def main():
    dataset=pd.read_csv('test.csv')
    
    dataset=drop_col(dataset)
    
    dataset=fusion(dataset)
    
    dataset=gestion_profession_nan(dataset)
    
    dataset = plus_frequent(["Degree","Dietary Habits"],dataset)
    
    dataset=gestion_bruit_encoder(dataset)
    
    dataset=gestion_bruit(["Degree","Profession"],dataset)
    
    dataset = categorical_features(dataset)
    
    print_info(dataset)
    
if __name__ == "__main__":
    main()
    
