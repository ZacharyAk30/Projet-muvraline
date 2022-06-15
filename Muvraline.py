import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#importation du dataframe
data=pd.read_csv("data.csv",sep=",")

#On change la chaine de caractère en date
data['datetime'] = pd.to_datetime(data['datetime'])

#Réindexation du dataframe
data.set_index('datetime', inplace=True)

#analyse du dataframe

print(data['count'].describe())
sns.distplot(data['count'])


#gestion des missing value
#il n'y a pas de missing value le data set est complet
#il ne manque aucune donné dans le DataFrame

#Matrice de corellation
def cor(data):
    corrmat =data.corr()
    f2, ax2 = plt.subplots(figsize=(12, 9))
    cols = corrmat.nlargest(11, 'count')['count'].index
    sns.set(font_scale=1.25)
    sns.heatmap(np.corrcoef(data[cols].values.T), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
#la matrice de corrélation nous apprend : 
    #temp et atemp sont corréllés
    #humidity n'est pas correlle a cout registered ou casual
clear_data=data.drop(['atemp','humidity'],axis=1)
    #count = casual + registered
    #il serai donc facile de crée un algorithme determinant le nombre de visiteur en fonction de ses deux donnés

    #nous allons donc entrainer 2 model permettant de determiner le nb de visiteur casual et le nb de visiteur registered.
X=clear_data.drop(['count','registered','casual'],axis=1)
Y=clear_data.drop(['temp','season','holiday','workingday','weather','windspeed'],axis=1)
Y1=Y.drop(['count','casual'],axis=1)
Y2=Y.drop(['count','registered'],axis=1)

#Separation du data set
X_train, X_test, y2_train, y2_test = train_test_split(X,Y2,test_size=0.30)
X_train, X_test, y1_train, y1_test = train_test_split(X,Y1,test_size=0.30)

#creation du model 1

model1 =LinearRegression()
model1.fit(X_train, y1_train)
print("model 1 : ",model1.score(X_test, y1_test))

#creation du model 2

model2 =LinearRegression()
model2.fit(X_train, y2_train)
print("model 2 : ",model2.score(X_test, y2_test))


