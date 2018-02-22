#importer les libs necessaires traitementdu donnees
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset

dataset = pd.read_csv('iris.csv')


#determiner les variables independantes et les variables dependantes
#et les mettre separement dans deux matrices

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#Gerer les donnees manquantes dans ce cas ils n'existent pas

#Gerer les variables categoriques on a le feature species qui est categorique
# on a 3 types de iris representer par la colonne de la matrice y
#pour la matrice X tout les valeurs des ses features sont numeriques
#donc pas besoin be l'encoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Y_encoder = LabelEncoder()
y = Y_encoder.fit_transform(y)
#onehotencoder = OneHotEncoder(categorical_features = [0])
#y= y.reshape(-1,1)
#y= onehotencoder.fit_transform(y).toarray()

# Diviser le dataset entre le Training set et le Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train)

#les valeurs des features ne  presentent pas un large decalage 
#donc c'est inutile dans ce cas de les standardiser en faisant le feature scalling


#construction du modele de machine learning baser sur la regression ploynomiale

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X_train)
#print (X_poly)
regressor = LinearRegression()
regressor.fit(X_poly,y_train)

# Visualiser les resultats
plt.scatter(X_train[:,:1],y_train, edgecolors='red')
plt.plot(X_train[:,:1], regressor.predict(X_poly), color = 'blue')
plt.title('f(Dimension du feuille )=Race du fleur')
plt.xlabel('Dimension du feuille')
plt.ylabel('Race du fleur')
plt.show()
# Visualiser les resultats (courbe plus lisse)
X_grid = np.arange(min(X_train), max(X_train), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train,y_train, edgecolors='red')    
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('f(Dimension du feuille )=Race du fleur')
plt.xlabel('Dimension du feuille')
plt.ylabel('Race du fleur')
plt.show()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
