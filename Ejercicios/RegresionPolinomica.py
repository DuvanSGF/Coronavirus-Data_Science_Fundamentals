
# coding: utf-8

# In[23]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

url = "https://raw.githubusercontent.com/DuvanSGF/Coronavirus-Data_Science_Fundamentals/master/Datasets/covidacumulado.csv"
data = pd.read_csv(url)  # load data set


# In[14]:


# Imprimimos los Datos
data


# In[15]:


x = data['dia'].values.reshape(-1, 1) # necesitamos un array de 2D para SkLearn
y = data['contagiados'].values.reshape(-1, 1)
plt.scatter(x,y)


# In[16]:


# Ajuste a un modelo lineal
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()


# In[ ]:


# No parece que el modelo lineal se ajuste demasiado bien. 
# Una forma de medir la bondad del ajuste es calcular la 
# media de la raíz del error cuadrático (root mean square error) 
# que nos da una medida del error cometido por el modelo. 
# En concreto se calcula la media de la desviación de los valores estimados. 
# Otra métrica interesante es la medida R2 (R cuadrado), cuyo valor 
# está entre 0 y 1, lo que la hace mejor a la hora 
# de interpretar su valor, y es la fracción de la suma total de 
# cuadrados que se 'explica por' la regresión.


# In[18]:


rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))


# In[ ]:


# Si nos fijamos, sobre todo en R2, vemos que el ajuste no es bueno 
# (mientras más cercano a 1, mejor ajuste). Estaremos de acuerdo en 
# que una curva se ajustaría mejor. Si queremos ajustar una curva a 
# los datos tendremos que trabajar con más dimensiones. 
# Es decir, tendremos que intentar ajustar un polinomio 
# (de segundo grado, por ejemplo). 
# Para ello fijémonos en la ecuación de la recta que hemos ajustado con la regresión lineal. y=a+bx
# 


# In[24]:


poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)
print(x)
print(x_poly)


# In[25]:


model.fit(x_poly, y)
y_pred = model.predict(x_poly)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))


# In[29]:


poly = PolynomialFeatures(degree=8, include_bias=False)
x_poly = poly.fit_transform(x)
 
model.fit(x_poly, y)
y_pred = model.predict(x_poly)
 
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()
 
rmse = np.sqrt(mean_squared_error(y,y_pred))
r2 = r2_score(y,y_pred)
print ('RMSE: ' + str(rmse))
print ('R2: ' + str(r2))

