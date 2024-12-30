import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados
anos = np.array([2003, 2007, 2008, 2011, 2012, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
salarios = np.array([640, 1000, 2000, 3000, 3200, 5000, 10000, 12630, 16000, 19000, 24000, 33000, 35640])

# Transformação polinomial
poly = PolynomialFeatures(degree=3)
anos_poly = poly.fit_transform(anos)

# Modelo
modelo = LinearRegression()
modelo.fit(anos_poly, salarios)

# Previsão
anos_pred = np.arange(2003, 2025).reshape(-1, 1)
salarios_pred = modelo.predict(poly.transform(anos_pred))

# Visualização
plt.scatter(anos, salarios, color='blue', label='Dados reais')
plt.plot(anos_pred, salarios_pred, color='red', label='Modelo Polinomial')
plt.xlabel('Ano')
plt.ylabel('Salário (R$)')
plt.legend()
plt.show()
