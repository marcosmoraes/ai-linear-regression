import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Dados
anos = np.array([2003, 2007, 2008, 2011, 2012, 2015, 2018, 2019, 2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
salarios = np.array([640, 1000, 2000, 3000, 3200, 5000, 10000, 12630, 16000, 19000, 24000, 27000, 30000])

# Transformação polinomial
poly = PolynomialFeatures(degree=3)
anos_poly = poly.fit_transform(anos)

# Modelo
modelo = LinearRegression()
modelo.fit(anos_poly, salarios)

# Previsão
anos_pred = np.arange(2003, 2026).reshape(-1, 1)  # Incluindo 2025
salarios_pred = modelo.predict(poly.transform(anos_pred))

# Previsão para 2025
ano_2025 = np.array([[2025]])
salario_2025 = modelo.predict(poly.transform(ano_2025))

# Visualização
plt.figure(figsize=(10, 6))
plt.scatter(anos, salarios, color='blue', label='Dados reais', s=100, zorder=5)
plt.plot(anos_pred, salarios_pred, color='red', label='Modelo Polinomial', linewidth=2)

# Adicionando os valores de salário ao lado dos pontos
for i, txt in enumerate(salarios):
    plt.text(anos[i][0], salarios[i], f'{txt:.0f}', fontsize=9, ha='right', color='black')

# Adicionando a previsão de 2025 ao gráfico
plt.scatter(2025, salario_2025, color='green', label='Previsão para 2025', s=100, zorder=5)
plt.text(2025, salario_2025, f'{salario_2025[0]:.0f}', fontsize=9, ha='left', color='green')

plt.title('Evolução Salarial ao Longo dos Anos com Previsão para 2025', fontsize=16)
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Salário (R$)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()
