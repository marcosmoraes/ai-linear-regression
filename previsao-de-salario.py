from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Criando dados de exemplo
X = np.array([[1], [2], [3], [4], [5]])  # Anos de experiência (váriavel dependente)
y = np.array([5000, 7000, 9000, 11000, 13000])  # Salários (váriavel independente)

# Dividindo em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criando e treinando o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazendo previsões
previsao = modelo.predict([[6]])  # Prevendo salário para 6 anos de experiência
print(f"Salário previsto: R${previsao[0]:.2f}")