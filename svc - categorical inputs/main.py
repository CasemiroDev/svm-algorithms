import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Leitura da base de dados com categorias de feminino e masculino
df = pd.read_csv('database.csv')

# Inputs e Outputs
X = df.drop('risco',axis=1)
y = df.risco

# Selecionando apenas os elementos categoricos
from sklearn.preprocessing import OneHotEncoder
X_cat = X.select_dtypes(include='object')

# Binarização - OneHotEncoder / normalização dos dados categoricos em 0 e 1
onehot = OneHotEncoder(sparse=False, drop="first")
X_bin = onehot.fit_transform(X_cat)

# Nomralização dos dados numéricos
X_num = X.select_dtypes(exclude='object')
minmax = MinMaxScaler()
X_num = minmax.fit_transform(X_num)

# Junção das colunas agora normalizadas com numpy
X_all = np.append(X_num, X_bin, axis=1)

# Divisão de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=2/3)

# Treinamento do classificador SVM
svc = SVC()
svc.fit(X_train, y_train)

# Resultado de precisão
print(accuracy_score(y_test, svc.predict(X_test)))

