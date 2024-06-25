
"""
INSTALAÇÃO PYCARET
"""

!pip install pycaret

"""IMPORTAÇÃO DE BIBLIOTECAS"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pycaret.classification import *

"""CARREGAMENTO DO CONJUNTO DE DADOS"""

url = 'csgo_data/csgo_round_snapshots.xlsx'
csgo_data = pd.read_excel(url)

"""DEFINIÇÃO DA VARIÁVEL ALVO"""

target_variable = 'round_winner'

"""REMOVENDO A COLUNA MAP"""

csgo_data.drop(['map'], axis=1, inplace=True)

"""SEPARAÇÃO DE FEATURES E TAGS"""

X = csgo_data.drop(target_variable, axis=1)
y = csgo_data[target_variable]

"""DIVISÃO DO CONJUNTO DE DADOS"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

"""DEFINIÇÃO DE TRANSFORMAÇÕES PARA DADOS NUMÉRICOS E CATEGÓRICOS"""

numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

# Aplicar transformações
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

"""TRANSFORMANDO DADOS DE TREINO E TESTE"""

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

"""CONFIGURANDO O AMBIENTE DO PYCARET"""

clf1 = setup(data=pd.concat([X_train, y_train], axis=1), target=target_variable, session_id=42)

"""COMPARANDO MODELOS E AVALIANDO DESEMPENHO"""

best_model = compare_models()

"""AVALIANDO O MODELO NO CONJUNTO DE TESTE"""

evaluate_model(best_model)

"""FAZENDO PREVISÕES NO CONJUNTO DE TESTE"""

predictions = predict_model(best_model, data=X_test)

"""IDENTIFICANDO A COLUNA DE PREVISÕES DINAMICAMENTE"""

prediction_column = predictions.columns[predictions.columns.str.contains('Label', case=False)].tolist()
if not prediction_column:
    raise ValueError("Nenhuma coluna de previsão encontrada. Verifique a estrutura da saída do PyCaret.")
else:
    prediction_column = prediction_column[0]

"""AVALIANDO PREVISÕES DE MELHOR TREINAMENTO"""

rf_accuracy_pycaret = accuracy_score(y_test, predictions[prediction_column])
print(f"Melhor Modelo (PyCaret) Accuracy: {rf_accuracy_pycaret:.2f}")
print("Melhor Modelo (PyCaret) Classification Report:")
print(classification_report(y_test, predictions[prediction_column]))

"""=======================================ETAPA 2=======================================

IMPORTAÇÃO DE BIBLIOTECAS
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from pycaret.classification import *

"""CARREGAMENTO DO CONJUNTO DE DADOS"""

url = 'csgo_data/csgo_round_snapshots.xlsx'
csgo_data = pd.read_excel(url)

"""DIVISÃO DO CONJUNTO DE DADOS"""

X_train, X_test, y_train, y_test = train_test_split(
    csgo_data.drop('round_winner', axis=1),
    csgo_data['round_winner'],
    test_size=0.25,
    random_state=42
)

"""CONFIGURANDO O AMBIENTE DO PYCARET"""

clf1 = setup(data=pd.concat([X_train, y_train], axis=1), target='round_winner', session_id=42)

"""COMPARANDO MODELOS E AVALIANDO DESEMPENHO"""

best_model = compare_models()

"""AVALIANDO MODELO NO CONJUNTO DE TESTES"""

evaluate_model(best_model)

"""FAZER PREVISÕES NO CONJUNTO DE TESTES"""

predictions = predict_model(best_model, data=X_test)

"""IDENTIFICANDO A COLUNA DE PREVISÃO DE MANEIRA DINAMICA"""

prediction_column = predictions.columns[predictions.columns.str.contains('Label', case=False)].tolist()
if not prediction_column:
    raise ValueError("Nenhuma coluna de previsão encontrada. Verifique a estrutura da saída do PyCaret.")
else:
    prediction_column = prediction_column[0]

"""AVALIANDO A PREVISÃO"""

rf_accuracy_pycaret = accuracy_score(y_test, predictions[prediction_column])
print(f"Melhor Modelo (PyCaret) Accuracy: {rf_accuracy_pycaret:.2f}")
print("Melhor Modelo (PyCaret) Classification Report:")
print(classification_report(y_test, predictions[prediction_column]))

"""CRIANDO PIPELINE"""

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

"""TREINANDO PIPELINE NO CONJUNTO DE TREINO"""

pipeline.fit(X_train, y_train)

"""FAZENDO PREVISÕES NO CONJUNTO DE TESTES"""

predictions_pipeline = pipeline.predict(X_test)

"""MOSTRANDO PREVISÕES"""

print("Predições do Pipeline:")
print(predictions_pipeline)

"""CALCULANDO E EXIBINDO MÉTRICAS"""

from sklearn.metrics import accuracy_score, classification_report

# Calcular e exibir métricas
accuracy = accuracy_score(y_test, predictions_pipeline)
classification_report_result = classification_report(y_test, predictions_pipeline)

print(f"Acurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(classification_report_result)
