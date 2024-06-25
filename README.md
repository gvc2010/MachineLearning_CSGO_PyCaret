# CS:GO Round Winner Prediction using PyCaret

Este projeto utiliza técnicas de aprendizado de máquina para prever o vencedor de uma rodada de CS:GO (Counter-Strike: Global Offensive) com base em snapshots de dados de rodadas. O objetivo é demonstrar como usar o PyCaret para configurar um ambiente de modelagem automatizada e comparar diferentes modelos de classificação.

## Instalação de Dependências

O projeto utiliza a biblioteca `pycaret`, entre outras, para facilitar a análise e modelagem dos dados. Para instalar o PyCaret, utilize o seguinte comando:

```bash
pip install pycaret
```

## Importação de Bibliotecas

As bibliotecas essenciais para a manipulação de dados, modelagem e avaliação são importadas no início do script:

```bash
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
```
