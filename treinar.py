import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from gerar_dados import gerar

ALVO = "tempo_entrega_min"


def preparar(df):
    # transforma tipo_comida em colunas 0/1
    return pd.get_dummies(df, columns=["tipo_comida"], dtype=int)


def treinar(df):
    dados = preparar(df)
    X = dados.drop(columns=[ALVO])
    y = dados[ALVO]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    modelos = {
        "media (baseline)": DummyRegressor(strategy="mean"),
        "regressao linear": LinearRegression(),
        "random forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient boosting": GradientBoostingRegressor(random_state=42),
    }

    resultados = []
    treinados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        previsto = modelo.predict(X_teste)
        resultados.append({
            "modelo": nome,
            "MAE": round(mean_absolute_error(y_teste, previsto), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_teste, previsto))), 2),
            "R2": round(r2_score(y_teste, previsto), 3),
        })
        treinados[nome] = modelo

    tabela = pd.DataFrame(resultados)
    melhor = tabela.sort_values("MAE").iloc[0]["modelo"]
    return treinados[melhor], melhor, tabela, list(X.columns)


if __name__ == "__main__":
    try:
        df = pd.read_csv("pedidos.csv")
    except FileNotFoundError:
        df = gerar()

    modelo, melhor, tabela, colunas = treinar(df)
    print(tabela.to_string(index=False))
    print("\nmelhor modelo:", melhor)

    if hasattr(modelo, "feature_importances_"):
        imp = pd.Series(modelo.feature_importances_, index=colunas).sort_values(ascending=False)
        print("\nvariaveis mais importantes:")
        print(imp.head(6).round(3))

    joblib.dump({"modelo": modelo, "colunas": colunas}, "modelo.joblib")
    print("\nmodelo salvo em modelo.joblib")
