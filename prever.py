import sys

import joblib
import pandas as pd

TIPOS = ["japonesa", "lanche", "marmita", "pizza", "saudavel"]


def prever(pedido, caminho="modelo.joblib"):
    salvo = joblib.load(caminho)
    linha = pd.DataFrame([pedido])
    # cria as colunas de tipo_comida do mesmo jeito do treino
    for tipo in TIPOS:
        linha[f"tipo_comida_{tipo}"] = int(pedido["tipo_comida"] == tipo)
    linha = linha.drop(columns=["tipo_comida"])
    linha = linha[salvo["colunas"]]
    return float(salvo["modelo"].predict(linha)[0])


if __name__ == "__main__":
    exemplo = {
        "distancia_km": 3.2,
        "hora": 20,
        "dia_semana": 4,
        "chuva": 1,
        "tipo_comida": "pizza",
        "entregadores_disponiveis": 4,
        "pedidos_na_fila": 8,
    }
    if len(sys.argv) > 1:
        print("uso: python prever.py  (edite o dicionario 'exemplo' no arquivo)")
    print("pedido:", exemplo)
    print(f"tempo estimado: {prever(exemplo):.0f} minutos")
