import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gerar_dados import gerar
from treinar import treinar


def test_dados_tem_colunas_e_valores_validos():
    df = gerar(500)
    assert len(df) == 500
    assert df["tempo_entrega_min"].min() >= 10
    assert set(df["chuva"]) <= {0, 1}


def test_modelo_melhor_que_baseline():
    df = gerar(2000)
    _, melhor, tabela, _ = treinar(df)
    mae_base = tabela[tabela["modelo"] == "media (baseline)"]["MAE"].iloc[0]
    mae_melhor = tabela[tabela["modelo"] == melhor]["MAE"].iloc[0]
    assert mae_melhor < mae_base * 0.6


def test_chuva_aumenta_tempo():
    from prever import prever
    import joblib, tempfile
    df = gerar(2000)
    modelo, _, _, colunas = treinar(df)
    with tempfile.TemporaryDirectory() as pasta:
        caminho = os.path.join(pasta, "m.joblib")
        joblib.dump({"modelo": modelo, "colunas": colunas}, caminho)
        base = {"distancia_km": 3, "hora": 15, "dia_semana": 2, "chuva": 0, "tipo_comida": "lanche",
                "entregadores_disponiveis": 8, "pedidos_na_fila": 5}
        com_chuva = dict(base, chuva=1)
        assert prever(com_chuva, caminho) > prever(base, caminho)
