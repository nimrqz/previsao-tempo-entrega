# Gera um dataset ficticio de pedidos de delivery (nao tenho acesso a dados reais do iFood).
# As regras abaixo sao inventadas por mim para simular o que influencia o tempo de entrega.
import numpy as np
import pandas as pd


def gerar(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    distancia = np.clip(rng.gamma(2.0, 1.8, n), 0.3, 15)  # km
    hora = rng.choice(range(10, 24), n, p=np.array([2, 5, 9, 8, 3, 2, 2, 3, 6, 9, 9, 5, 3, 2]) / 68)
    dia_semana = rng.integers(0, 7, n)  # 0 = segunda
    chuva = (rng.random(n) < 0.2).astype(int)
    tipo_comida = rng.choice(["lanche", "pizza", "japonesa", "saudavel", "marmita"], n)
    entregadores = rng.integers(1, 16, n)
    fila = rng.integers(0, 21, n)  # pedidos na fila do restaurante

    preparo_base = {"lanche": 12, "pizza": 25, "japonesa": 22, "saudavel": 14, "marmita": 10}
    preparo = np.array([preparo_base[t] for t in tipo_comida]) + rng.normal(0, 3, n)

    pico = ((hora >= 12) & (hora <= 13)) | ((hora >= 19) & (hora <= 21))
    tempo = (
        preparo
        + distancia * 3.5          # ~17 km/h de moto no transito
        + chuva * 5
        + pico * 6
        + fila * 0.8
        - entregadores * 0.6
        + (dia_semana >= 5) * 2    # fim de semana
        + rng.normal(0, 4, n)
    )
    tempo = np.clip(tempo, 10, None)

    return pd.DataFrame({
        "distancia_km": distancia.round(2),
        "hora": hora,
        "dia_semana": dia_semana,
        "chuva": chuva,
        "tipo_comida": tipo_comida,
        "entregadores_disponiveis": entregadores,
        "pedidos_na_fila": fila,
        "tempo_entrega_min": tempo.round(1),
    })


if __name__ == "__main__":
    df = gerar()
    df.to_csv("pedidos.csv", index=False)
    print("pedidos.csv gerado com", len(df), "linhas")
