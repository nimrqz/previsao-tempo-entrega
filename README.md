# Previsão de tempo de entrega

Modelo de regressão que estima quantos minutos um pedido de delivery vai levar, usando distância, horário, chuva, tipo de comida, entregadores disponíveis e fila do restaurante.

## Sobre os dados

Não tenho acesso a dados reais de delivery, então **gerei um dataset fictício** (`gerar_dados.py`, 5000 pedidos). As regras que usei para gerar são inventadas por mim (ex.: cada km soma ~3,5 min, chuva soma 5 min, horário de pico soma 6 min). Por isso os resultados abaixo mostram que o pipeline funciona, mas **não dizem como o modelo se sairia com dados reais**.

## Como rodar

```bash
pip install -r requirements.txt
python gerar_dados.py     # cria pedidos.csv
python treinar.py         # treina, compara modelos e salva modelo.joblib
python prever.py          # exemplo de previsão
python -m pytest tests
```

## Resultados (20% dos dados separados para teste)

| Modelo | MAE (min) | RMSE (min) | R2 |
|---|---|---|---|
| Média (baseline) | 10,47 | 13,28 | 0,00 |
| Regressão linear | 4,57 | 5,70 | 0,82 |
| Random forest | 4,75 | 5,94 | 0,80 |
| Gradient boosting | 4,16 | 5,26 | 0,84 |

O gradient boosting foi o melhor, com erro médio de uns 4 minutos contra 10,5 do baseline. A regressão linear ficou perto porque os dados que eu gerei são quase lineares; com dados reais eu esperaria mais diferença entre os modelos.

Variáveis mais importantes: distância (0,52), pedidos na fila (0,15) e tipo de comida pizza (0,11).

## O que eu faria com mais tempo

- Testar com dados reais (ex.: algum dataset de delivery do Kaggle)
- Validação cruzada e ajuste de hiperparâmetros (usei os padrões)
- Prever um intervalo (ex.: "entre 35 e 45 min") em vez de um número só
- Colocar o modelo em uma API
