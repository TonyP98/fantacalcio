# fantacalcio

Strumento locale per analisi prezzi e performance dei calciatori in fase d'asta.

## Setup rapido

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Comandi principali

```bash
python -m src.main ingest --input examples/sample_players.csv
python -m src.main build-features
python -m src.main price --method baseline
python -m src.main train-prices --method linear
python -m src.main rank --by value --role ALL --top 20 --budget 500
```

## Esempio input

`examples/sample_players.csv` con colonne:
`name,team,role,price,goals,assists,mins,pens_scored,pens_missed,yc,rc`.

## Output

I risultati sono salvati in `data/` e `data/outputs/`.

Il comando `train-prices` genera prezzi stimati dei giocatori
utilizzando le statistiche storiche pesate e le quotazioni ufficiali.
I risultati vengono salvati in `data/processed/derived_prices.csv` e
un riepilogo del modello è disponibile in
`data/outputs/price_model_summary.txt`.

File di input richiesti per il training:

- `quotes_2025_26_FVM_budget500.csv`
- `stats_master_with_weights.csv`
- `goalkeepers_grid_matrix_square.csv`
