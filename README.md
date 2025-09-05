# fantacalcio

Strumento locale per analisi prezzi e performance dei calciatori in fase d'asta.

## Setup rapido

Aggiornare `pip`, `setuptools` e `wheel` prima di installare le dipendenze:

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
python -m src.main rank --by value --role ALL --top 20 --budget 500
```

## Esempio input

`examples/sample_players.csv` con colonne:
`name,team,role,price,goals,assists,mins,pens_scored,pens_missed,yc,rc`.

## Output

I risultati sono salvati in `data/` e `data/outputs/`.

L'applicazione utilizza direttamente il prezzo ``price_500`` fornito dalle
quotazioni ufficiali FVM, senza calcolare prezzi "stimati".

## UI

L'app Streamlit offre:

- barra di ricerca con dettagli del giocatore e raccomandazione BUY/AVOID;
- gestione del log d'asta con aggiornamento di budget e roster;
- ottimizzatore del roster completo.

### Strategie di prezzo

È disponibile una sola strategia prezzi: **Prezzo FVM 500 (`price_500`)**.
Il punteggio di valore è `expected_points / effective_price` dove
`effective_price` coincide con `price_500`.

### Roster Optimizer

L'ottimizzatore seleziona 25 giocatori (3P/8D/8C/6A) rispettando il budget
di 500 crediti e il vincolo di massimo 3 giocatori per squadra.
Il roster consigliato è salvato in `data/outputs/recommended_roster.csv`.

### Esportazioni

Il pannello "Il mio roster" mostra i giocatori acquistati e permette di
esportare il roster in `data/outputs/my_roster.csv`.
