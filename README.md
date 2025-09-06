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
python -m src.main rank --by score_z_role --role ALL --top 20 --budget 500
```

## Esempio input

`examples/sample_players.csv` con colonne:
`name,team,role,price,goals,assists,mins,pens_scored,pens_missed,yc,rc`.

## Output

I risultati sono salvati in `data/` e `data/outputs/`.

Il database SQLite `data/fanta.db` contiene la tabella `players` ed è creato automaticamente.

La metrica composita per le raccomandazioni è:

```
price_pct_role = rank_pct(price_500 | per ruolo, ascending=True)
score_raw = 0.70 * fanta_avg + 0.30 * price_pct_role
score_z_role = zscore(score_raw | per ruolo)
```

La label Recommendation usa le soglie configurabili `BUY alpha` e `HOLD alpha`
sul relativo `score_z_role`.

## UI

L'app Streamlit offre:

- barra di ricerca con dettagli del giocatore e raccomandazione BUY/AVOID;
- gestione del log d'asta con aggiornamento di budget e roster;
- ottimizzatore del roster completo.

L'app utilizza sempre i prezzi ufficiali `price_500`.

### Roster Optimizer

L'ottimizzatore seleziona 25 giocatori (3P/8D/8C/6A) rispettando il budget
di 500 crediti e il vincolo di massimo 3 giocatori per squadra.
Il roster consigliato è salvato in `data/outputs/recommended_roster.csv`.

### Esportazioni

Il pannello "Il mio roster" mostra i giocatori acquistati e permette di
esportare il roster in `data/outputs/my_roster.csv`.

### Il mio roster

Dal pannello "Log d'asta" è possibile spuntare **Acquired** per segnare
un giocatore come acquistato; il giocatore viene salvato nel database
`data/fanta.db` e appare immediatamente nella sezione "Il mio roster"
senza bisogno di ricaricare manualmente la pagina.
