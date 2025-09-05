"""Streamlit UI for roster optimisation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src import dataio, pricing, services


# Paths (only used for displaying status)
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DIR / "fanta.db"
DERIVED_CSV = PROCESSED_DIR / "derived_prices.csv"

DATA_PROCESSED = f"DB: {DB_PATH} | table: derived_prices"

st.set_page_config(page_title="Fantacalcio - Pricing & Roster", layout="wide")

st.title("Pricing & Roster")

with st.expander("Derived prices trainer", expanded=True):
    overwrite = st.checkbox(
        "Overwrite derived prices (reset table)",
        value=True,
        help="Se attivo, rimpiazza completamente tabella e CSV di output.",
    )
    if st.button("Train derived prices", type="primary"):
        try:
            out = pricing.train_derived_prices(overwrite=overwrite)
            st.success(
                f"Derived prices: training completato. Righe: {out.get('rows', 'n/a')}"
            )
        except Exception as e:  # pragma: no cover - UI feedback
            st.error(f"Failed to train derived prices: {e}")
            st.exception(e)

    st.caption(f"pricing loaded: {pricing.__file__}")
    st.caption(DATA_PROCESSED)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Debug derived_prices schema"):
            try:
                schema = pricing.debug_derived_prices_schema()
                st.code(schema, language="sql")
            except Exception as e:  # pragma: no cover - UI feedback
                st.error(f"Schema debug failed: {e}")
    with c2:
        if st.button("Reset derived_prices table (drop + drop indexes)"):
            try:
                pricing.reset_derived_prices_table()
                st.success("Tabella derived_prices resettata.")
            except Exception as e:  # pragma: no cover - UI feedback
                st.error(f"Reset failed: {e}")

    # Show only true INPUT requirements (do NOT require derived_prices.csv here)
    missing_inputs = pricing.list_missing_required_inputs()
    if missing_inputs:
        st.warning("Missing required data files (input):")
        for m in missing_inputs:
            st.markdown(f"- `{m}`")
    else:
        st.info(f"All required inputs present. Output CSV atteso: `{DERIVED_CSV}`")
