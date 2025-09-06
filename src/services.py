"""Roster optimisation services."""

import numpy as np
import pandas as pd
import os
from typing import Optional
try:  # pragma: no cover - optional dependency
    import pulp
except Exception:  # pragma: no cover - fallback when pulp is missing
    pulp = None

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/outputs")


def _truthy(s) -> pd.Series:
    """Converte qualunque colonna (0/1, bool, 'true', 'yes', 'Y') in booleano."""
    return pd.Series(s).astype(str).str.strip().str.lower().isin([
        "1",
        "true",
        "yes",
        "y",
        "si",
        "s",
        "x",
        "\u2713",
        "\u2714",
    ])


def attach_my_roster(players: pd.DataFrame, st: Optional[object] = None) -> pd.DataFrame:
    """Arricchisci ``players`` con le colonne ``my_acquired`` e ``my_price``.

    Le informazioni sugli acquisti vengono lette da una delle seguenti sorgenti,
    in ordine di priorità:

    1. Database SQLite tramite :func:`src.db.get_my_roster`.
    2. CSV ``my_roster.csv`` in :data:`OUTPUT_DIR` o in ``data/outputs``.
    3. ``st.session_state.my_roster`` se disponibile.
    4. ``st.session_state.auction_log`` filtrato per ``acquired == true``.

    L'unione avviene sulla colonna ``id`` se presente, altrimenti sul
    combinato ``name`` + ``team``. In assenza di una sorgente valida viene
    aggiunta la colonna ``my_acquired`` a ``0`` e, se assente, ``my_price``
    con ``NaN``.
    """

    df = players.copy()

    # Colonna ``status`` (AVAILABLE/SOLD) dal DB se non presente
    if "status" not in df.columns:
        df["status"] = np.nan
    try:  # pragma: no cover - DB potrebbe non essere disponibile
        from . import db as _db

        if "id" in df.columns:
            status_df = _db.read_players_df()[["id", "is_sold"]]
            status_df["status_db"] = np.where(
                status_df["is_sold"].astype(int) == 1, "SOLD", "AVAILABLE"
            )
            df = df.merge(status_df[["id", "status_db"]], on="id", how="left")
            df["status"] = df.pop("status_db").combine_first(df["status"])
    except Exception:  # pragma: no cover - best effort
        pass
    df["status"] = df["status"].fillna("AVAILABLE")

    roster_df = None

    # 1) database
    try:
        from . import db as _db

        roster_players = _db.get_my_roster()
        if roster_players:
            roster_df = pd.DataFrame(
                {
                    "id": [p.id for p in roster_players],
                    "name": [p.name for p in roster_players],
                    "team": [p.team for p in roster_players],
                    "my_price": [p.my_price for p in roster_players],
                    "acquired": [1] * len(roster_players),
                }
            )
    except Exception:
        roster_df = None

    # 2) CSV su disco
    if roster_df is None:
        for path in [
            os.path.join(OUTPUT_DIR, "my_roster.csv"),
            os.path.join("data", "outputs", "my_roster.csv"),
            os.path.join("data", "my_roster.csv"),
        ]:
            if os.path.exists(path):
                try:
                    roster_df = pd.read_csv(path)
                    break
                except Exception:
                    pass

    # 3) session_state
    if roster_df is None and st is not None and hasattr(st, "session_state"):
        if "my_roster" in st.session_state:
            roster_df = pd.DataFrame(st.session_state["my_roster"])
        elif "auction_log" in st.session_state:
            al = pd.DataFrame(st.session_state["auction_log"])
            if not al.empty and "acquired" in al.columns:
                al = al[_truthy(al["acquired"])]
                roster_df = al

    if roster_df is None or roster_df.empty:
        df["my_acquired"] = 0
        if "my_price" not in df.columns:
            df["my_price"] = np.nan
        return df

    r = roster_df.copy()

    # Normalizza colonne prezzo / acquired
    if "my_price" not in r.columns:
        for c in ["my_price", "sold_price", "paid", "price"]:
            if c in r.columns:
                r = r.rename(columns={c: "my_price"})
                break
    if "acquired" in r.columns:
        r["acquired"] = _truthy(r["acquired"])

    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
    if "player_id" in df.columns and "id" not in df.columns:
        df = df.rename(columns={"player_id": "id"})
        df["id"] = pd.to_numeric(df["id"], errors="coerce")

    if "id" in r.columns:
        r["id"] = pd.to_numeric(r["id"], errors="coerce")
        r_key = ["id"]
    else:
        for col in ["name", "team"]:
            if col not in r.columns or col not in df.columns:
                df["my_acquired"] = 0
                if "my_price" not in df.columns:
                    df["my_price"] = np.nan
                return df
        r["__key__"] = (
            r["name"].astype(str).str.strip().str.lower()
            + "|"
            + r["team"].astype(str).str.strip().str.lower()
        )
        df["__key__"] = (
            df["name"].astype(str).str.strip().str.lower()
            + "|"
            + df["team"].astype(str).str.strip().str.lower()
        )
        r_key = ["__key__"]

    subset_cols = r_key
    if "acquired_at" in r.columns:
        r = r.sort_values("acquired_at").drop_duplicates(subset=subset_cols, keep="last")
    else:
        r = r.drop_duplicates(subset=subset_cols, keep="last")

    cols = r_key + [c for c in ["my_price", "acquired"] if c in r.columns]
    if r_key == ["id"] and "id" in df.columns:
        m = df.merge(r[cols], on="id", how="left", indicator=True)
    else:
        m = df.merge(r[cols], on="__key__", how="left", indicator=True)
        m = m.drop(columns=["__key__"])

    present = m["_merge"].eq("both")
    acquired_flag = _truthy(m["acquired"]) if "acquired" in m.columns else False
    m["my_acquired"] = (present | acquired_flag).astype(int)
    m = m.drop(columns=["_merge"])
    if "my_price" not in m.columns:
        m["my_price"] = np.nan
    return m


# --- logging shim -----------------------------------------------------------
def _mk_logger(log):
    """Return an object exposing ``info``/``warning``/``error`` methods.

    If ``log`` already implements those, it's returned untouched. Otherwise we
    try to route messages to ``streamlit`` if available, falling back to
    printing to stdout.
    """

    class _Shim:
        def __init__(self, sink=None):
            self.sink = sink

        def _emit(self, level, msg):
            try:
                if self.sink is None:
                    print(f"[{level.upper()}] {msg}")
                else:
                    fn = getattr(self.sink, level, None)
                    if callable(fn):
                        fn(msg)
                    else:
                        print(f"[{level.upper()}] {msg}")
            except Exception:  # pragma: no cover - extremely defensive
                print(f"[{level.upper()}] {msg}")

        def info(self, msg):
            self._emit("info", msg)

        def warning(self, msg):
            self._emit("warning", msg)

        def error(self, msg):
            self._emit("error", msg)

    if hasattr(log, "info") and hasattr(log, "warning") and hasattr(log, "error"):
        return log
    try:  # pragma: no cover - streamlit might not be installed during tests
        import streamlit as st

        return _Shim(st)
    except Exception:  # pragma: no cover - best effort fallback
        return _Shim()


def optimize_roster(
    players: pd.DataFrame,
    log,
    budget_total: int,
    team_cap: int,
) -> pd.DataFrame:
    """Greedy roster optimiser with locked-player support and best-effort fill.

    Parameters
    ----------
    players : pd.DataFrame
        Player pool. Required columns: ``role``, ``team``, ``price_500``,
        ``score_z_role``. Optional columns: ``status``, ``my_acquired``,
        ``my_price``.
    log : Any
        Optional logger-like object. If missing the required methods, a simple
        shim will be used so that ``info``, ``warning`` and ``error`` calls are
        still handled.
    budget_total : int
        Total available budget.
    team_cap : int
        Maximum number of players per real team.

    Returns
    -------
    pd.DataFrame
        Selected roster including locked players. Contains extra columns:
        ``locked``, ``eff_price``, ``budget_total``, ``budget_locked`` and
        ``budget_left``.
    """

    roles_need = {"P": 3, "D": 8, "C": 8, "A": 6}

    logger = _mk_logger(log)
    df = players.copy()
    # Deduplicate potential multiple entries for the same player. When the
    # input dataframe contains duplicates (for example coming from joins or
    # data glitches), locked players would be counted multiple times and the
    # optimiser could try to "lock" the same player more than once.
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last")
    elif {"name", "team"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["name", "team"], keep="last")

    if df.empty:
        logger.error("No players provided.")
        return df

    # Disponibili: NON escludere i locked anche se non "AVAILABLE"
    if "my_acquired" in df.columns:
        df["my_acquired"] = pd.to_numeric(df["my_acquired"], errors="coerce").fillna(0).astype(int)
        is_locked = df["my_acquired"] == 1
    else:
        is_locked = pd.Series(False, index=df.index)
    if "status" in df.columns:
        df = df[
            df["status"].fillna("AVAILABLE").eq("AVAILABLE") | is_locked
        ].copy()

    required = ["role", "team", "price_500", "score_z_role"]
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()

    df["price_500"] = pd.to_numeric(df["price_500"], errors="coerce").fillna(0)
    df = df[df["price_500"] >= 0].copy()

    # Locked / già acquistati (ora df include anche eventuali non AVAILABLE ma locked)
    if "my_acquired" in df.columns:
        locked = df[df["my_acquired"] == 1].copy()
        # Ensure each player appears at most once among the locked ones
        if not locked.empty:
            if "id" in locked.columns:
                locked = locked.drop_duplicates(subset=["id"], keep="last")
            elif {"name", "team"}.issubset(locked.columns):
                locked = locked.drop_duplicates(subset=["name", "team"], keep="last")
        locked["locked"] = True
        _mp = pd.to_numeric(locked.get("my_price"), errors="coerce")
        locked["eff_price"] = _mp.where(_mp.notna() & (_mp > 0), locked["price_500"])
    else:
        locked = df.iloc[0:0].copy()
        locked["locked"] = False
        locked["eff_price"] = []
    logger.info(
        f"Locked seen -> total={len(locked)} | by role={locked['role'].value_counts().to_dict() if not locked.empty else {}}"
    )

    budget_locked = locked["eff_price"].sum() if not locked.empty else 0.0
    budget_left = float(budget_total) - float(budget_locked)
    if budget_left < 0:
        logger.warning(
            f"Locked players exceed budget by {-budget_left:.1f}. Proceeding best-effort with zero free budget."
        )
        budget_left = 0.0

    # Remaining needs per role
    need = roles_need.copy()
    if not locked.empty:
        counts = locked["role"].value_counts().to_dict()
        for r, cnt in counts.items():
            if r in need:
                need[r] = max(0, need[r] - int(cnt))

    team_count = locked["team"].value_counts().to_dict() if not locked.empty else {}

    # Candidate pool excluding locked players
    pool = df.copy()
    if "my_acquired" in pool.columns:
        pool = pool[pool["my_acquired"] != 1].copy()
    pool["locked"] = False
    pool["eff_price"] = pool["price_500"]
    pool = pool.sort_values(["role", "score_z_role", "price_500"], ascending=[True, False, False])

    selected_rows = []

    # ----- Selezione GLOBALE senza priorità di ruolo -----
    # prezzo "cheap" per ruolo (stima prudente per la riserva)
    try:
        q35 = pool.groupby("role")["eff_price"].quantile(0.35)
    except Exception:
        q35 = pd.Series(dtype=float)
    min_price = pool.groupby("role")["eff_price"].min()
    cheap_price = {}
    for r in ["P", "D", "C", "A"]:
        v = q35.get(r, np.nan)
        if pd.isna(v):
            v = min_price.get(r, 0.0)
        cheap_price[r] = float(v if pd.notna(v) else 0.0)

    # helper: riserva di budget per coprire i ruoli rimanenti
    def reserve_budget(need_dict: dict, excluding_role: str = None) -> float:
        reserve = 0.0
        for rr, nn in need_dict.items():
            if excluding_role is not None and rr == excluding_role:
                continue
            if nn > 0:
                reserve += cheap_price.get(rr, 0.0) * int(nn)
        return reserve

    # controllo fattibilità di massima: budget minimo teorico (ignorando team_cap)
    min_budget_lb = 0.0
    for r in ["P", "D", "C", "A"]:
        if need.get(r, 0) > 0:
            # somma dei più economici per coprire il ruolo r
            costs = pool.loc[pool["role"] == r, "eff_price"].sort_values().head(int(need[r]))
            if len(costs) < int(need[r]):
                logger.error(f"Infeasible: not enough candidates for role {r}.")
            min_budget_lb += float(costs.sum())
    best_effort = False
    if min_budget_lb > budget_left + 1e-9:
        logger.error(
            f"Infeasible: even cheapest combo needs {min_budget_lb:.1f} > budget_left {budget_left:.1f}."
        )
        best_effort = True

    # ====== ILP exact solver (preferito) ======
    eta = 0.05   # penalizza prezzo (parca)
    gamma = 0.02  # spinge a spendere (molto lieve)

    def _solve_with_ilp(
        pool_df: pd.DataFrame,
        need_dict: dict,
        budget_left_f: float,
        team_count_locked: dict,
        team_cap_i: int,
        budget_tot_f: float,
    ):
        if pulp is None:
            return None, "pulp_not_available"
        cand = pool_df.copy()
        cand = cand[cand["role"].map(lambda r: need_dict.get(r, 0) > 0)]
        if cand.empty:
            return None, "no_candidates"
        idxs = list(cand.index)
        x = pulp.LpVariable.dicts("x", idxs, lowBound=0, upBound=1, cat=pulp.LpBinary)
        prob = pulp.LpProblem("roster_select", pulp.LpMaximize)
        util = {}
        for i in idxs:
            z = float(cand.at[i, "score_z_role"])
            p = float(cand.at[i, "eff_price"]) / max(1.0, budget_tot_f)
            util[i] = z - eta * p + gamma * p
        prob += pulp.lpSum(util[i] * x[i] for i in idxs)
        for r, k in need_dict.items():
            if k > 0:
                prob += pulp.lpSum(x[i] for i in idxs if cand.at[i, "role"] == r) == int(k)
        prob += pulp.lpSum(float(cand.at[i, "eff_price"]) * x[i] for i in idxs) <= float(budget_left_f)
        teams = cand["team"].dropna().unique().tolist()
        for t in teams:
            locked_here = int(team_count_locked.get(t, 0))
            cap_left = team_cap_i - locked_here
            if cap_left < 0:
                return None, f"cap_violation_locked_{t}"
            prob += pulp.lpSum(x[i] for i in idxs if cand.at[i, "team"] == t) <= cap_left
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception as e:  # pragma: no cover - solver failures are rare
            return None, f"solver_error:{e}"
        if pulp.LpStatus[prob.status] != "Optimal":
            return None, f"status_{pulp.LpStatus[prob.status]}"
        chosen = [i for i in idxs if pulp.value(x[i]) > 0.5]
        return cand.loc[chosen].copy(), "ok"

    ilp_sel, ilp_status = (None, "skipped")
    if not best_effort:
        ilp_sel, ilp_status = _solve_with_ilp(
            pool, need, budget_left, team_count, team_cap, budget_total
        )

    if ilp_sel is None:
        if not best_effort:
            logger.error(f"ILP infeasible/fallback ({ilp_status}). Using greedy heuristic.")

        eta = 0.05  # penalizzazione prezzo molto lieve: regola se vuoi più parsimonia

        def utility(row) -> float:
            return float(row["score_z_role"]) - eta * (
                float(row["eff_price"]) / max(1.0, float(budget_total))
            )

        safety = 10000
        while sum(need.values()) > 0 and safety > 0:
            safety -= 1
            pool["__can_role__"] = pool["role"].map(lambda r: need.get(r, 0) > 0)
            cand = pool[pool["__can_role__"]].copy()
            if cand.empty:
                logger.error("Infeasible: no candidates left to fill remaining roles.")
                roster = pd.concat([locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False)
                roster["budget_total"] = float(budget_total)
                roster["budget_locked"] = float(budget_locked)
                roster["budget_left"] = float(budget_left)
                return roster
            cand["__u__"] = cand.apply(utility, axis=1)
            cand = cand.sort_values(["__u__", "score_z_role"], ascending=[False, False])
            picked = False
            for idx, row in cand.iterrows():
                r = row["role"]
                price = float(row["eff_price"])
                t = row["team"]
                reserve = 0.0 if best_effort else reserve_budget(need, excluding_role=r)
                if (price <= (budget_left - reserve) + 1e-9) and (
                    team_count.get(t, 0) < team_cap
                ):
                    selected_rows.append(row)
                    budget_left -= price
                    team_count[t] = team_count.get(t, 0) + 1
                    need[r] = int(need.get(r, 0)) - 1
                    pool = pool.drop(index=[idx])
                    picked = True
                    break
            if not picked:
                logger.error(
                    "Infeasible under team_cap/budget with remaining needs: " + str(need)
                )
                roster = pd.concat(
                    [locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False
                )
                roster["budget_total"] = float(budget_total)
                roster["budget_locked"] = float(budget_locked)
                roster["budget_left"] = float(budget_left)
                return roster

        logger.info("All roles filled exactly with global selection.")
        roster = pd.concat([locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False)
        if roster.empty:
            logger.error("No feasible roster built. Returning empty DataFrame.")
            return roster

        def upgrade_loop(
            roster_df: pd.DataFrame, pool_df: pd.DataFrame, budget_left: float, team_cap: int
        ):
            improved = True
            while improved:
                improved = False
                team_counts_local = roster_df["team"].value_counts().to_dict()
                for r in ["P", "D", "C", "A"]:
                    current = roster_df[(roster_df["role"] == r) & (roster_df["locked"] != True)]
                    if current.empty:
                        continue
                    worst_idx = current["score_z_role"].astype(float).idxmin()
                    worst = roster_df.loc[worst_idx]
                    better = pool_df[
                        (pool_df["role"] == r) & (pool_df["score_z_role"] > worst["score_z_role"])
                    ]
                    if better.empty:
                        continue
                    for bidx, cand in better.sort_values(
                        ["score_z_role", "price_500"], ascending=[False, False]
                    ).iterrows():
                        delta = float(cand["eff_price"]) - float(worst["eff_price"])
                        if delta <= budget_left:
                            t_old, t_new = worst["team"], cand["team"]
                            ok_team = True
                            if t_new != t_old:
                                ok_team = team_counts_local.get(t_new, 0) + 1 <= team_cap
                            if ok_team:
                                budget_left -= max(0.0, delta)
                                roster_df.loc[worst_idx] = cand
                                if t_new != t_old:
                                    team_counts_local[t_new] = team_counts_local.get(t_new, 0) + 1
                                    team_counts_local[t_old] = max(
                                        0, team_counts_local.get(t_old, 0) - 1
                                    )
                                pool_df = pool_df.drop(index=[bidx]).append(
                                    worst, ignore_index=True
                                )
                                improved = True
                                break
            return roster_df, pool_df, budget_left

        roster, pool, budget_left = upgrade_loop(roster, pool, budget_left, team_cap)
    else:
        roster = pd.concat([locked, ilp_sel], ignore_index=True, sort=False)
        spent_ilp = float(ilp_sel["eff_price"].sum())
        budget_left = max(0.0, float(budget_left) - spent_ilp)
        logger.info("All roles filled exactly with ILP.")

    roster = roster.sort_values(["role", "score_z_role", "price_500"], ascending=[True, False, False]).reset_index(drop=True)

    # normalizza colonne prezzo effettivo
    if "eff_price" not in roster.columns:
        roster["eff_price"] = roster["price_500"]
    roster["eff_price"] = pd.to_numeric(roster["eff_price"], errors="coerce").fillna(0.0)

    # cumulato di spesa riga-per-riga per la UI
    roster["spent"] = roster["eff_price"].astype(float)
    roster["cum_budget"] = roster["spent"].cumsum()

    # metadati di budget (ripetuti su ogni riga per semplicità di export)
    roster["budget_total"] = float(budget_total)
    roster["budget_locked"] = float(budget_locked)
    roster["budget_left"] = float(budget_left)
    return roster


__all__ = ["optimize_roster"]

