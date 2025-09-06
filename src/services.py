"""Roster optimisation services."""

import numpy as np
import pandas as pd


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
    if df.empty:
        logger.error("No players provided.")
        return df

    if "status" in df.columns:
        df = df[df["status"].fillna("AVAILABLE") == "AVAILABLE"].copy()

    required = ["role", "team", "price_500", "score_z_role"]
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()

    df["price_500"] = pd.to_numeric(df["price_500"], errors="coerce").fillna(0)
    df = df[df["price_500"] >= 0].copy()

    # Locked players already acquired
    if "my_acquired" in df.columns:
        locked = df[df["my_acquired"] == 1].copy()
        locked["locked"] = True
        locked["eff_price"] = pd.to_numeric(
            locked.get("my_price", locked["price_500"]), errors="coerce"
        ).fillna(locked["price_500"])
    else:
        locked = df.iloc[0:0].copy()
        locked["locked"] = False
        locked["eff_price"] = []

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

    # utilità: media fra z_role (primario) e costo (secondario dolce)
    eta = 0.05  # penalizzazione prezzo molto lieve: regola se vuoi più parsimonia
    def utility(row) -> float:
        return float(row["score_z_role"]) - eta * (float(row["eff_price"]) / max(1.0, float(budget_total)))

    # ciclo finché non completiamo TUTTI i ruoli richiesti
    safety = 10000
    while sum(need.values()) > 0 and safety > 0:
        safety -= 1
        # (a) costruiamo la lista dei candidati ammissibili per i ruoli ancora da riempire
        pool["__can_role__"] = pool["role"].map(lambda r: need.get(r, 0) > 0)
        cand = pool[pool["__can_role__"]].copy()
        if cand.empty:
            logger.error("Infeasible: no candidates left to fill remaining roles.")
            roster = pd.concat([locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False)
            roster["budget_total"] = float(budget_total)
            roster["budget_locked"] = float(budget_locked)
            roster["budget_left"] = float(budget_left)
            return roster

        # (b) calcolo utilità e pre-filtro su budget con riserva
        cand["__u__"] = cand.apply(utility, axis=1)
        cand = cand.sort_values(["__u__", "score_z_role"], ascending=[False, False])

        picked = False
        for idx, row in cand.iterrows():
            r = row["role"]
            price = float(row["eff_price"])
            t = row["team"]
            # riserva: quanto devo tenere per completare il resto (escluso il ruolo r di questo pick)
            reserve = reserve_budget(need, excluding_role=r)
            if best_effort:
                reserve = 0.0
            # ammissibilità: budget dopo il pick deve restare >= riserva, e team_cap rispettato
            if (price <= (budget_left - reserve) + 1e-9) and (team_count.get(t, 0) < team_cap):
                # prendo
                selected_rows.append(row)
                budget_left -= price
                team_count[t] = team_count.get(t, 0) + 1
                need[r] = int(need.get(r, 0)) - 1
                pool = pool.drop(index=[idx])
                picked = True
                break
        if not picked:
            # non ho trovato nessun candidato che rispetti contemporaneamente budget+riserva e team_cap
            logger.error("Infeasible under team_cap/budget with remaining needs: " + str(need))
            roster = pd.concat([locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False)
            roster["budget_total"] = float(budget_total)
            roster["budget_locked"] = float(budget_locked)
            roster["budget_left"] = float(budget_left)
            return roster

    # se arrivo qui ho coperto tutti i ruoli richiesti
    logger.info("All roles filled exactly with global selection.")

    roster = pd.concat([locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False)
    if roster.empty:
        logger.error("No feasible roster built. Returning empty DataFrame.")
        return roster

    def upgrade_loop(roster_df: pd.DataFrame, pool_df: pd.DataFrame, budget_left: float, team_cap: int):
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
                better = pool_df[(pool_df["role"] == r) & (pool_df["score_z_role"] > worst["score_z_role"])]
                if better.empty:
                    continue
                for bidx, cand in better.sort_values(["score_z_role", "price_500"], ascending=[False, False]).iterrows():
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
                                team_counts_local[t_old] = max(0, team_counts_local.get(t_old, 0) - 1)
                            pool_df = pool_df.drop(index=[bidx]).append(worst, ignore_index=True)
                            improved = True
                            break
        return roster_df, pool_df, budget_left

    roster, pool, budget_left = upgrade_loop(roster, pool, budget_left, team_cap)

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

