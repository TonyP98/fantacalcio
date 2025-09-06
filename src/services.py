"""Roster optimisation services."""

import numpy as np
import pandas as pd


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
    log : Logger-like
        Object implementing ``warning`` and ``error`` methods.
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

    df = players.copy()
    if df.empty:
        log.error("No players provided.")
        return df

    if "status" in df.columns:
        df = df[df["status"].fillna("AVAILABLE") == "AVAILABLE"].copy()

    required = ["role", "team", "price_500", "score_z_role"]
    for col in required:
        if col not in df.columns:
            log.error(f"Missing required column: {col}")
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
        log.warning(
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

    for r in ["P", "D", "C", "A"]:
        n = int(need.get(r, 0))
        if n <= 0:
            continue
        cand_r = pool[pool["role"] == r].copy()
        for idx, row in cand_r.iterrows():
            if n <= 0:
                break
            price = float(row["eff_price"])
            t = row["team"]
            if price <= budget_left and team_count.get(t, 0) < team_cap:
                selected_rows.append(row)
                budget_left -= price
                team_count[t] = team_count.get(t, 0) + 1
                n -= 1
                pool = pool.drop(index=[idx])
        if n > 0:
            log.warning(f"[Best-effort] Role {r}: missing {n} due to budget/team_cap/data.")

    roster = pd.concat([locked, pd.DataFrame(selected_rows)], ignore_index=True, sort=False)
    if roster.empty:
        log.error("No feasible roster built. Returning empty DataFrame.")
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
    roster["budget_total"] = float(budget_total)
    roster["budget_locked"] = float(budget_locked)
    roster["budget_left"] = float(budget_left)
    return roster


__all__ = ["optimize_roster"]

