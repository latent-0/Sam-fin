"""Turns the simulator's allocations for a chosen coalition into a readable plan.

The six agents here are the same mechanics the RL environment trains on, so the
plan the user sees is exactly what the orchestrator scored. This module adds
the conflict-resolution layer, the text and a 12-month projection.
"""
from __future__ import annotations

import numpy as np

from . import env as E

PLAN_DRAWS = 400


def _run(P1: dict, mask: int, steps: int, draws: int = 1, seed: int = 3, mean_path: bool = True):
    """Simulate one household. ``mean_path`` switches off income and return noise."""
    sub = {k: np.repeat(np.asarray(v, float)[:1], draws) for k, v in P1.items()}
    e = E.Env(sub, seed=seed)
    if mean_path:
        e.z = np.zeros(draws)
    rows, first = [], None
    for t in range(steps):
        noise = (np.zeros(draws), np.zeros(draws)) if mean_path else None
        _, _, info = e.step(np.full(draws, mask), noise=noise)
        if first is None:
            first = {k: float(np.mean(v)) for k, v in info["alloc"].items()}
        S = e.S
        rows.append(dict(month=t + 1, reserve=float(S["R"].mean()), debt=float(S["D"].mean()),
                         goal=float(S["Gs"].mean()), invest=float(S["I"].mean()),
                         liquid=float(S["L"].mean()),
                         net_worth=float((S["L"] + S["R"] + S["Gs"] + S["I"] - S["D"]).mean()),
                         shortfall_p=float(info["shortfall"].mean())))
    return first, rows


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def _action_text(agent: str, amt: float, P1: dict, scale: float) -> str:
    c = float(P1["c"][0])
    exp = float((P1["fixed"] + P1["var"] + P1["emi"])[0])
    if agent == "budget":
        return f"Trim discretionary spending by about {_fmt(amt)} a month (12% of flexible costs)."
    if agent == "debt":
        if amt <= 0:
            return "No debt left to pay down; keep paying the regular instalment."
        return f"Put {_fmt(amt)} a month extra toward debt at {P1['rd'][0] * 100:.0f}% interest."
    if agent == "emergency":
        tgt = float(E.reserve_target(c))
        hold = float(P1["R"][0]) / exp
        if amt <= 0:
            return f"Your reserve already meets the target of {tgt:.1f} months of expenses; no transfer needed."
        return (f"Move {_fmt(amt)} a month into the emergency reserve. Target is {tgt:.1f} months of expenses "
                f"({_fmt(tgt * exp * scale)}); you hold {hold:.1f}.")
    if agent == "goal":
        if amt <= 0:
            return "Your goal is already funded; nothing extra to set aside."
        return f"Set aside {_fmt(amt)} a month for your goal."
    if agent == "risk":
        return (f"Take income-shock cover costing about {_fmt(amt)} a month (1.5% of income); it pays half of any "
                f"shortfall below 75% of your average income.")
    if agent == "investment":
        rho = float(P1["rho"][0])
        return (f"Invest about {_fmt(amt)} a month (expected return {(0.04 + 0.08 * rho) * 100:.1f}% a year, "
                f"volatility {(0.04 + 0.16 * rho) * 100:.0f}%).")
    raise KeyError(agent)


def conflicts(names: list, P1: dict) -> list:
    out = []
    cov = float(P1["R"][0] / (E.reserve_target(P1["c"][0]) * (P1["fixed"] + P1["var"] + P1["emi"])[0]))
    if "debt" in names and "goal" in names:
        out.append(dict(between=["debt", "goal"],
                        resolution="Surplus is split in proportion to urgency (interest rate vs goal deadline).",
                        reason="Both draw on the same monthly surplus."))
    if "emergency" in names and "investment" in names and cov < 1:
        out.append(dict(between=["emergency", "investment"],
                        resolution="The reserve is weighted ahead of investing until it reaches its target.",
                        reason="Investing before a reserve exists is unsuitable for volatile income."))
    if "debt" in names and "investment" in names and float(P1["rd"][0]) > 0.10:
        out.append(dict(between=["debt", "investment"],
                        resolution="Debt receives the larger share while its rate exceeds the expected return.",
                        reason="Paying a high rate is a guaranteed return; investing is not."))
    return out


def build_plan(P1: dict, scale: float, mask: int, horizon: int = 12) -> dict:
    names = E.mask_names(mask)
    first, path = _run(P1, mask, horizon)
    _, base = _run(P1, 0, horizon)
    _, mc = _run(P1, mask, horizon, draws=PLAN_DRAWS, mean_path=False)
    _, mc0 = _run(P1, 0, horizon, draws=PLAN_DRAWS, mean_path=False)

    actions = []
    for a in ("budget", "emergency", "debt", "goal", "risk", "investment"):
        if a in names:
            amt = round(first[a] * scale)
            actions.append(dict(agent=a, amount=float(amt), unit="/month", text=_action_text(a, amt, P1, scale)))

    exp = float((P1["fixed"] + P1["var"] + P1["emi"])[0])

    def proj(rows):
        return [dict(month=r["month"], net_worth=float(round(r["net_worth"] * scale)),
                     reserve=float(round(r["reserve"] * scale)), debt=float(round(r["debt"] * scale))) for r in rows]

    with_plan, without = proj(path), proj(base)
    return dict(
        agents=names,
        actions=actions,
        conflicts=conflicts(names, P1),
        reserve_target_months=round(float(E.reserve_target(P1["c"][0])), 1),
        reserve_coverage_months=round(float(P1["R"][0]) / exp, 2),
        projection=dict(with_plan=with_plan, without_plan=without),
        risk=dict(shortfall_month_prob_with_plan=round(float(np.mean([r["shortfall_p"] for r in mc])), 4),
                  shortfall_month_prob_without_plan=round(float(np.mean([r["shortfall_p"] for r in mc0])), 4)),
        net_worth_gain_12m=float(with_plan[-1]["net_worth"] - without[-1]["net_worth"]),
    )
