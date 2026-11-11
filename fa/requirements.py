"""Requirement-by-requirement readiness checklist (paper Sec. III-B).

Every criterion is illustrative, not a lender policy. Anything for which no
data was entered is reported as "unknown" (cannot assess) rather than inferred.
"""
from __future__ import annotations

SOURCE = "Illustrative project criterion - not a lender policy."


def _fmt(v: float) -> str:
    return f"{v:,.0f}"


def _item(key, title, status, detail, value, comparison, action=None):
    return dict(key=key, title=title, status=status, detail=detail, value=value,
                comparison=comparison, action=action, source=SOURCE)


def checklist(history, expenses: float, existing_emi: float, new_emi: float, savings: float,
              stab: dict, ruin: float, safe_emi: float, req_buffer: float,
              missed_payments=None, alpha: float = 0.10) -> list:
    n, mean, cv, lo = stab["n"], stab["mean"], stab["cv"], stab["min"]
    out = []

    # 1. history
    if n >= 6:
        st, det = "met", "Six or more recorded months give a usable estimate of variation."
    elif n >= 3:
        st, det = "met", "Recorded months show actual variation; three is a minimum useful sample."
    else:
        st, det = "not_met", "Fewer than three months cannot show how income varies."
    out.append(_item("history", "Income history", st, det, f"{n} recorded months", "3+ months (6+ is better)",
                     None if n >= 6 else "Keep recording income each month; 6 months sharply reduces model uncertainty."))

    # 2. stability
    st = "met" if cv < 0.20 else "needs_improvement" if cv < 0.40 else "not_met"
    out.append(_item("stability", "Income stability", st,
                     "Variation is standard deviation divided by average income.",
                     f"{cv * 100:.1f}% variation (stability index {stab['isi']})", "Below 20% variation",
                     None if st == "met" else
                     "Build a buffer for lower-income months and avoid fixed payments sized on your best months."))

    # 3. repayment capacity in the lowest month
    margin = lo - expenses - new_emi
    out.append(_item("repayment", "Repayment capacity (lowest month)", "met" if margin >= 0 else "not_met",
                     "Lowest recorded income minus current expenses (existing EMI included) minus the new EMI.",
                     _fmt(margin), "0 or more",
                     None if margin >= 0 else f"Reduce the EMI toward {_fmt(safe_emi)} or raise your low-month income."))

    # 4. cash-flow stability
    short = sum(1 for x in history if x < expenses + new_emi)
    st = "met" if short == 0 else "not_met" if short == n else "needs_improvement"
    out.append(_item("cashflow", "Cash-flow stability", st,
                     "Each recorded month is tested against total spending plus the new EMI.",
                     f"{short} of {n} months short", "No shortfall month",
                     None if short == 0 else
                     "Review the shortfall months and build a buffer before adding a fixed payment."))

    # 5. emergency reserve
    months = savings / expenses if expenses > 0 else 0.0
    st = "met" if months >= 3 else "needs_improvement" if months >= 1 else "not_met"
    gap = max(3 * expenses - savings, 0.0)
    out.append(_item("reserve", "Emergency reserve", st, "Savings divided by current monthly expenses.",
                     f"{months:.2f} months", "3 months of expenses",
                     None if st == "met" else f"Add about {_fmt(round(gap, -3))} to reach three months."))

    # 6. existing debt payments
    dti = existing_emi / mean if mean > 0 else 0.0
    st = "met" if dti < 0.20 else "needs_improvement" if dti < 0.40 else "not_met"
    out.append(_item("debt", "Existing debt payments", st,
                     "Existing EMI is already part of expenses and is not subtracted twice.",
                     f"{dti * 100:.1f}% of average income", "Below 20%",
                     None if st == "met" else "Pay down existing loans before adding a new instalment."))

    # 7. ruin probability
    st = "met" if ruin <= alpha else "not_met"
    out.append(_item("ruin", "Ruin probability (12 months, simulated)", st,
                     "Chance that cumulative cash goes below zero within a year, from the fitted income model.",
                     f"{ruin * 100:.1f}%", f"At most {alpha * 100:.0f}%",
                     None if st == "met" else
                     f"Largest EMI meeting this test: {_fmt(safe_emi)}; or hold a buffer near {_fmt(req_buffer)}."))

    # 8. repayment consistency
    if missed_payments is None:
        out.append(_item("consistency", "Recorded repayment consistency", "unknown", "No repayment history entered.",
                         "Not provided", "No missed payments", "Add any missed payments so the check is complete."))
    else:
        mp = int(missed_payments)
        out.append(_item("consistency", "Recorded repayment consistency", "met" if mp == 0 else "not_met",
                         "Missed payments entered by the user.", f"{mp} missed payments", "No missed payments",
                         None if mp == 0 else "Keep every payment on time for at least 6 months before applying."))

    # 9. credit information: never generated or inferred
    out.append(_item("credit_info", "Credit information", "unknown", "No bureau score is generated or inferred.",
                     "Not provided / cannot assess", "Information to review with a lender"))
    return out
