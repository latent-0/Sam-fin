"""Model feature names, kept free of heavy imports so the serverless runtime can load them."""

CORE = ["age", "debt_ratio", "utilization", "d30", "d60", "d90", "open_lines", "dependents"]
VOL = ["income_cv", "semi_dev_ratio", "history_months", "buffer_months", "emi_ratio", "gig", "expense_ratio"]
FEATS = CORE + VOL
