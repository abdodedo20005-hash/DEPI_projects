"""
Project paths, computed from the location of this file.

Because of this, the app, the scripts and the tests find data/, policies/ and invoices/
no matter which folder they are started from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
POLICY_DIR = PROJECT_ROOT / "policies"
INVOICE_DIR = PROJECT_ROOT / "invoices"
RESULTS_DIR = PROJECT_ROOT / "eval_results"
