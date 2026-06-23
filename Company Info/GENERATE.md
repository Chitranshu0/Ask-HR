# Generate HR Knowledge Base

Run the synthetic data generator to produce CSV datasets and markdown HR documents.

Prerequisites:

- Python 3.11+
- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Generate files:

```powershell
python scripts/generate_hr_kb.py
```

Outputs:

- `data/` : CSV files (employees.csv, projects.csv, attendance.csv, etc.)
- `docs/` : HR markdown documents (policies, handbook, FAQs)
