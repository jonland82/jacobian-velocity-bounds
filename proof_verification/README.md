# Proof Verification

This folder contains symbolic and numerical verification scripts for the manuscript's main theorem chain and the cached synthetic experiment summaries.

## Generate the report

```powershell
python proof_verification/generate_report.py
```

The command writes:

- `proof_verification/verification_results.json`
- `proof_verification/verification_report.html`

The HTML report reuses the visual styling from the repo's `index.html`.
