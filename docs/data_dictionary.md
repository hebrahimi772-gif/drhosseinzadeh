# Data Dictionary Summary

This summary provides a reviewer-friendly overview of key variables in the dataset. The complete variable list and detailed descriptions are available in `data/metadata/codebook_ocmcap_v1.csv`.

| Variable | Level | Type | Range / Notes |
|----------|-------|------|---------------|
| `id` | individual | numeric | 1–230 (sequential identifier; pseudonymous) |
| `org_id_anon` | organization | integer | 1–15 (anonymized organization identifier; used for grouping) |
| `gender` | individual | numeric | 1–2 (coded) |
| `marital` | individual | numeric | 1–2 (coded) |
| `age` | individual | numeric | 1–4 (age categories) |
| `education` | individual | numeric | 1–4 (education levels) |

Survey items `quest1` through `quest80` are individual-level Likert responses (1–5) assessing various dimensions of change management constructs. Missing values are coded as system-missing/NA.

Organizational composite scores (computed from subsets of the survey items) include:

- `org_climate_score` (numeric)
- `org_support_score` (numeric)
- `org_context_score` (numeric)
- `org_resources_score` (numeric)

The primary target variable is:

- `change_mgmt_capability_score` (numeric) – a composite measure of an organization’s change management capability, computed according to the manuscript’s protocol.

Additional derived predictors (`x1`–`x14`) and psychometric subscales (`p1`–`p14`) are included; their definitions are detailed in the full codebook.

Missing values for numeric fields are represented as blank entries or `NA` in the CSV. Users should refer to the codebook for variable-specific missingness coding and valid ranges.