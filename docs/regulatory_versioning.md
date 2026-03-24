# Regulatory Configuration Versioning

## Strategy

Regulatory rules change frequently. CreditRiskEngine uses a layered strategy
to manage configuration changes without breaking existing calculations.

### YAML Config Versioning

Each jurisdiction YAML file includes metadata:

```yaml
jurisdiction: EU
framework: CRR3 / CRD6
effective_date: "2025-01-01"
document_reference: Regulation (EU) 2024/1623
```

### Change Management Process

1. **New Regulation**: Create a new YAML file (e.g., `crr3_v2.yml`) or update
   the existing one with the new `effective_date`.

2. **Amendments**: Add amendment metadata to the YAML:
   ```yaml
   amendments:
     - date: "2025-06-15"
       reference: "EBA/GL/2025/XX"
       description: "Updated SME threshold"
   ```

3. **Backward Compatibility**: The `load_config()` function accepts an optional
   `effective_date` parameter to load the configuration as of a specific date.

### Semantic Versioning for the Library

| Version Bump | Trigger |
|-------------|---------|
| **Patch** (0.1.x) | Bug fixes, documentation, test improvements |
| **Minor** (0.x.0) | New features, new jurisdictions, non-breaking API changes |
| **Major** (x.0.0) | Breaking API changes, fundamental formula corrections |

### Regulatory Config Changelog

Changes to YAML configs are tracked separately from code changes:

| Date | Jurisdiction | Change | Reference |
|------|-------------|--------|-----------|
| 2026-03-22 | All | Initial release with 17 jurisdictions | BCBS d424 |
| 2026-03-24 | EU, UK | Fixed LGD supervisory values (0.20→0.10 for secured) | CRE32.22-24 |
| 2026-03-24 | EU | Fixed PD floor from 5 bps to 3 bps | CRE32.13 |
| 2026-03-24 | EU | Added full LTV-bucket RRE/CRE tables | CRR3 Art. 125-126 |
| 2026-03-24 | UK | Added loan-splitting configuration | PRA PS9/24 |
