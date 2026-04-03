# Regulatory Disclaimers

## Important Notice

CreditRiskEngine is an **open-source analytical tool** provided under the
Apache 2.0 license. It is NOT a certified regulatory calculation system.

### Usage Limitations

1. **Not a Substitute for Professional Judgment**: The formulas and parameters
   implemented in this library reflect the developers' interpretation of
   published regulatory texts (BCBS d424, CRR3, PRA PS17/23, etc.). These
   interpretations may differ from those of your national supervisor.

2. **No Regulatory Endorsement**: This software has not been reviewed, approved,
   or endorsed by the Basel Committee, EBA, PRA, Federal Reserve, RBI, or any
   other regulatory authority.

3. **Model Risk**: Users are responsible for their own model validation,
   independent review, and regulatory approval processes. Line coverage in
   automated tests does not constitute model validation evidence.

4. **Jurisdiction-Specific Rules**: Regulatory requirements vary by jurisdiction
   and are subject to change. YAML configuration files represent a snapshot in
   time and may not reflect the latest amendments, guidelines, or supervisory
   expectations.

5. **No Warranty**: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
   KIND. See the Apache 2.0 license for full terms.

### Recommended Use Cases

| Use Case | Suitability |
|----------|------------|
| **Education & Training** | Excellent — learn Basel III mechanics hands-on |
| **Prototyping & Research** | Excellent — rapid iteration on capital models |
| **Benchmarking** | Good — cross-check vendor tools and internal models |
| **Impact Analysis** | Good — assess regulatory change impacts (CRR3, Basel IV) |
| **Production (with validation)** | Possible — requires independent model validation, audit, and supervisory approval |
| **Sole Production System** | Not recommended without thorough validation |

### Regulatory Interpretation Choices

Where the Basel text is ambiguous, we have made the following choices:

| Topic | Our Interpretation | Alternative |
|-------|-------------------|-------------|
| PD floor for all IRB | 3 bps (CRE32.13) | Some jurisdictions apply 5 bps |
| QRRE transactor scalar | 0.75× applied (CRE31.9 fn 15) | Some implementations omit this |
| SME firm-size adjustment floor | S floored at EUR 5M | Some apply floor at EUR 1M |
| Maturity adjustment at 2.5y | MA > 1.0 (formula yields >1) | This is mathematically correct |
| UK PRA loan-splitting threshold | 55% of property value | Subject to PRA clarification |

### Audit Trail

All calculations can be traced through the `AuditTrail` class which records:

- Input parameters (PD, LGD, EAD, maturity, asset class)
- Output values (K, RW, RWA, capital requirement)
- Regulatory reference (Basel paragraph number)
- Engine version and timestamp
- Any warnings or overrides applied

### Management Overlay Governance

Post-model adjustments (management overlays) are tracked via
`OverlayAuditRecord` with immutable lifecycle events:

- Overlay name, type, and classification
- Model ECL before and after overlay
- Rationale and regulatory basis (e.g., IFRS 9.B5.5.52)
- Approval authority and date
- Effective and expiry dates
- Portfolio scope

The `validate_overlay()` function checks governance completeness against
EBA/GL/2020/06 and PRA Dear CFO letter (Jul 2020) expectations, including:
rationale, approval authority, expiry date, portfolio scope, and non-zero
impact.

### Scenario Weight Governance

Scenario probability weights are governed via `ScenarioSetMetadata` which
records approval chain, review cadence, calibration methodology, and data
sources. The `validate_scenario_governance()` function checks compliance
with IFRS 9.B5.5.41-43 and EBA/GL/2017/06 para 74, including minimum
scenario count, weight sum, and review date scheduling.

`scenario_sensitivity_analysis()` quantifies ECL dependence on individual
scenario weights, supporting the "unbiased and probability-weighted"
requirement in IFRS 9.5.5.17.

### Reporting Issues

If you identify a discrepancy between this library's output and the regulatory
text, please file an issue at:
<https://github.com/ankitjha67/baselkit/issues>

Include the specific Basel/CRR3/PRA paragraph number and expected vs. actual values.
