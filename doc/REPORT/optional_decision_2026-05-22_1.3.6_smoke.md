# Optional Decision 1.3.6 Smoke

```yaml
date: 2026-05-22
baseline_stage: "1.3.5"
target_model: "w6"
missing_metric: "not_measured_yet"
current_value: null
target_value: null
remaining_gflops_budget_percent: 5.0
requested_experiment: "psa_p5_smoke_or_gelan_neck_smoke"
expected_gain: "validate optional gate and model-build path only"
stop_condition: "any route/channel/build failure or GFLOPs increase >= 10%"
status: "smoke_only_not_promoted"
```

This report is a gate document for 1.3.6 code-level smoke validation. It does not promote any optional experiment to a default path.
