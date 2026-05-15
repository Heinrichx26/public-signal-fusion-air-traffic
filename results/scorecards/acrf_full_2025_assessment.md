# ACRF smoke-test assessment

Scope: airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO; 2025 months 1,2,3,4,5,6,7,8,9,10,11,12.

Smoke gate: learned-eta Full ACRF should beat D-S and Yager on long-delay AUC, keep positive post-window residual belief, and avoid a Brier loss larger than 0.002 versus the calendar-weather-demand baseline.

Assessment: usable for full-year expansion.

- Learned-eta Full ACRF long-delay: AUC 0.707 (+0.036 vs baseline), PR-AUC 0.237, Brier gain +0.0035, top-decile lift 3.34; gap to early fusion AUC -0.014.
- Learned-eta Full ACRF cancellation: AUC 0.713 (+0.043 vs baseline), PR-AUC 0.069, Brier gain +0.0002, top-decile lift 4.14; gap to early fusion AUC -0.030.
- D-S/Yager long-delay AUC: 0.705/0.702; learned-eta Full ACRF post residual-belief diff: +0.590.
- Fixed-eta Full ACRF long-delay AUC 0.706; cancellation AUC 0.710.

Model ranking by long-delay AUC:
- early_fusion_wad: AUC 0.721; PR-AUC 0.263; Brier gain +0.0048; top lift 3.61.
- residual_logistic_fusion: AUC 0.721; PR-AUC 0.263; Brier gain +0.0048; top lift 3.61.
- acrf_no_reliability: AUC 0.707; PR-AUC 0.237; Brier gain +0.0035; top lift 3.35.
- full_acrf_eta: AUC 0.707; PR-AUC 0.237; Brier gain +0.0035; top lift 3.34.
- full_acrf: AUC 0.706; PR-AUC 0.236; Brier gain +0.0035; top lift 3.34.
- acrf_no_async: AUC 0.706; PR-AUC 0.230; Brier gain +0.0032; top lift 3.34.
- ds_fusion: AUC 0.705; PR-AUC 0.229; Brier gain +0.0031; top lift 3.34.
- yager_fusion: AUC 0.702; PR-AUC 0.226; Brier gain +0.0031; top lift 3.35.
- calendar_weather_demand: AUC 0.671; PR-AUC 0.170; Brier gain +0.0000; top lift 2.34.