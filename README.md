# Reverse-Engineering SWIFT Payment Rules & Pre-Settlement Risk Control

**UMDAC Datathon 2026 | Team: Just do it** (Teoh Zhen Ming, Koo Min Qi, Foong Chi San)

## Project Overview
Full-population analysis and pre-settlement risk guardrail backtest across **10,000,000 synthetic SWIFT payment records (4GB / $972.04B total volume)** using memory-safe Python streaming (`pandas`, `NumPy`, `Matplotlib`) and **Google Looker Studio**.

## Key Empirical Findings
- **Bilateral Value Topology:** Transaction amounts follow a multiplicative Ordering × Beneficiary entity model, with `B → B` generating **72.19%** of normal volume ($22,883 median vs. $464 for `C → C`).
- **Hop-Driven Latency Causality:** Full-population OLS regression (`n = 9,992,994`) proves each intermediary bank hop adds **+200.52s** settlement delay (`p = 0`), whereas FX conversions (`β = -0.51s, p = 0.531`) and Friday evening cutoffs show zero latency penalty.
- **Independent Telemetry Truncation:** `CONTINUITY_QFLAG = 'N'` (`7.67%` overall incidence) follows a memoryless **4.99% per-link drop rate** (`1 - (1 - 0.0499)^(h-1)`), invariant across onshore hubs and offshore centers.
- **10M-Scale Risk Control Backtest:** While anomalies represent `0.07%` of transactions (`7,006` rows), they drive **$176.62B (18.17%)** in dollar exposure. Account-relative log-amount thresholds (`> 2.5` soft step-up / `> 3.0` hard hold) and a 2-hour idempotency lock capture **44.02% / 31.34%** of `FAT_FINGER_MISTAKE` outliers and **72.75%** of `DUPLICATE_PAYMENT` replays at **< 1% false positive rate**.

## Repository Structure
- `src/` — Python streaming ETL, OLS causal verification, and 10M risk-control backtest scripts
- `reports/` — Final presentation deck and 7-page interactive Looker Studio dashboard export (PDF)
- `assets/` — High-resolution statistical figures and dashboard visualizations

## Key Visuals
![Latency Scaling by Network Hops](assets/chart1_latency_hops.png)
![Anomaly Volume and Tail-Risk Impact](assets/chart4_anomaly_volume_amount.png)
![Presentation slide](reports/Just%20do%20it_slide.pdf)
![Dashboard](reports/Just%20do%20it_dashboard.pdf)