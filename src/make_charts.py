import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 统一视觉风格（深蓝 + 琥珀橙 + 学术灰，高分辨率无边框）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

SLIM_FILE = "swift_slim.csv"
ANOM_FILE = "anomalies_full.csv"

# 1. 读取轻量列用于绘图与阈值灵敏度测试
use_cols = [
    "USD_AMOUNT", "ORDERING_BIC8", "BENEFICIARY_BIC8", "TOTAL_DURATION",
    "CURRENCY_CODE", "PARTICIPANT_COUNT", "TRANSACTION_TYPE_CODE",
    "ORDERING_ACCOUNT", "ANOMALY_TYPE",
]
arr_dur, arr_hops, arr_route, arr_amt, arr_is_anom, arr_is_ff = [], [], [], [], [], []
arr_b_cty, arr_curr, arr_acc = [], [], []

for chunk in pd.read_csv(SLIM_FILE, usecols=use_cols, chunksize=500_000, low_memory=False):
    au = chunk["ANOMALY_TYPE"].fillna("NORMAL").astype(str).str.upper()
    is_an = (~au.isin(["NONE", "NORMAL", "NAN", ""])).astype(np.int8)
    arr_is_anom.append(is_an.to_numpy())
    arr_is_ff.append(au.str.contains("FAT_FINGER").astype(np.int8).to_numpy())

    arr_dur.append(chunk["TOTAL_DURATION"].to_numpy(dtype=np.float32))
    arr_hops.append(chunk["PARTICIPANT_COUNT"].fillna(-1).to_numpy(dtype=np.int8))
    arr_amt.append(chunk["USD_AMOUNT"].to_numpy(dtype=np.float64))

    rt = chunk["TRANSACTION_TYPE_CODE"].astype(str).str.upper()
    rc = np.full(len(chunk), -1, dtype=np.int8)
    rc[rt.str.startswith("DIR")] = 0
    rc[rt.str.startswith("IND")] = 1
    rc[rt.str.contains("COV")] = 2
    arr_route.append(rc)

    arr_b_cty.append(chunk["BENEFICIARY_BIC8"].astype(str).str[4:6].to_numpy())
    arr_curr.append(chunk["CURRENCY_CODE"].astype(str).to_numpy())
    arr_acc.append(pd.util.hash_pandas_object(chunk["ORDERING_ACCOUNT"].astype(str), index=False).to_numpy(dtype=np.uint64))

dur = np.concatenate(arr_dur)
hops = np.concatenate(arr_hops)
route = np.concatenate(arr_route)
amt = np.concatenate(arr_amt)
is_anom = np.concatenate(arr_is_anom)
is_ff = np.concatenate(arr_is_ff)
b_cty = pd.Categorical(np.concatenate(arr_b_cty))
curr = pd.Categorical(np.concatenate(arr_curr))
acc_h = np.concatenate(arr_acc)
norm = is_anom == 0

# 打印 |Δ log10| 阈值对比（1.5 vs 2.5 vs 3.0）
pos = amt > 0
df_t = pd.DataFrame({"acc": acc_h[pos], "l10": np.log10(amt[pos]), "nm": norm[pos], "ff": is_ff[pos] == 1})
med_s = df_t[df_t["nm"]].groupby("acc")["l10"].median().rename("med")
df_t = df_t.join(med_s, on="acc")
df_t["med"] = df_t["med"].fillna(df_t.loc[df_t["nm"], "l10"].median())
diff = (df_t["l10"] - df_t["med"]).abs()

print("【补充：|Δ log10| 阈值敏感性对比】")
for th in [1.5, 2.0, 2.5, 3.0]:
    flg = diff > th
    rec = (flg & df_t["ff"]).sum() / df_t["ff"].sum() * 100
    fpr = (flg & df_t["nm"]).sum() / df_t["nm"].sum() * 100
    print(f"  阈值 > {th:.1f}: FAT_FINGER Recall = {rec:5.2f}% | 正常交易 FPR = {fpr:6.3f}%")

# ==============================================================================
# 图 1：Slide 9 专用 — 2-hop Direct 异常中位时长 vs 正常基线 (39.0s)
# ==============================================================================
df_anom = pd.read_csv(ANOM_FILE)
df_anom["is_2dir"] = (df_anom["PARTICIPANT_COUNT"] == 2) & df_anom["TRANSACTION_TYPE_CODE"].astype(str).str.upper().str.startswith("DIR")
sub_2d = df_anom[df_anom["is_2dir"]]

items = [
    ("ROMANCE_SCAM\n(n=500 events)", sub_2d[sub_2d["ANOMALY_TYPE"] == "ROMANCE_SCAM"]["TOTAL_DURATION"].median(), "#1E3A8A"),
    ("CEO_SCAM\n(n=945 events)", sub_2d[sub_2d["ANOMALY_TYPE"] == "CEO_SCAM"]["TOTAL_DURATION"].median(), "#1E3A8A"),
    ("ACCT_REPLACEMENT\n(n=649 events)", sub_2d[sub_2d["ANOMALY_TYPE"] == "ACCOUNT_REPLACEMENT_SCAM"]["TOTAL_DURATION"].median(), "#1E3A8A"),
    ("DUPLICATE_PAY\n(n=185 events)", sub_2d[sub_2d["ANOMALY_TYPE"] == "DUPLICATE_PAYMENT"]["TOTAL_DURATION"].median(), "#F59E0B"),
    ("CUSTOMER_RECALL\n(n=317 events)", sub_2d[sub_2d["ANOMALY_TYPE"] == "CUSTOMER_RECALL"]["TOTAL_DURATION"].median(), "#F59E0B"),
    ("FAT_FINGER\n(n=1,940 events)", sub_2d[sub_2d["ANOMALY_TYPE"] == "FAT_FINGER_MISTAKE"]["TOTAL_DURATION"].median(), "#F59E0B"),
    ("NORMAL Baseline\n(2-hop Direct)", 39.0, "#64748B"),
]
labels, vals, colors = zip(*items)

fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=300)
bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.62)
ax.axvline(39.0, color="#DC2626", linestyle="--", linewidth=1.3, label="Normal 2-hop Direct Baseline (39.0s)")
for bar, v in zip(bars, vals[::-1]):
    ax.text(v + 22, bar.get_y() + bar.get_height() / 2, f"{v:,.1f}s", va="center", fontsize=9.5, fontweight="bold", color="#1E293B")
ax.set_xlim(0, 1280)
ax.set_xlabel("Median Settlement Duration in 2-Hop Direct Tier (Seconds)", fontsize=10)
ax.set_title("Social Engineering Adds 454s–1,102s Holds; Operational Anomalies Clear at ~40s STP Baseline", fontsize=11, fontweight="bold", pad=12, color="#0F172A")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("chart1_2hop_anomaly_latency.png", dpi=300)
plt.close()

# ==============================================================================
# 图 2：Slide 6 & Dashboard 专用 — 按跳数分布的中位数、P90 与 <=120s 快速结算占比
# ==============================================================================
hs = [2, 3, 4, 5, 6, 7, 8]
meds, p90s, fast_pcts = [], [], []
for h in hs:
    d = dur[norm & (hops == h) & np.isfinite(dur)]
    meds.append(np.median(d))
    p90s.append(np.percentile(d, 90))
    fast_pcts.append((d <= 120).mean() * 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)
x = np.arange(len(hs))
w = 0.36
b1 = ax1.bar(x - w / 2, meds, width=w, color="#1E3A8A", label="Median Duration (s)")
b2 = ax1.bar(x + w / 2, p90s, width=w, color="#93C5FD", label="P90 Tail Duration (s)")
ax1.axhline(56.0, color="#DC2626", linestyle="--", linewidth=1.1, label="All-Normal Median (56.0s)")
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, f"{int(bar.get_height())}", ha="center", fontsize=8, color="#1E3A8A", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{h} hops" for h in hs], fontsize=9)
ax1.set_ylabel("Seconds", fontsize=9.5)
ax1.set_ylim(0, 5000)
ax1.set_title("Median & P90 Latency Scale with Network Hops", fontsize=10.5, fontweight="bold", color="#0F172A")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.legend(fontsize=8, loc="upper left")

bars2 = ax2.bar(x, fast_pcts, width=0.52, color="#F59E0B")
for bar, pct in zip(bars2, fast_pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2, pct + 1.2, f"{pct:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#1E293B")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{h} hops" for h in hs], fontsize=9)
ax2.set_ylabel("Share Settled Within 120s (%)", fontsize=9.5)
ax2.set_ylim(0, 70)
ax2.set_title("Fast-Track (<=120s) Share Drops from 59.5% to 15.1%", fontsize=10.5, fontweight="bold", color="#0F172A")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("chart2_hops_latency_distribution.png", dpi=300)
plt.close()

# ==============================================================================
# 图 3：Slide 11/13 & Dashboard Page 5 — 货币对齐对比柱状图（替换圆环图）+ 本币提升倍数
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=300)

# 左子图：直接替换原误导性的圆环图（两根柱对比平均 FX 次数）
cats = ["Local Currency\n(Aligned)", "Foreign Currency\n(Unaligned)"]
fx_vals = [0.78, 1.60]
b_fx = ax1.bar(cats, fx_vals, width=0.45, color=["#1E3A8A", "#F59E0B"])
for bar, v in zip(b_fx, fx_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.2f} FX", ha="center", fontsize=10, fontweight="bold", color="#0F172A")
ax1.set_ylim(0, 1.95)
ax1.set_ylabel("Mean FX Conversions per Payment", fontsize=9.5)
ax1.set_title("Currency Mismatch Doubles FX Conversions (0.78 vs. 1.60)", fontsize=10.2, fontweight="bold", color="#0F172A")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", linestyle=":", alpha=0.5)

# 右子图：收款国本币相对全球份额提升倍数（解释 EUR 非欧元区流动 + 本币加成规则）
glob_curr = pd.Series(curr[norm]).value_counts(normalize=True)
df_bc = pd.DataFrame({"b": b_cty[norm], "c": curr[norm]})
lift_items = [
    ("JP (JPY)", "JP", "JPY"), ("CN (CNY)", "CN", "CNY"), ("GB (GBP)", "GB", "GBP"),
    ("SG (SGD)", "SG", "SGD"), ("HK (HKD)", "HK", "HKD"), ("CA (CAD)", "CA", "CAD"),
    ("US (USD)", "US", "USD"), ("AU (AUD)", "AU", "AUD"), ("CH (CHF)", "CH", "CHF"),
    ("DE (EUR)", "DE", "EUR"),
]
l_names, l_vals = [], []
for lbl, cty, hc in lift_items:
    sub = df_bc[df_bc["b"] == cty]
    l_names.append(lbl)
    l_vals.append((sub["c"] == hc).mean() / glob_curr[hc])

bars_l = ax2.barh(l_names[::-1], l_vals[::-1], color="#1E3A8A", height=0.62)
ax2.axvline(1.0, color="#DC2626", linestyle="--", linewidth=1.2, label="Global Baseline Weight (1.00x)")
for bar, v in zip(bars_l, l_vals[::-1]):
    ax2.text(v + 0.04, bar.get_y() + bar.get_height() / 2, f"{v:.2f}x", va="center", fontsize=8.5, fontweight="bold", color="#1E293B")
ax2.set_xlim(0, 3.0)
ax2.set_xlabel("Local Currency Share / Global Currency Share (Lift)", fontsize=9)
ax2.set_title("Currency Choice = Global Pareto Weight × Home-Currency Boost", fontsize=10.2, fontweight="bold", color="#0F172A")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", linestyle=":", alpha=0.5)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("chart3_currency_alignment_lift.png", dpi=300)
plt.close()

# ==============================================================================
# 图 4：Dashboard Page 3 & Slide 9 备用 — 异常笔数 vs 中位金额双子图（替换误导的双轴折线）
# ==============================================================================
anom_order = [
    "ROMANCE_SCAM", "FAT_FINGER_MISTAKE", "CEO_SCAM",
    "DUPLICATE_PAYMENT", "ACCOUNT_REPLACEMENT_SCAM", "CUSTOMER_RECALL",
]
short_names = ["ROMANCE", "FAT_FINGER", "CEO_SCAM", "DUPLICATE", "ACCT_REPL", "CUST_RECALL"]
cnts = [int((df_anom["ANOMALY_TYPE"] == k).sum()) for k in anom_order]
med_usds = [float(df_anom[df_anom["ANOMALY_TYPE"] == k]["USD_AMOUNT"].median()) for k in anom_order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)
b_c = ax1.barh(short_names[::-1], cnts[::-1], color="#1E3A8A", height=0.6)
for bar, v in zip(b_c, cnts[::-1]):
    ax1.text(v + 40, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", fontsize=8.5, fontweight="bold")
ax1.set_xlim(0, 2900)
ax1.set_xlabel("Flagged Transaction Count (Total = 7,006)", fontsize=9)
ax1.set_title("Anomaly Volume by Class (0.07% of 10M Flows)", fontsize=10.5, fontweight="bold", color="#0F172A")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="x", linestyle=":", alpha=0.5)

b_m = ax2.barh(short_names[::-1], med_usds[::-1], color="#F59E0B", height=0.6)
ax2.set_xscale("log")
ax2.axvline(3185, color="#DC2626", linestyle="--", linewidth=1.1, label="Normal Median (~$3.2K)")
for bar, v in zip(b_m, med_usds[::-1]):
    ax2.text(v * 1.18, bar.get_y() + bar.get_height() / 2, f"${v:,.0f}", va="center", fontsize=8.5, fontweight="bold")
ax2.set_xlim(50, 1_500_000)
ax2.set_xlabel("Median USD Amount (Log Scale)", fontsize=9)
ax2.set_title("FAT_FINGER Reaches $262K Median (~83.5x Account Baseline)", fontsize=10.5, fontweight="bold", color="#0F172A")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="x", linestyle=":", alpha=0.5)
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("chart4_anomaly_volume_and_median.png", dpi=300)
plt.close()

print("\n4 张高清图表已全部生成完毕:")
print("  1. chart1_2hop_anomaly_latency.png")
print("  2. chart2_hops_latency_distribution.png")
print("  3. chart3_currency_alignment_lift.png")
print("  4. chart4_anomaly_volume_and_median.png")