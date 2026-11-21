import math
import numpy as np
import pandas as pd

SLIM_FILE = "swift_slim.csv"
ANOM_FILE = "anomalies_full.csv"
OUT_FILE = "subgoal1_results.txt"

logs = []


def log(msg=""):
    print(msg)
    logs.append(str(msg))


df_anom = pd.read_csv(ANOM_FILE)
target_accounts = set(df_anom["ORDERING_ACCOUNT"].dropna().astype(str))

# 常见发款国 UTC 时区偏移（用于局部时间与周五晚排队检验）
TZ_OFFSET = {
    "US": -5, "CA": -5, "BR": -3,
    "GB": 0, "IE": 0, "PT": 0,
    "DE": 1, "FR": 1, "CH": 1, "NL": 1, "IT": 1, "ES": 1, "BE": 1, "AT": 1, "LU": 1, "SE": 1, "NO": 1, "DK": 1, "PL": 1,
    "GR": 2, "FI": 2, "ZA": 2,
    "AE": 4, "SA": 3, "IN": 5.5,
    "SG": 8, "CN": 8, "HK": 8, "MY": 8, "TW": 8,
    "JP": 9, "KR": 9, "AU": 10, "NZ": 12,
}

# ==============================================================================
# 1. 分块加载 1,000 万行核心列（总内存占用约 450MB）
# ==============================================================================
read_cols = [
    "EPOCH_SECONDS", "CURRENCY_CODE", "USD_AMOUNT",
    "ORDERING_BIC8", "BENEFICIARY_BIC8", "TOTAL_DURATION",
    "CONTINUITY_QFLAG", "CURRENCY_CONVERSION_COUNT", "PARTICIPANT_COUNT",
    "TRANSACTION_TYPE_CODE", "ORDERING_ACCOUNT",
    "ORDERING_TYPE", "BENEFICIARY_TYPE", "ANOMALY_TYPE",
]

arr_dur, arr_hops, arr_fx, arr_route, arr_qflag = [], [], [], [], []
arr_o_type, arr_b_type, arr_amt, arr_is_anom, arr_anom_code = [], [], [], [], []
arr_o_cty, arr_b_cty, arr_curr = [], [], []
arr_epoch, arr_acc_hash = [], []
acc_norm_records = []

type_map = {"B": 0, "C": 1}
first_chunk = True

for chunk in pd.read_csv(SLIM_FILE, usecols=read_cols, chunksize=500_000):
    if first_chunk:
        log("TRANSACTION_TYPE_CODE 原始取值分布（前50万行）:")
        log(chunk["TRANSACTION_TYPE_CODE"].value_counts(dropna=False).to_string())
        log("\nANOMALY_TYPE 原始取值分布（前50万行）:")
        log(chunk["ANOMALY_TYPE"].value_counts(dropna=False).to_string())
        log("-" * 72)
        first_chunk = False

    n = len(chunk)
    au = chunk["ANOMALY_TYPE"].fillna("NORMAL").astype(str).str.upper()
    is_an = (~au.isin(["NONE", "NORMAL", "NAN", ""])).astype(np.int8)
    arr_is_anom.append(is_an.to_numpy())

    # 异常细分编码: 0=Normal, 1=FAT_FINGER, 2=DUPLICATE, 3=OTHER_ANOM
    ac = np.zeros(n, dtype=np.int8)
    ac[is_an == 1] = 3
    ac[au.str.contains("FAT_FINGER")] = 1
    ac[au.str.contains("DUPLICATE")] = 2
    arr_anom_code.append(ac)

    arr_dur.append(chunk["TOTAL_DURATION"].to_numpy(dtype=np.float32))
    arr_hops.append(chunk["PARTICIPANT_COUNT"].fillna(-1).to_numpy(dtype=np.int8))
    arr_fx.append(chunk["CURRENCY_CONVERSION_COUNT"].fillna(-1).to_numpy(dtype=np.int8))
    arr_amt.append(chunk["USD_AMOUNT"].to_numpy(dtype=np.float64))
    arr_epoch.append(chunk["EPOCH_SECONDS"].to_numpy(dtype=np.int64))

    # 路由编码: 0=Direct, 1=Indirect, 2=Covered
    rt = chunk["TRANSACTION_TYPE_CODE"].astype(str).str.upper()
    rt_code = np.full(n, -1, dtype=np.int8)
    rt_code[rt.str.startswith("DIR")] = 0
    rt_code[rt.str.startswith("IND")] = 1
    rt_code[rt.str.contains("COV")] = 2
    arr_route.append(rt_code)

    qf = (chunk["CONTINUITY_QFLAG"].astype(str).str.upper() == "N").astype(np.int8)
    arr_qflag.append(qf.to_numpy())

    ot = chunk["ORDERING_TYPE"].astype(str).str.upper().map(type_map).fillna(-1).to_numpy(dtype=np.int8)
    bt = chunk["BENEFICIARY_TYPE"].astype(str).str.upper().map(type_map).fillna(-1).to_numpy(dtype=np.int8)
    arr_o_type.append(ot)
    arr_b_type.append(bt)

    octy = chunk["ORDERING_BIC8"].astype(str).str[4:6]
    bcty = chunk["BENEFICIARY_BIC8"].astype(str).str[4:6]
    arr_o_cty.append(octy.to_numpy())
    arr_b_cty.append(bcty.to_numpy())
    arr_curr.append(chunk["CURRENCY_CODE"].astype(str).to_numpy())

    # 账户哈希（用于千万级高速分组回测）
    acc_str = chunk["ORDERING_ACCOUNT"].astype(str)
    arr_acc_hash.append(pd.util.hash_pandas_object(acc_str, index=False).to_numpy(dtype=np.uint64))

    mask_target = (is_an == 0) & acc_str.isin(target_accounts)
    if mask_target.any():
        acc_norm_records.append(
            pd.DataFrame({
                "acc": acc_str[mask_target],
                "amt": chunk.loc[mask_target, "USD_AMOUNT"].to_numpy(dtype=np.float64),
            })
        )

dur = np.concatenate(arr_dur)
hops = np.concatenate(arr_hops)
fx = np.concatenate(arr_fx)
route = np.concatenate(arr_route)
qflag_n = np.concatenate(arr_qflag)
o_type = np.concatenate(arr_o_type)
b_type = np.concatenate(arr_b_type)
amt = np.concatenate(arr_amt)
is_anom = np.concatenate(arr_is_anom)
anom_code = np.concatenate(arr_anom_code)
epoch = np.concatenate(arr_epoch)
acc_hash = np.concatenate(arr_acc_hash)
o_cty = pd.Categorical(np.concatenate(arr_o_cty))
b_cty = pd.Categorical(np.concatenate(arr_b_cty))
curr = pd.Categorical(np.concatenate(arr_curr))

norm_mask = is_anom == 0
N_total = len(dur)
N_norm = int(norm_mask.sum())
log(f"\n全量加载完成: 总行数 = {N_total:,}, 正常交易 = {N_norm:,}, 异常交易 = {N_total - N_norm:,}\n")

# ==============================================================================
# [B0] 金额模型：付款方 × 受益方 (2×2 矩阵) + Lognormal σ + 按类型分层的币种中位数
# ==============================================================================
log("=" * 72)
log("[B0] 金额生成规则验证（Normal Traffic）")
pair_labels = {(0, 0): "B -> B", (0, 1): "B -> C", (1, 0): "C -> B", (1, 1): "C -> C"}
tot_norm_usd = np.nansum(amt[norm_mask])
for (ot_v, bt_v), label in pair_labels.items():
    m = norm_mask & (o_type == ot_v) & (b_type == bt_v) & (amt > 0)
    sub_a = amt[m]
    log(
        f"  {label}: 笔数占比={len(sub_a)/N_norm*100:5.2f}% | 金额占比={np.sum(sub_a)/tot_norm_usd*100:5.2f}% |"
        f" Median=${np.median(sub_a):8.2f} | Mean=${np.mean(sub_a):10.2f} | P90=${np.percentile(sub_a, 90):10.2f}"
    )

b_cnt_b = np.sum(norm_mask & (b_type == 0))
b_cnt_c = np.sum(norm_mask & (b_type == 1))
log(f"  受益方类型占比 (Normal): B = {b_cnt_b/(b_cnt_b+b_cnt_c)*100:.2f}%, C = {b_cnt_c/(b_cnt_b+b_cnt_c)*100:.2f}%")

for ot_v, name in [(0, "Business (B)"), (1, "Consumer (C)")]:
    sub_a = amt[norm_mask & (o_type == ot_v) & (amt > 0)]
    log(f"  {name} log(Amount) 标准差 σ = {np.std(np.log(sub_a)):.3f}")

log("\n  主要币种中位数（按 B->B 与 C->C 分层）:")
for c_code in ["EUR", "USD", "GBP", "CHF", "SGD", "CNY"]:
    m_bb = norm_mask & (curr == c_code) & (o_type == 0) & (b_type == 0) & (amt > 0)
    m_cc = norm_mask & (curr == c_code) & (o_type == 1) & (b_type == 1) & (amt > 0)
    m_all = norm_mask & (curr == c_code) & (amt > 0)
    if m_all.any():
        log(
            f"    {c_code}: 全量Median=${np.median(amt[m_all]):8.2f} |"
            f" B->B Median=${np.median(amt[m_bb]):8.2f} | C->C Median=${np.median(amt[m_cc]):7.2f}"
        )

# ==============================================================================
# [B1] 剔除异常后的走廊金额与总金额对比
# ==============================================================================
log("\n" + "=" * 72)
log("[B1] 剔除异常前后的走廊金额对比 (Top 10 Corridors by Normal Total USD)")
tot_all_usd = np.nansum(amt)
anom_usd = np.nansum(amt[~norm_mask])
log(
    f"  全量总金额 = ${tot_all_usd/1e9:.2f}B | 异常总金额 = ${anom_usd/1e9:.2f}B"
    f" ({anom_usd/tot_all_usd*100:.2f}%) | 正常总金额 = ${tot_norm_usd/1e9:.2f}B"
)

df_corr = pd.DataFrame({"o": o_cty, "b": b_cty, "amt": amt, "is_norm": norm_mask})
df_corr["corridor"] = df_corr["o"].astype(str) + " -> " + df_corr["b"].astype(str)
g_all = df_corr.groupby("corridor")["amt"].sum().rename("Total_All")
g_norm = df_corr[df_corr["is_norm"]].groupby("corridor")["amt"].agg(Tx_Norm="count", Total_Norm="sum", Median_Norm="median")
corr_cmp = pd.concat([g_all, g_norm], axis=1).sort_values("Total_Norm", ascending=False)
corr_cmp["Anom_Share_Pct"] = (corr_cmp["Total_All"] - corr_cmp["Total_Norm"]) / corr_cmp["Total_All"] * 100
for idx, r in corr_cmp.head(10).iterrows():
    log(
        f"  {idx:10s}: 正常总额=${r['Total_Norm']/1e9:6.2f}B (全量=${r['Total_All']/1e9:6.2f}B,"
        f" 异常占比={r['Anom_Share_Pct']:4.1f}%) | 正常笔数={int(r['Tx_Norm']):,} | 正常Median=${r['Median_Norm']:7.2f}"
    )
log(f"  独立发款国家数 = {len(o_cty.categories)}, 独立收款国家数 = {len(b_cty.categories)}")

# ==============================================================================
# [B2] 异常“指纹”表 & _DETAILS/_TOKENS 检查
# ==============================================================================
log("\n" + "=" * 72)
log("[B2] 异常交易事件级指纹与 2-hop Direct 基准")
norm_2hop_med = np.median(dur[norm_mask & (hops == 2) & (route == 0)])
norm_all_med = np.median(dur[norm_mask])
log(f"  正常交易中位时长: 全量 = {norm_all_med:.1f}s | 2-hop Direct = {norm_2hop_med:.1f}s")

df_anom["o_c"] = df_anom["ORDERING_BIC8"].astype(str).str[4:6]
df_anom["b_c"] = df_anom["BENEFICIARY_BIC8"].astype(str).str[4:6]
df_anom["is_xb"] = df_anom["o_c"] != df_anom["b_c"]
df_anom["is_cov"] = df_anom["TRANSACTION_TYPE_CODE"].astype(str).str.upper().str.contains("COV")
df_anom["is_2dir"] = (df_anom["PARTICIPANT_COUNT"] == 2) & df_anom["TRANSACTION_TYPE_CODE"].astype(str).str.upper().str.startswith("DIR")

for atype, grp in df_anom.groupby("ANOMALY_TYPE"):
    ev_cnt = grp.groupby(["ORDERING_ACCOUNT", "BENEFICIARY_ACCOUNT", "TOTAL_DURATION"]).ngroups
    med_all = grp["TOTAL_DURATION"].median()
    med_2d = grp.loc[grp["is_2dir"], "TOTAL_DURATION"].median()
    xb_pct = grp["is_xb"].mean() * 100
    cov_cnt = int(grp["is_cov"].sum())
    cov_pct = grp["is_cov"].mean() * 100
    log(
        f"  {atype:28s}: 行数={len(grp):4d} | 估算事件数={ev_cnt:4d} |"
        f" 全量Med={med_all:6.1f}s | 2跳Direct Med={med_2d:6.1f}s |"
        f" 跨境={xb_pct:5.1f}% | Covered={cov_cnt}/{len(grp)} ({cov_pct:.2f}%)"
    )

if acc_norm_records:
    df_acc_base = pd.concat(acc_norm_records, ignore_index=True).groupby("acc")["amt"].median().rename("base_med")
    ff = df_anom[df_anom["ANOMALY_TYPE"].astype(str).str.contains("FAT_FINGER")].copy()
    ff["acc"] = ff["ORDERING_ACCOUNT"].astype(str)
    ff = ff.join(df_acc_base, on="acc")
    ff = ff[ff["base_med"] > 0]
    ff["ratio"] = ff["USD_AMOUNT"] / ff["base_med"]
    log(
        f"\n  FAT_FINGER 相对账户正常中位数倍率 (n={len(ff)}):"
        f" <0.01x占比={(ff['ratio'] < 0.01).mean()*100:.1f}% |"
        f" >1x占比={(ff['ratio'] > 1.0).mean()*100:.1f}% |"
        f" P10={np.percentile(ff['ratio'], 10):.4f}x |"
        f" P50={np.percentile(ff['ratio'], 50):.1f}x |"
        f" P90={np.percentile(ff['ratio'], 90):.0f}x |"
        f" Max金额=${ff['USD_AMOUNT'].max()/1e9:.2f}B"
    )

# 检查 anomalies_full.csv 中的 _DETAILS / _TOKENS 非空率与样例
log("\n  anomalies_full.csv 中 _DETAILS / _TOKENS 字段非空率:")
for col in ["ORDERING_DETAILS", "ORDERING_TOKENS", "BENEFICIARY_DETAILS", "BENEFICIARY_TOKENS"]:
    non_null = df_anom[col].notna().mean() * 100
    sample_val = df_anom[col].dropna().iloc[0] if df_anom[col].notna().any() else "N/A"
    log(f"    {col:22s}: 非空率={non_null:5.1f}% | 样例={str(sample_val)[:60]}")

# ==============================================================================
# [B3] QFLAG = 'N' 独立链路截断模型验证（按跳数 & 按发款国）
# ==============================================================================
log("\n" + "=" * 72)
log("[B3] QFLAG = 'N' 链路独立截断验证")
p_direct = qflag_n[norm_mask & (hops == 2) & (route == 0)].mean()
log(f"  Direct (2-hop) 实测截断率 p = {p_direct*100:.2f}%")
for h in range(2, 9):
    m = norm_mask & (hops == h)
    if m.any():
        obs = qflag_n[m].mean() * 100
        pred = (1.0 - (1.0 - p_direct) ** (h - 1)) * 100
        log(f"  Hops={h} (n={m.sum():7,}): 实测 QFLAG='N' = {obs:5.2f}% | 独立边截断预测 1-(1-p)^(h-1) = {pred:5.2f}%")

df_qf_cty = (
    pd.DataFrame({"cty": o_cty[norm_mask], "qn": qflag_n[norm_mask]})
    .groupby("cty", observed=True)["qn"]
    .agg(Total="count", QN="sum")
)
df_qf_cty["Rate_Pct"] = df_qf_cty["QN"] / df_qf_cty["Total"] * 100
df_qf_cty = df_qf_cty.sort_values("Total", ascending=False)
log("\n  全部发款国归一化 QFLAG='N' 比例:")
for cty, r in df_qf_cty.iterrows():
    log(f"    {cty}: {r['Rate_Pct']:5.2f}% ({int(r['QN']):,} / {int(r['Total']):,})")

# ==============================================================================
# [B4] 多跳路由深度几何分布验证 (N_{k+1} / N_k)
# ==============================================================================
log("\n" + "=" * 72)
log("[B4] 多跳路由深度几何分布 (Indirect + Covered)")
mh_mask = norm_mask & (route > 0)
prev_cnt = None
for h in range(3, 9):
    cnt = int(np.sum(mh_mask & (hops == h)))
    ratio_str = f" | 比值 N_{h}/N_{h-1} = {cnt/prev_cnt:.3f}" if prev_cnt else ""
    log(f"  Hops={h}: {cnt:9,} 笔{ratio_str}")
    prev_cnt = cnt

# ==============================================================================
# [B5] 延迟分布（中位数、P90、2分钟内完成率）+ 多元线性回归
# ==============================================================================
log("\n" + "=" * 72)
log("[B5] 正常交易延迟分布（按跳数）与多元线性回归")
for h in range(2, 9):
    m = norm_mask & (hops == h) & np.isfinite(dur)
    if m.any():
        d = dur[m]
        log(
            f"  Hops={h} (n={len(d):7,}): Median={np.median(d):6.1f}s | Mean={np.mean(d):6.1f}s |"
            f" P90={np.percentile(d, 90):7.1f}s | <=120s占比={(d <= 120).mean()*100:5.1f}%"
        )

valid_reg = norm_mask & np.isfinite(dur) & (hops >= 2) & (fx >= 0) & (route >= 0)
y_r = dur[valid_reg].astype(np.float64)
X_r = np.column_stack([
    np.ones(len(y_r)),
    hops[valid_reg].astype(np.float64),
    fx[valid_reg].astype(np.float64),
    (route[valid_reg] == 1).astype(np.float64),
    (route[valid_reg] == 2).astype(np.float64),
])
XtX = X_r.T @ X_r
Xty = X_r.T @ y_r
beta = np.linalg.solve(XtX, Xty)
rss = float(y_r @ y_r - beta @ Xty)
tss = float(y_r @ y_r - (y_r.sum() ** 2) / len(y_r))
r2 = 1.0 - rss / tss
se = np.sqrt(np.diag((rss / (len(y_r) - 5)) * np.linalg.inv(XtX)))
t_vals = beta / se
p_vals = [math.erfc(abs(t) / math.sqrt(2.0)) for t in t_vals]

log(f"\n  多元线性回归 OLS (n={len(y_r):,}, R²={r2:.4f}):")
reg_names = [
    "Intercept (Direct基准)",
    "PARTICIPANT_COUNT (跳数)",
    "CURRENCY_CONVERSION_COUNT (换汇)",
    "Route: Indirect (vs Direct)",
    "Route: Covered (vs Direct)",
]
for nm, b, s, p in zip(reg_names, beta, se, p_vals):
    log(f"    {nm:34s}: beta = {b:+8.2f}s | SE = {s:.3f} | p = {p:.2e}")

cov_mean = np.mean(dur[norm_mask & (route == 2)])
ind_mean = np.mean(dur[norm_mask & (route == 1)])
log(f"  Covered vs Indirect 平均耗时: Indirect={ind_mean:.2f}s vs Covered={cov_mean:.2f}s (差值 = {ind_mean - cov_mean:.2f}s)")

# ==============================================================================
# [B6] 币种选择规律：本币相对全球份额的提升倍数 & EUR 非欧元区比例
# ==============================================================================
log("\n" + "=" * 72)
log("[B6] 币种选择机制：收款国本币相对全球份额的提升倍数")
glob_curr = pd.Series(curr[norm_mask]).value_counts(normalize=True)
home_map = {
    "US": "USD", "GB": "GBP", "CH": "CHF", "SG": "SGD", "CN": "CNY",
    "HK": "HKD", "JP": "JPY", "AU": "AUD", "CA": "CAD",
    "DE": "EUR", "FR": "EUR", "NL": "EUR", "IT": "EUR", "ES": "EUR",
}
df_bc = pd.DataFrame({"b_cty": b_cty[norm_mask], "curr": curr[norm_mask]})
for cty, h_curr in home_map.items():
    sub = df_bc[df_bc["b_cty"] == cty]
    if len(sub) > 10_000:
        local_share = (sub["curr"] == h_curr).mean()
        glob_share = glob_curr.get(h_curr, np.nan)
        log(
            f"  收款国 {cty} ({h_curr}): 当地占比={local_share*100:5.2f}% vs"
            f" 全球占比={glob_share*100:5.2f}% -> 提升倍数 = {local_share/glob_share:.2f}x"
        )

eurozone = {"AT", "BE", "CY", "EE", "FI", "FR", "DE", "GR", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PT", "SK", "SI", "ES", "HR"}
eur_mask = norm_mask & (curr == "EUR")
eur_to_non_ez = (~pd.Series(b_cty[eur_mask]).isin(eurozone)).mean() * 100
log(f"  EUR 付款流向非欧元区收款国的比例 = {eur_to_non_ez:.2f}%")

# ==============================================================================
# [B7] 1,000 万全量风控规则回测（|Δ log10| > 1.5 与 2 小时重复锁）
# ==============================================================================
log("\n" + "=" * 72)
log("[B7] 1,000 万全量风控规则回测 (Recall & 正常交易误报率 FPR)")
pos_mask = amt > 0
df_bt = pd.DataFrame({
    "acc": acc_hash[pos_mask],
    "log10": np.log10(amt[pos_mask]),
    "amt": amt[pos_mask],
    "ep": epoch[pos_mask],
    "ac": anom_code[pos_mask],
})
acc_med = df_bt[df_bt["ac"] == 0].groupby("acc")["log10"].median().rename("med")
df_bt = df_bt.join(acc_med, on="acc")
df_bt["med"] = df_bt["med"].fillna(df_bt.loc[df_bt["ac"] == 0, "log10"].median())

flag_ff = (df_bt["log10"] - df_bt["med"]).abs() > 1.5
is_ff = df_bt["ac"] == 1
is_nm = df_bt["ac"] == 0

tp_ff = int((flag_ff & is_ff).sum())
tot_ff = int(is_ff.sum())
fp_ff = int((flag_ff & is_nm).sum())
tot_nm = int(is_nm.sum())
log(
    f"  规则 A (|Δ log10| > 1.5 拦截 FAT_FINGER): Recall = {tp_ff/tot_ff*100:.2f}% ({tp_ff:,}/{tot_ff:,}) |"
    f" 正常交易 FPR = {fp_ff/tot_nm*100:.3f}% ({fp_ff:,}/{tot_nm:,}) | Precision = {tp_ff/(tp_ff+fp_ff)*100:.2f}%"
)

df_bt = df_bt.sort_values(["acc", "amt", "ep"])
same_pair = (df_bt["acc"] == df_bt["acc"].shift(1)) & (df_bt["amt"] == df_bt["amt"].shift(1))
dt = df_bt["ep"] - df_bt["ep"].shift(1)
flag_dup = same_pair & (dt >= 0) & (dt <= 7200)
is_dup = df_bt["ac"] == 2
tp_dup = int((flag_dup & is_dup).sum())
tot_dup = int(is_dup.sum())
fp_dup = int((flag_dup & is_nm).sum())
log(
    f"  规则 B (2小时同账户同金额重复锁): Recall = {tp_dup/tot_dup*100:.2f}% ({tp_dup:,}/{tot_dup:,}) |"
    f" 正常交易 FPR = {fp_dup/tot_nm*100:.4f}% ({fp_dup:,}/{tot_nm:,})"
)

# ==============================================================================
# [C1 & C2] 周五晚排队检验 + BIC8_PATH 中间行检查（前50万行样例）
# ==============================================================================
log("\n" + "=" * 72)
log("[C1 & C2] 周五晚延迟检验 & BIC8_PATH 中间行结构")
offsets = pd.Series(o_cty).map(TZ_OFFSET).fillna(0).to_numpy()
local_ep = epoch + (offsets * 3600).astype(np.int64)
local_dt = pd.to_datetime(local_ep, unit="s", utc=True)
dow = local_dt.dayofweek.to_numpy()
hour = local_dt.hour.to_numpy()

fri_eve = norm_mask & (dow == 4) & (hour >= 16) & np.isfinite(dur)
wk_day = norm_mask & (dow <= 3) & (hour >= 9) & (hour <= 15) & np.isfinite(dur)
log(f"  周一至四白天 (09-15h local, n={wk_day.sum():,}): Median={np.median(dur[wk_day]):.1f}s, Mean={np.mean(dur[wk_day]):.1f}s")
log(f"  周五晚间     (16-23h local, n={fri_eve.sum():,}): Median={np.median(dur[fri_eve]):.1f}s, Mean={np.mean(dur[fri_eve]):.1f}s")

sample_path = pd.read_csv(
    SLIM_FILE,
    usecols=["BIC8_PATH", "PARTICIPANT_COUNT", "CONTINUITY_QFLAG", "CURRENCY_CODE", "CURRENCY_CONVERSION_COUNT"],
    nrows=200_000,
)
log("\n  BIC8_PATH 前5行样例（含 QFLAG='Y' 与 'N'）:")
log(sample_path[sample_path["CONTINUITY_QFLAG"] == "Y"].head(3).to_string(index=False))
log(sample_path[sample_path["CONTINUITY_QFLAG"] == "N"].head(3).to_string(index=False))

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(logs))
log(f"\n全部结果已保存至 {OUT_FILE}")