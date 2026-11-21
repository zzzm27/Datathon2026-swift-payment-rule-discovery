import pandas as pd

df = pd.read_csv('anomalies_full.csv')

# 示例：导出每种异常的笔数和金额中位数，供看板直接画图
summary = (
    df.groupby('ANOMALY_TYPE')
    .agg(tx_count=('USD_AMOUNT', 'count'), median_usd=('USD_AMOUNT', 'median'))
    .reset_index()
)

summary.to_csv('chart_anomaly_summary.csv', index=False)