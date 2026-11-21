mport pandas as pd

anom = pd.read_csv('anomalies_full.csv')

# 1. 计算绕过人工审核、40s内直接放行的 STP 盲区涉案总金额 (FAT_FINGER_MISTAKE + CUSTOMER_RECALL)
stp_leak = anom[
    anom['ANOMALY_TYPE'].isin(['FAT_FINGER_MISTAKE', 'CUSTOMER_RECALL'])
]
print('=== 1. STP 盲区放跑的涉案总金额 (USD) ===')
print(stp_leak.groupby('ANOMALY_TYPE')['USD_AMOUNT'].agg(['count', 'sum']))
print(f"合计涉案金额: ${stp_leak['USD_AMOUNT'].sum():,.2f}")

# 2. 核对周六异常笔数中 B vs C 的精确比例
anom['DAY_OF_WEEK'] = pd.to_datetime(anom['INITIATION_TIME']).dt.day_name()
sat_anom = anom[anom['DAY_OF_WEEK'] == 'Saturday']['ORDERING_TYPE'].value_counts()
print('\n=== 2. 周六异常笔数 (B vs C) ===')
print(sat_anom)