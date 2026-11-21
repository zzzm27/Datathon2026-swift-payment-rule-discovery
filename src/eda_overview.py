from collections import Counter
import time
import pandas as pd

start_time = time.time()
file_path = 'swift_umdac_datathon_2026_fictitious_payment_data.csv'

# 跳过占内存极大的长文本列，读取速度提升 10 倍以上
skip_cols = {
    'DISCLAIMER',
    'ORDERING_DETAILS',
    'BENEFICIARY_DETAILS',
    'ORDERING_TOKENS',
    'BENEFICIARY_TOKENS',
}

total_rows = 0
anomaly_counts = Counter()
pair_anomaly_counts = Counter()
tx_type_stats = {}
continuity_counts = Counter()
amount_stats = {}

print('开始分块极速扫描全量数据...')
for chunk in pd.read_csv(
    file_path, usecols=lambda c: c not in skip_cols, chunksize=500_000
):
  total_rows += len(chunk)
  anom = chunk['ANOMALY_TYPE'].fillna('NORMAL')

  # 1. 异常类型统计
  anomaly_counts.update(anom.value_counts().to_dict())

  # 2. 交易双方组合 vs 异常
  pairs = (
      chunk['ORDERING_TYPE'].astype(str)
      + ' -> '
      + chunk['BENEFICIARY_TYPE'].astype(str)
  )
  pair_anomaly_counts.update(zip(pairs, anom))

  # 3. 路由特征统计
  dur_col = (
      'TOTAL_DURATION'
      if 'TOTAL_DURATION' in chunk.columns
      else 'TOTAL DURATION'
  )
  for tx_type, grp in chunk.groupby('TRANSACTION_TYPE_CODE'):
    if tx_type not in tx_type_stats:
      tx_type_stats[tx_type] = [0, 0.0, 0.0, 0.0]
    tx_type_stats[tx_type][0] += len(grp)
    tx_type_stats[tx_type][1] += grp['PARTICIPANT_COUNT'].sum()
    tx_type_stats[tx_type][2] += grp['CURRENCY_CONVERSION_COUNT'].sum()
    tx_type_stats[tx_type][3] += grp[dur_col].sum()

  # 4. 连续性统计
  continuity_counts.update(
      zip(chunk['TRANSACTION_TYPE_CODE'], chunk['CONTINUITY_QFLAG'])
  )

  # 5. 金额统计
  for a_type, grp in chunk.groupby(anom):
    if a_type not in amount_stats:
      amount_stats[a_type] = [0, 0.0, float('inf'), float('-inf')]
    amount_stats[a_type][0] += len(grp)
    amount_stats[a_type][1] += grp['USD_AMOUNT'].sum()
    amount_stats[a_type][2] = min(
        amount_stats[a_type][2], grp['USD_AMOUNT'].min()
    )
    amount_stats[a_type][3] = max(
        amount_stats[a_type][3], grp['USD_AMOUNT'].max()
    )

  print(f'已扫描 {total_rows:,} 行... (耗时 {time.time() - start_time:.1f} 秒)')

print('\n================ 1. 总规模 ================')
print(f'总交易笔数: {total_rows:,} 行')

print('\n================ 2. 异常类型分布 ================')
print(pd.Series(anomaly_counts).sort_values(ascending=False))

print('\n================ 3. 交易双方组合 vs 异常 ================')
s_pair = pd.Series(pair_anomaly_counts)
s_pair.index = pd.MultiIndex.from_tuples(
    s_pair.index, names=['PAIR_TYPE', 'ANOMALY']
)
print(s_pair.unstack(fill_value=0))

print('\n================ 4. 路由方式平均特征 ================')
df_tx = pd.DataFrame.from_dict(
    tx_type_stats,
    orient='index',
    columns=['总笔数', '总参与者数', '总换汇次数', '总耗时'],
)
df_tx['平均参与者数'] = df_tx['总参与者数'] / df_tx['总笔数']
df_tx['平均换汇次数'] = df_tx['总换汇次数'] / df_tx['总笔数']
df_tx['平均耗时(秒)'] = df_tx['总耗时'] / df_tx['总笔数']
print(df_tx[['总笔数', '平均参与者数', '平均换汇次数', '平均耗时(秒)']])

print('\n================ 5. 路由方式 vs 路径连续性 (Y/N) ================')
s_cont = pd.Series(continuity_counts)
s_cont.index = pd.MultiIndex.from_tuples(
    s_cont.index, names=['TX_TYPE', 'CONTINUITY']
)
print(s_cont.unstack(fill_value=0))

print('\n================ 6. 各类交易金额 (USD_AMOUNT) ================')
df_amt = pd.DataFrame.from_dict(
    amount_stats, orient='index', columns=['笔数', '总金额', '最小金额', '最大金额']
)
df_amt['平均金额'] = df_amt['总金额'] / df_amt['笔数']
print(df_amt[['笔数', '平均金额', '最小金额', '最大金额']])