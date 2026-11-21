import pandas as pd

file_path = 'swift_umdac_datathon_2026_fictitious_payment_data.csv'
long_text_cols = {
    'DISCLAIMER',
    'ORDERING_DETAILS',
    'BENEFICIARY_DETAILS',
    'ORDERING_TOKENS',
    'BENEFICIARY_TOKENS',
}

first_slim = True
first_anom = True

print('正在拆分生成轻量数据集...')
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=500_000)):
  # 1. 导出不含长文本的全量轻量表
  slim_chunk = chunk.drop(
      columns=[c for c in long_text_cols if c in chunk.columns]
  )
  slim_chunk.to_csv(
      'swift_slim.csv', mode='w' if first_slim else 'a', index=False, header=first_slim
  )
  first_slim = False

  # 2. 导出包含完整文本的所有异常记录
  anom_chunk = chunk[chunk['ANOMALY_TYPE'].notna()]
  if len(anom_chunk) > 0:
    anom_chunk.to_csv(
        'anomalies_full.csv',
        mode='w' if first_anom else 'a',
        index=False,
        header=first_anom,
    )
    first_anom = False

  print(f'已处理 {(i + 1) * 500_000:,} 行...')

print('完成！已生成 swift_slim.csv 和 anomalies_full.csv')