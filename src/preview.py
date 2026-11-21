import pandas as pd

file_path = 'swift_umdac_datathon_2026_fictitious_payment_data.csv'

# 只读取前 10 行预览，速度极快且不会卡死
df = pd.read_csv(file_path, nrows=10)

print('=== 所有列名 ===')
print(df.columns.tolist())

print('\n=== 前 10 行数据 ===')
print(df)