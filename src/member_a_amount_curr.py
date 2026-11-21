import pandas as pd

df = pd.read_csv('swift_slim.csv')

# 自动匹配金额列和币种列
amt_col = [c for c in df.columns if 'AMOUNT' in c or 'USD' in c][0]
curr_cols = [c for c in df.columns if 'CURR' in c and 'COUNT' not in c]
print(f'识别到金额列: {amt_col} | 币种列: {curr_cols}')

# 1. 规则一：企业 (B) vs 个人 (C) 的交易金额差异
if 'ORDERING_TYPE' in df.columns:
  amt_summary = (
      df.groupby('ORDERING_TYPE')[amt_col]
      .agg(
          Tx_Count='count',
          Mean_Amount='mean',
          Median_Amount='median',
          P90_Amount=lambda x: x.quantile(0.90),
      )
      .reset_index()
  )
  print('\n=== 1. 企业(B) vs 个人(C) 金额分布 ===')
  print(amt_summary.to_string(index=False))
  amt_summary.to_csv('chart_amount_entity.csv', index=False)

# 2. 规则二：主要结算币种分布 Top 10
main_curr = curr_cols[0]
curr_summary = (
    df.groupby(main_curr)[amt_col]
    .agg(Tx_Count='count', Median_Amount='median')
    .sort_values(by='Tx_Count', ascending=False)
    .head(10)
    .reset_index()
)
curr_summary['Share_Pct'] = (curr_summary['Tx_Count'] / len(df) * 100).round(2)
print(f'\n=== 2. Top 10 结算币种 ({main_curr}) ===')
print(curr_summary.to_string(index=False))
curr_summary.to_csv('chart_top_currencies.csv', index=False)