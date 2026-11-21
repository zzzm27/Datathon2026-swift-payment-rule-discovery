import pandas as pd

print('正在读取精简版数据 swift_slim.csv ...')
df = pd.read_csv('swift_slim.csv')

# 1. 从 BIC8 提取国家代码（第 5-6 位字母代表 ISO 国家）
df['FROM_COUNTRY'] = df['ORDERING_BIC8'].str[4:6]
df['TO_COUNTRY'] = df['BENEFICIARY_BIC8'].str[4:6]

# 2. 统计不同路由类型（Direct / Indirect / Covered）的底层数学规律
print('\n================ 1. 路由类型与核心指标平均值 ================')
route_stats = (
    df.groupby('TRANSACTION_TYPE_CODE')
    .agg(
        总笔数=('USD_AMOUNT', 'count'),
        平均参与银行数=('PARTICIPANT_COUNT', 'mean'),
        平均换汇次数=('CURRENCY_CONVERSION_COUNT', 'mean'),
        平均耗时秒=('TOTAL_DURATION', 'mean'),
    )
    .reset_index()
)
print(route_stats.to_string())

# 3. 统计路径连续性 (CONTINUITY_QFLAG) 在不同路由下的分布
print('\n================ 2. 路由方式 vs 路径连续性(Y/N) ================')
print(pd.crosstab(df['TRANSACTION_TYPE_CODE'], df['CONTINUITY_QFLAG']))

# 4. 导出给 Looker Studio 看板用的汇总表
route_stats.to_csv('chart_route_overview.csv', index=False)
continuity_df = (
    df.groupby(['TRANSACTION_TYPE_CODE', 'CONTINUITY_QFLAG'])
    .size()
    .reset_index(name='count')
)
continuity_df.to_csv('chart_continuity_dist.csv', index=False)

print(
    '\n[完成] 已成功导出 chart_route_overview.csv 和'
    ' chart_continuity_dist.csv，可直接传给 Looker Studio 画图！'
)