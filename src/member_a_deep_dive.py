import pandas as pd

df = pd.read_csv('swift_slim.csv')

# 1. 检验“参与银行数” vs “换汇次数”到底谁才是决定延迟的主因（用于验证/推翻假设）
print(
    '================ 1. 延迟拆解：跳数 vs 换汇次数对耗时(秒)的影响'
    ' ================'
)
latency_pivot = (
    df.groupby([
        'TRANSACTION_TYPE_CODE',
        'PARTICIPANT_COUNT',
        'CURRENCY_CONVERSION_COUNT',
    ])
    .agg(笔数=('TOTAL_DURATION', 'count'), 平均耗时秒=('TOTAL_DURATION', 'mean'))
    .reset_index()
)
print(
    latency_pivot.sort_values(
        by=['TRANSACTION_TYPE_CODE', 'PARTICIPANT_COUNT']
    ).to_string(index=False)
)

# 2. 排查 Direct 路由中为什么会有 31.5 万笔路径断层 (QFLAG == 'N')
print('\n================ 2. Direct 路由断层 (QFLAG=N) 国家分布 Top 10 ================')
direct_n = df[
    (df['TRANSACTION_TYPE_CODE'] == 'Direct') & (df['CONTINUITY_QFLAG'] == 'N')
].copy()
direct_n['FROM_COUNTRY'] = direct_n['ORDERING_BIC8'].str[4:6]
direct_n['TO_COUNTRY'] = direct_n['BENEFICIARY_BIC8'].str[4:6]
print('发款国 Top 10:')
print(direct_n['FROM_COUNTRY'].value_counts().head(10))
print('\n收款国 Top 10:')
print(direct_n['TO_COUNTRY'].value_counts().head(10))