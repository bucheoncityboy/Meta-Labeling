#regime
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from meta_labeling.data_generation import dual_regime, prep_data, classification_stats, add_strat_metrics

def calculate_mdd(returns_series):
    """
    Calculates the maximum drawdown from a pandas Series of returns.
    MDD is presented as a negative value.
    """
    cumulative_returns = (1 + returns_series.dropna()).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

all_results = []

for z in range(0, 1000):
    steps = 10000
    prob_switch = 0.20
    stdev = 0.014543365294448746

    data = dual_regime(total_steps=steps, prob_switch=prob_switch, stdev=stdev)
    model_data, data = prep_data(data=data, with_flags=True)

    # --- Modeling ---
    train, test = train_test_split(model_data, test_size=0.4, shuffle=False)

    X_train_info = train[['rets', 'rets2', 'rets3']]
    X_test_info = test[['rets', 'rets2', 'rets3']]
    X_train_regime = train[['rets', 'rets2', 'rets3', 'regime']]
    X_test_regime = test[['rets', 'rets2', 'rets3', 'regime']]
    y_train = train['target']
    y_test = test['target']

    scaler_info = StandardScaler()
    scaler_regime = StandardScaler()
    X_train_info_scaled = scaler_info.fit_transform(X_train_info)
    X_train_regime_scaled = scaler_regime.fit_transform(X_train_regime)
    X_test_info_scaled = scaler_info.transform(X_test_info)
    X_test_regime_scaled = scaler_regime.transform(X_test_regime)

    meta_model_info = LogisticRegression(random_state=0, penalty=None)
    meta_model_regime = LogisticRegression(random_state=0, penalty=None)
    meta_model_info.fit(X_train_info_scaled, y_train)
    meta_model_regime.fit(X_train_regime_scaled, y_train)

    test['pred_info'] = meta_model_info.predict(X_test_info_scaled)
    test['pred_regime'] = meta_model_regime.predict(X_test_regime_scaled)

    data_test_set = data.loc[test.index[0]:test.index[-1]].copy()
    meta_rets_info = (test['pred_info'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets_info'] = meta_rets_info
    meta_rets_regime = (test['pred_regime'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets_regime'] = meta_rets_regime
    data_test_set.dropna(inplace=True)

    # --- Overall Statistics ---
    brow = classification_stats(actual=test['target'], predicted=test['pmodel'], prefix='b', get_specificity=False)
    irow = classification_stats(actual=y_test, predicted=test['pred_info'], prefix='mi', get_specificity=True)
    fprow = classification_stats(actual=y_test, predicted=test['pred_regime'], prefix='fp', get_specificity=True)
    final_row = pd.concat([brow, irow, fprow], axis=1)

    # ==============================================================================
    # <<-- ADDED: Regime-Specific Performance Calculation -->>
    # ------------------------------------------------------------------------------
    for regime_val in test['regime'].unique():
        test_subset = test[test['regime'] == regime_val]
        if not test_subset.empty:
            # Primary model stats for regime
            brow_r = classification_stats(actual=test_subset['target'], predicted=test_subset['pmodel'], prefix=f'b_r{regime_val}', get_specificity=False)
            # Meta-Info model stats for regime
            irow_r = classification_stats(actual=test_subset['target'], predicted=test_subset['pred_info'], prefix=f'mi_r{regime_val}', get_specificity=False)
            # Meta-FP model stats for regime
            fprow_r = classification_stats(actual=test_subset['target'], predicted=test_subset['pred_regime'], prefix=f'fp_r{regime_val}', get_specificity=False)
            
            regime_row = pd.concat([brow_r, irow_r, fprow_r], axis=1)
            final_row = pd.concat([final_row, regime_row], axis=1)
    # ==============================================================================

    # --- Strategy and Comparison Metrics ---
    add_strat_metrics(row=final_row, rets=data_test_set['rets'], prefix='bah')
    add_strat_metrics(row=final_row, rets=data_test_set['prets'], prefix='p')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets_info'], prefix='imeta')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets_regime'], prefix='fmeta')
    
    final_row['bah_mdd'] = calculate_mdd(data_test_set['rets'])
    final_row['p_mdd'] = calculate_mdd(data_test_set['prets'])
    final_row['imeta_mdd'] = calculate_mdd(data_test_set['meta_rets_info'])
    final_row['fmeta_mdd'] = calculate_mdd(data_test_set['meta_rets_regime'])
    
    # ... (The rest of your comparison metrics code) ...
    
    all_results.append(final_row)

final_report = pd.concat(all_results, ignore_index=True)
final_report.to_csv('hyp1_all_1000_results_regime.csv', index=False)

print("Regime별 성능 지표가 포함된 CSV 파일이 생성되었습니다.")


#지표 평가

#-----------------------------------------------------------
# 지표 평가
#-----------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 저장된 몬테카를로 시뮬레이션 결과 불러오기
try:
    results_df = pd.read_csv('hyp1_all_1000_results_regime.csv')
except FileNotFoundError:
    print("오류: 'hyp1_all_1000_results.csv' 파일을 찾을 수 없습니다.")
    results_df = pd.DataFrame()

if not results_df.empty:
    # --- 1. Create a Summary Table for Regime Performance ---
    models = {
        'Primary Model': 'b',
        'Meta-Info Model': 'mi',
        'Meta-FP Model': 'fp'
    }
    # metrics 리스트에 'auc' 추가
    metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'auc']
    
    summary_list = []
    for model_name, prefix in models.items():
        for metric in metrics:
            row = {'Model': model_name, 'Metric': metric}
            for regime in [0.0, 1.0]:
                col_name = f'{prefix}_r{regime}_{metric}'
                if col_name in results_df.columns:
                    row[f'Regime {int(regime)} Avg.'] = results_df[col_name].mean()
                else:
                    row[f'Regime {int(regime)} Avg.'] = np.nan 
            summary_list.append(row)
    
    summary_df = pd.DataFrame(summary_list).set_index(['Model', 'Metric'])
    
    print("="*80)
    print("## 모델 성능 Regime별 비교 (시뮬레이션 평균)")
    print("="*80)
    print(summary_df.round(4))

    # --- 2. Visualize Performance Difference ---
    # F1-Score 시각화
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    plot_data_f1 = pd.DataFrame({
        'Regime 0': results_df.get('fp_r0.0_f1_score'),
        'Regime 1': results_df.get('fp_r1.0_f1_score')
    }).dropna()

    if not plot_data_f1.empty:
        sns.boxplot(data=plot_data_f1, ax=ax1, palette='viridis')
        ax1.set_title('Meta-FP Model: F1-Score Distribution by Regime', fontsize=16, pad=20)
        ax1.set_ylabel('F1-Score', fontsize=12)
        ax1.set_xlabel('Regime', fontsize=12)
        print("\n\n" + "="*80)
        print("## 시각화: Meta-FP 모델의 Regime별 F1 Score 분포")
        print("="*80)
        plt.show()
