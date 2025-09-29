import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from meta_labeling.data_generation import dual_regime, prep_data, classification_stats, add_strat_metrics

# ==============================================================================
# ------------------------------------------------------------------------------
def calculate_mdd(returns_series):
    """
    Calculates the maximum drawdown from a pandas Series of returns.
    MDD is presented as a negative value.
    """
    # Calculate the cumulative returns (wealth index)
    cumulative_returns = (1 + returns_series.dropna()).cumprod()
    # Calculate the running maximum
    running_max = cumulative_returns.cummax()
    # Calculate the drawdown series
    drawdown = (cumulative_returns - running_max) / running_max
    # Return the minimum (i.e., most negative) value of the drawdown series
    return drawdown.min()
# ==============================================================================

all_results = []

for z in range(0, 1000):
    steps = 10000
    prob_switch = 0.20
    stdev = 0.014543365294448746 # About the same as IBM stdev

    data = dual_regime(total_steps=steps, prob_switch=prob_switch, stdev=stdev)

    model_data, data = prep_data(data=data, with_flags=True)

    # --- Modeling ---
    # -----------------------------------------------------------------------------------------------
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
    # Test data
    X_test_info_scaled = scaler_info.transform(X_test_info)
    X_test_regime_scaled = scaler_regime.transform(X_test_regime)

    # Train 2 models (Info, FP)
    meta_model_info = LogisticRegression(random_state=0, penalty=None)
    meta_model_regime = LogisticRegression(random_state=0, penalty=None)
    meta_model_info.fit(X_train_info_scaled, y_train)
    meta_model_regime.fit(X_train_regime_scaled, y_train)

    train_pred_info = meta_model_info.predict(X_train_info_scaled)
    train_pred_regime = meta_model_regime.predict(X_train_regime_scaled)

    train['pred_info'] = train_pred_info
    train['pred_regime'] = train_pred_regime

    train['prob_info'] = meta_model_info.predict_proba(X_train_info_scaled)[:, 1]
    train['prob_regime'] = meta_model_regime.predict_proba(X_train_regime_scaled)[:, 1]

    data['pred_info'] = 0.0
    data['prob_info'] = 0.0
    data['pred_regime'] = 0.0
    data['prob_regime'] = 0.0

    data.loc[train.index, 'pred_info'] = train['pred_info']
    data.loc[train.index, 'prob_info'] = train['prob_info']
    data.loc[train.index, 'pred_regime'] = train['pred_regime']
    data.loc[train.index, 'prob_regime'] = train['prob_regime']

    data_train_set = data.loc[train.index[0]:train.index[-1]].copy()

    meta_rets_info = (data_train_set['pred_info'] * data_train_set['target_rets']).shift(1)
    data_train_set['meta_rets_info'] = meta_rets_info
    meta_rets_regime = (data_train_set['pred_regime'] * data_train_set['target_rets']).shift(1)
    data_train_set['meta_rets_regime'] = meta_rets_regime
    data_train_set.dropna(inplace=True)

    train_cumrets = pd.DataFrame({'meta_info': ((data_train_set['meta_rets_info'] + 1).cumprod()),
                                  'meta_regime': ((data_train_set['meta_rets_regime'] + 1).cumprod()),
                                  'primary': ((data_train_set['prets'] + 1).cumprod()),
                                  'BAH': ((data_train_set['rets'] + 1).cumprod())})

    test['pred_info'] = meta_model_info.predict(X_test_info_scaled)
    test['prob_info'] = meta_model_info.predict_proba(X_test_info_scaled)[:, 1]
    test['pred_regime'] = meta_model_regime.predict(X_test_regime_scaled)
    test['prob_regime'] = meta_model_regime.predict_proba(X_test_regime_scaled)[:, 1]

    data.loc[test.index, 'pred_info'] = test['pred_info']
    data.loc[test.index, 'prob_info'] = test['prob_info']
    data.loc[test.index, 'pred_regime'] = test['pred_regime']
    data.loc[test.index, 'prob_regime'] = test['prob_regime']

    data_test_set = data.loc[test.index[0]:test.index[-1]].copy()

    meta_rets_info = (data_test_set['pred_info'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets_info'] = meta_rets_info
    meta_rets_regime = (data_test_set['pred_regime'] * data_test_set['target_rets']).shift(1)
    data_test_set['meta_rets_regime'] = meta_rets_regime
    data_test_set.dropna(inplace=True)

    test_cumrets = pd.DataFrame({'meta_info': ((data_test_set['meta_rets_info'] + 1).cumprod()),
                                 'meta_regime': ((data_test_set['meta_rets_regime'] + 1).cumprod()),
                                 'primary': ((data_test_set['prets'] + 1).cumprod()),
                                 'BAH': ((data_test_set['rets'] + 1).cumprod())})

    # --- Statistics ---
    # -----------------------------------------------------------------------------------------------
    # Primary model stats
    brow = classification_stats(actual=test['target'], predicted=test['pmodel'], prefix='b',
                                get_specificity=False)
    # Information Advantage
    irow = classification_stats(actual=y_test, predicted=test['pred_info'], prefix='mi',
                                get_specificity=True)
    # False Positive Modeling
    fprow = classification_stats(actual=y_test, predicted=test['pred_regime'], prefix='fp',
                                 get_specificity=True)
    # Concat data
    final_row = pd.concat([brow, irow, fprow], axis=1)

    # Add Strategy Metrics
    add_strat_metrics(row=final_row, rets=data_test_set['rets'], prefix='bah')
    add_strat_metrics(row=final_row, rets=data_test_set['prets'], prefix='p')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets_info'], prefix='imeta')
    add_strat_metrics(row=final_row, rets=data_test_set['meta_rets_regime'], prefix='fmeta')
    
    # ==============================================================================
    # <<-- ADDED: MDD calculation for each strategy -->>
    # ------------------------------------------------------------------------------
    final_row['bah_mdd'] = calculate_mdd(data_test_set['rets'])
    final_row['p_mdd'] = calculate_mdd(data_test_set['prets'])
    final_row['imeta_mdd'] = calculate_mdd(data_test_set['meta_rets_info'])
    final_row['fmeta_mdd'] = calculate_mdd(data_test_set['meta_rets_regime'])
    # ==============================================================================
    
    final_row['num_samples'] = y_test.shape[0]

    # Comparison Metrics
    final_row['ip_sr'] = final_row['imeta_sr'] - final_row['p_sr']
    final_row['ip_avg'] = final_row['imeta_mean'] - final_row['p_mean']
    final_row['ip_std'] = final_row['imeta_stdev'] - final_row['p_stdev']

    final_row['fp_sr'] = final_row['fmeta_sr'] - final_row['p_sr']
    final_row['fp_avg'] = final_row['fmeta_mean'] - final_row['p_mean']
    final_row['fp_std'] = final_row['fmeta_stdev'] - final_row['p_stdev']

    # Classification Metrics
    final_row['ip_recall'] = final_row['mi_recall'] - final_row['b_recall']
    final_row['fp_recall'] = final_row['fp_recall'] - final_row['b_recall']

    final_row['ip_prec'] = final_row['mi_precision'] - final_row['b_precision']
    final_row['fp_prec'] = final_row['fp_precision'] - final_row['b_precision']

    final_row['ip_acc'] = final_row['mi_accuracy'] - final_row['b_accuracy']
    final_row['fp_acc'] = final_row['fp_accuracy'] - final_row['b_accuracy']

    final_row['ip_f1'] = final_row['mi_weighted_avg_f1'] - final_row['b_weighted_avg_f1']
    final_row['fp_f1'] = final_row['fp_weighted_avg_f1'] - final_row['b_weighted_avg_f1']

    final_row['ip_auc'] = final_row['mi_auc'] - final_row['b_auc']
    final_row['fp_auc'] = final_row['fp_auc'] - final_row['b_auc']
    
    all_results.append(final_row)

final_report = pd.concat(all_results, ignore_index=True)
final_report.to_csv('hyp1_all_1000_results.csv', index=False)

print("완료")



#------------------------------------------------
#지표 평가 및 시각화

# ==============================================================================
# 셀: 최종 결과 요약, 비교 분석 및 시각화
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    results_df = pd.read_csv('hyp1_all_1000_results.csv')
except FileNotFoundError:
    print("오류: 'hyp1_all_1000_results.csv' 파일을 찾을 수 없습니다.")
    results_df = pd.DataFrame()

if not results_df.empty:
    results_df.dropna(inplace=True)
    
    # --- 1. 분류 성능 지표 요약표 ---
    summary_class_data = {
        'Primary Model (Before)': {
            'Precision': results_df['b_precision'].mean(),
            'Recall': results_df['b_recall'].abs().mean(),
            'F1-score': results_df['b_f1_score'].mean(),
            'Accuracy': results_df['b_accuracy'].mean(),
            'AUC': results_df['b_auc'].mean() 
        },
        'Meta-Info Model': {
            'Precision': results_df['mi_precision'].mean(),
            'Recall': results_df['mi_recall'].abs().mean(),
            'F1-score': results_df['mi_f1_score'].mean(),
            'Accuracy': results_df['mi_accuracy'].mean(),
            'AUC': results_df['mi_auc'].mean()  
        },
        'Meta-FP Model (After)': {
            'Precision': results_df['fp_precision'].mean(),
            'Recall': results_df['fp_recall'].abs().mean(),
            'F1-score': results_df['fp_f1_score'].mean(),
            'Accuracy': results_df['fp_accuracy'].mean(),
            'AUC': results_df['fp_auc'].mean() 
        }
    }
    summary_class = pd.DataFrame(summary_class_data)

    # --- 2. 전략 성과 지표 요약표 ---
    strategy_order = ['bah', 'p', 'imeta', 'fmeta']
    strategy_display_names = ['BAH', 'Primary', 'Meta-Info', 'Meta-FP']
    
    summary_data_strat = []
    mdd_available = all(f'{prefix}_mdd' in results_df.columns for prefix in strategy_order)

    for i, strat_prefix in enumerate(strategy_order):
        row = {'Strategy': strategy_display_names[i]}
        row['Sharpe Ratio'] = results_df[f'{strat_prefix}_sr'].mean()
        row['Mean Return'] = results_df[f'{strat_prefix}_mean'].mean()
        row['Volatility'] = results_df[f'{strat_prefix}_stdev'].mean()
        if mdd_available:
            row['Maximum Drawdown'] = results_df[f'{strat_prefix}_mdd'].mean()
        summary_data_strat.append(row)

    summary_strat = pd.DataFrame(summary_data_strat).set_index('Strategy')

    # --- 결과 출력 ---
    print("="*80)
    print("## 모델별 분류 성능 비교 (시뮬레이션 평균)")
    print("="*80)
    print(summary_class.round(4))

    print("\n" + "="*80)
    print("## 전체 전략 성과 지표 비교 (시뮬레이션 평균)")
    print("="*80)
    print(summary_strat.round(4))

    # --- 3. 분류 성능 지표 시각화 (Bar Chart) ---
    sns.set(style="whitegrid")
    summary_class.T.plot(kind='bar', figsize=(14, 8), rot=0, colormap='viridis')
    plt.title('Classification Performance Comparison (Average of Simulations)', fontsize=16)
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig('classification_performance.png')
    plt.show() 
    
    # --- 4. 전략 성과 지표 시각화 (Boxplots) ---
    value_vars_to_plot = [f'{s}_sr' for s in strategy_order] + \
                         [f'{s}_mean' for s in strategy_order] + \
                         [f'{s}_stdev' for s in strategy_order]
    
    if mdd_available:
        value_vars_to_plot.extend([f'{s}_mdd' for s in strategy_order])

    plot_df_strat_melted = pd.melt(results_df, 
                                   value_vars=value_vars_to_plot,
                                   var_name='Metric_Name', value_name='Value')
    
    plot_df_strat_melted['Strategy'] = plot_df_strat_melted['Metric_Name'].apply(lambda x: strategy_display_names[strategy_order.index(x.split('_')[0])])
    plot_df_strat_melted['Metric'] = plot_df_strat_melted['Metric_Name'].apply(lambda x: x.split('_', 1)[1]) 
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Strategy Performance Metrics Distribution (Simulations)', fontsize=20)
    
    sns.boxplot(x='Strategy', y='Value', data=plot_df_strat_melted[plot_df_strat_melted['Metric'] == 'sr'], ax=axes[0, 0], order=strategy_display_names)
    axes[0, 0].set_title('Sharpe Ratio', fontsize=15)
    
    sns.boxplot(x='Strategy', y='Value', data=plot_df_strat_melted[plot_df_strat_melted['Metric'] == 'mean'], ax=axes[0, 1], order=strategy_display_names)
    axes[0, 1].set_title('Mean Return', fontsize=15)

    sns.boxplot(x='Strategy', y='Value', data=plot_df_strat_melted[plot_df_strat_melted['Metric'] == 'stdev'], ax=axes[1, 0], order=strategy_display_names)
    axes[1, 0].set_title('Volatility', fontsize=15)
    
    if mdd_available:
        sns.boxplot(x='Strategy', y='Value', data=plot_df_strat_melted[plot_df_strat_melted['Metric'] == 'mdd'], ax=axes[1, 1], order=strategy_display_names)
        axes[1, 1].set_title('Maximum Drawdown', fontsize=15)
        axes[1, 1].set_visible(True)
    else:
        axes[1, 1].set_visible(False)

    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('strategy_performance.png')
    plt.show() 

else:
    print("\n결과 파일이 비어 있어 요약 및 시각화를 진행할 수 없습니다.")
