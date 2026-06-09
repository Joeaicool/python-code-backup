#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, warnings, itertools, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats

# Scikit-learn 和各大模型库全系引入
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score,
                             f1_score, cohen_kappa_score, confusion_matrix, brier_score_loss, PrecisionRecallDisplay, auc)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.calibration import calibration_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'DejaVu Sans'

class Config:
    DATA_FILE = 'Final_Cleaned_Data.xlsx'
    TARGET_COL = 'status'
    RANDOM_SEED = 2516
    TEST_SIZE = 0.3
    RFE_N_FEATURES = 9

os.chdir(r'/home/lin/rx/肝占位病变/Model_Results(0.75)/S2516_Rem5_Top9_TestAUC0.770')

def generate_baseline_table(df, target_col, id_col):
    stats_list = []
    groups = sorted(df[target_col].dropna().unique())
    df_groups = {g: df[df[target_col] == g] for g in groups}
    for col in df.columns:
        if col in [target_col, id_col]: continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
            m_all, s_all = df[col].mean(), df[col].std()
            q1_all, med_all, q3_all = df[col].quantile([0.25, 0.5, 0.75])
            row_mean = {'Feature': f'{col} (Mean ± SD)', 'Overall': f'{m_all:.2f} ± {s_all:.2f}'}
            row_med = {'Feature': f'{col} (Median [IQR])', 'Overall': f'{med_all:.2f} [{q1_all:.2f}-{q3_all:.2f}]'}
            for g in groups:
                row_mean[f'Group {g}'] = f'{df_groups[g][col].mean():.2f} ± {df_groups[g][col].std():.2f}'
                q1g, medg, q3g = df_groups[g][col].quantile([0.25, 0.5, 0.75])
                row_med[f'Group {g}'] = f'{medg:.2f} [{q1g:.2f}-{q3g:.2f}]'
            stats_list.extend([row_mean, row_med])
        else:
            vc = df[col].value_counts(dropna=False)
            for val in vc.index:
                c_all = vc[val]; p_all = c_all / len(df) * 100
                row_cat = {'Feature': f'{col} = {val}', 'Overall': f'{c_all} ({p_all:.1f}%)'}
                for g in groups:
                    c_g = df_groups[g][col].value_counts(dropna=False).get(val, 0)
                    p_g = (c_g / len(df_groups[g]) * 100) if len(df_groups[g]) > 0 else 0
                    row_cat[f'Group {g}'] = f'{c_g} ({p_g:.1f}%)'
                stats_list.append(row_cat)
    return pd.DataFrame(stats_list)

def get_bootstrap_metrics_ci(y_true, y_prob, metric_func, n_bootstraps=1000):
    values = []; y_true, y_prob = np.array(y_true), np.array(y_prob)
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        yt, yp = y_true[idx], y_prob[idx]; ypr = (yp >= 0.5).astype(int)
        if metric_func == 'AUC': values.append(roc_auc_score(yt, yp))
        elif metric_func == 'ACC': values.append(accuracy_score(yt, ypr))
        elif metric_func == 'SENS': values.append(recall_score(yt, ypr))
        elif metric_func == 'SPEC': values.append(recall_score(yt, ypr, pos_label=0))
        elif metric_func == 'F1': values.append(f1_score(yt, ypr))
    if not values: return "NaN"
    return f"{np.mean(values):.3f} ({np.percentile(values, 2.5):.3f}-{np.percentile(values, 97.5):.3f})"

def brier_score_confidence_interval(y_true, y_pred_prob, n_bootstraps=1000):
    scores = []; yt = np.array(y_true)
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(yt), size=len(yt), replace=True)
        scores.append(np.mean((y_pred_prob[idx] - yt[idx]) ** 2))
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def bootstrap_auc_pvalue(y_true, p_a, p_b, seed=42, n_boot=2000):
    rng = np.random.RandomState(seed); y_true, p_a, p_b = np.array(y_true), np.array(p_a), np.array(p_b)
    orig_diff = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b); diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: continue
        diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    return float(np.mean(np.abs(diffs) >= np.abs(orig_diff)))

def calc_net_benefit(y_true, y_prob, thresholds):
    nb = []; y_t = np.array(y_true); y_p = np.array(y_prob)
    for t in thresholds:
        tp = np.sum((y_t == 1) & (y_p >= t)); fp = np.sum((y_t == 0) & (y_p >= t))
        nb.append((tp / len(y_t)) - (fp / len(y_t)) * (t / (1 - t)))
    return np.array(nb)

def robust_impute(X_train, X_test, seed):
    """防数据泄露插补：仅在训练集 fit，然后在训练和测试集分别 transform"""
    X_tr_imp = X_train.copy()
    X_te_imp = X_test.copy()
    
    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(exclude=np.number).columns
    
    if len(num_cols) > 0:
        iter_imputer = IterativeImputer(max_iter=10, random_state=seed)
        X_tr_imp[num_cols] = iter_imputer.fit_transform(X_train[num_cols])
        X_te_imp[num_cols] = iter_imputer.transform(X_test[num_cols])
        
    if len(cat_cols) > 0:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        X_tr_imp[cat_cols] = mode_imputer.fit_transform(X_train[cat_cols])
        X_te_imp[cat_cols] = mode_imputer.transform(X_test[cat_cols])
        
    return X_tr_imp, X_te_imp

def main():
    # 强制全局锁死 Seed，保证所有底层随机过程绝对 100% 可复现！
    np.random.seed(Config.RANDOM_SEED)
    
    print("Running...")
    # 注意：此时读取的数据是包含 NaNs 的 Raw Data
    df = pd.read_excel(Config.DATA_FILE); y_all = df[Config.TARGET_COL]
    X_all = df.drop(columns=[Config.TARGET_COL, 'ID'])
    
    # 严格遵循防泄露原则：先拆分数据 (使用全局一致的 seed 和 stratify)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED, stratify=y_all
    )
    
    # 然后进行安全插补
    X_train, X_test = robust_impute(X_train_raw, X_test_raw, Config.RANDOM_SEED)
    X_imputed, y = X_train, y_train


    # ==============================================================================
    # 全新特征重要性及前向选择交叉验证可视化块
    # ==============================================================================
    print("运行特征重要性排名与前向选择验证评估...")
    import lightgbm as lgb
    import scipy.stats as stats
    import matplotlib.colors as mcolors

    # 1. 计算重要性并画 Top30 柱状图
    lgbm_base = lgb.LGBMClassifier(random_state=Config.RANDOM_SEED, verbose=-1, n_jobs=-1)
    lgbm_base.fit(X_imputed, y)
    feature_imp_df = pd.DataFrame({
        'Feature': X_imputed.columns,
        'Importance': lgbm_base.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    top_n_bar = min(30, len(feature_imp_df))
    top_features_bar = feature_imp_df.head(top_n_bar)
    
    plt.figure(figsize=(12, 8), dpi=300)
    plt.barh(top_features_bar['Feature'], top_features_bar['Importance'], color='skyblue')
    plt.xlabel('Importance', fontsize=14); plt.ylabel('Feature', fontsize=14)
    plt.title(f'Top {top_n_bar} Feature Importance', fontsize=16)
    plt.gca().invert_yaxis(); plt.savefig("Top_30_Feature_Importance.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # 2. 前向选择交叉验证 (最多取前30个做递增验证)
    top_features_eval = feature_imp_df.head(30)
    selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
    selected_features_eval = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
    fold_columns = [f'Fold_{i+1}_ROC' for i in range(kf.get_n_splits())]

    for i in range(len(top_features_eval)):
        curr_feat = top_features_eval.iloc[i]['Feature']
        selected_features_eval.append(curr_feat)
        fold_roc_scores = []
        for train_idx, val_idx in kf.split(X_imputed):
            X_tr_f, X_val_f = X_imputed.iloc[train_idx][selected_features_eval], X_imputed.iloc[val_idx][selected_features_eval]
            y_tr_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
            
            clf_fold = lgb.LGBMClassifier(random_state=Config.RANDOM_SEED, verbose=-1)
            clf_fold.fit(X_tr_f, y_tr_f)
            fold_roc_scores.append(roc_auc_score(y_val_f, clf_fold.predict_proba(X_val_f)[:, 1]))
            
        row_data = {
            'Feature': curr_feat, 'Importance': top_features_eval.iloc[i]['Importance'],
            'Mean_ROC': np.mean(fold_roc_scores)
        }
        for j, score in enumerate(fold_roc_scores): row_data[fold_columns[j]] = score
        selection_results = pd.concat([selection_results, pd.DataFrame([row_data])], ignore_index=True)

    # 3. 计算置信区间并归一化画图
    selection_results['Importance'] = pd.to_numeric(selection_results['Importance'], errors='coerce')
    if selection_results['Importance'].sum() > 0:
        selection_results['Importance'] = selection_results['Importance'] / selection_results['Importance'].sum()
        
    selection_results['CI_Lower'], selection_results['CI_Upper'] = None, None
    for idx, row in selection_results.iterrows():
        f_scores = [row[f] for f in fold_columns]
        m_roc = row['Mean_ROC']; se = stats.sem(f_scores); t_val = stats.t.ppf(0.975, df=len(f_scores)-1)
        selection_results.at[idx, 'CI_Lower'] = m_roc - t_val * se
        selection_results.at[idx, 'CI_Upper'] = min(1.0, m_roc + t_val * se)

    selection_results['CI_Lower'] = pd.to_numeric(selection_results['CI_Lower'], errors='coerce')
    selection_results['CI_Upper'] = pd.to_numeric(selection_results['CI_Upper'], errors='coerce')

    n_features = min(Config.RFE_N_FEATURES, len(selection_results)) # 当前命中的特征数
    fig, ax1 = plt.subplots(figsize=(16, 6), dpi=300)
    norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
    colors = plt.cm.Blues(norm(selection_results['Importance']))

    ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
    ax1.set_xlabel("Features", fontsize=18, fontweight='bold'); ax1.set_ylabel("Feature Importance", fontsize=18, fontweight='bold')
    
    x_labels = selection_results['Feature'].tolist()
    x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45, fontsize=12)
    for tick_label, color in zip(ax1.get_xticklabels(), x_colors): tick_label.set_color(color)

    ax2 = ax1.twinx()
    ax2.plot(selection_results['Feature'][:n_features + 1], selection_results['Mean_ROC'][:n_features + 1], color="red", marker='o', linestyle='-', label="Mean AUC (Top Features)")
    ax2.plot(selection_results['Feature'][max(0, n_features-1):], selection_results['Mean_ROC'][max(0, n_features-1):], color="black", marker='o', linestyle='-', label="Mean AUC (Other Features)")
    ax2.fill_between(selection_results['Feature'], selection_results['CI_Lower'], selection_results['CI_Upper'], color='red', alpha=0.2)
    ax2.set_ylabel("Mean AUC", fontsize=18, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    plt.title(f"Feature Contribution and AUC Performance (Top {n_features} Highlighted)", fontsize=18, fontweight='bold')
    fig.tight_layout(); plt.savefig("Feature_Selection_Validation.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    FINAL_SELECTED_FEATURES = x_labels[:n_features]


    features = FINAL_SELECTED_FEATURES
    print(f"🏆 Final Selected Features: {features}")
    scaler = StandardScaler(); X_train_s = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features)
    X_test_s = pd.DataFrame(scaler.transform(X_test[features]), columns=features)

    # ================= 扩充至完整 9 大模型池 =================
    models = {
        'ANN':  (MLPClassifier(max_iter=500, random_state=Config.RANDOM_SEED), {'hidden_layer_sizes': [(50,), (100, 50)], 'learning_rate_init': [0.001, 0.01]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=Config.RANDOM_SEED), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}),
        'Extra Trees': (ExtraTreesClassifier(random_state=Config.RANDOM_SEED), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=Config.RANDOM_SEED), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 10]}),
        'LightGBM': (lgb.LGBMClassifier(random_state=Config.RANDOM_SEED, verbose=-1, n_jobs=-1), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}),
        'Random Forest': (RandomForestClassifier(random_state=Config.RANDOM_SEED), {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}),
        'SVM': (SVC(probability=True, random_state=Config.RANDOM_SEED), {'C': [0.1, 1, 10]}),
        'XGBoost': (XGBClassifier(eval_metric='logloss', random_state=Config.RANDOM_SEED, n_jobs=-1), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]})
    }
    
    scaled_models = {'ANN', 'SVM', 'KNN'}; res = {}; best_params_list = []
    
    for name, (m, p) in models.items():
        xt, xv = (X_train_s, X_test_s) if name in scaled_models else (X_train[features], X_test[features])
        g = GridSearchCV(m, p, cv=5, scoring='neg_log_loss', n_jobs=-1).fit(xt, y_train)
        best_mod = g.best_estimator_
        res[name] = {'mod': best_mod, 'ptr': best_mod.predict_proba(xt)[:, 1], 'pte': best_mod.predict_proba(xv)[:, 1], 'xt': xt, 'xv': xv}
        best_params_list.append({'Model': name, **g.best_params_})

    # ================================================================
    # 可视化系列 I: 为每个模型绘制带 Total 的完美混淆矩阵 (18张)
    # ================================================================
    def draw_confusion_matrix(y_true, y_pred, model_name, dataset_type):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_tot = np.vstack([cm, cm.sum(axis=0)]); cm_tot = np.column_stack([cm_tot, cm_tot.sum(axis=1)])
        labels = ['Negative', 'Positive', 'Total']
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        heatmap_data = np.pad(cm_norm, ((0,1), (0,1)), mode='constant')
        sns.heatmap(heatmap_data, annot=False, cmap=sns.color_palette("Greens", as_cmap=True), 
                    xticklabels=labels, yticklabels=labels, cbar=False, square=True, linewidths=1.5, linecolor="white", ax=ax)
        
        grey_color = "#f0f0f0"
        for i in range(cm.shape[0]): ax.add_patch(plt.Rectangle((cm.shape[1], i), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
        for j in range(cm.shape[1]): ax.add_patch(plt.Rectangle((j, cm.shape[0]), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
        ax.add_patch(plt.Rectangle((cm.shape[1], cm.shape[0]), 1, 1, fill=True, color=grey_color, edgecolor="white", lw=1.5))
        
        for i in range(cm_tot.shape[0]):
            for j in range(cm_tot.shape[1]):
                if i < cm.shape[0] and j < cm.shape[1]:
                    ax.text(j+0.5, i+0.5, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center", fontsize=14, color="black")
                    ax.text(j+0.5, i+0.65, f"{cm[i,j]}", ha="center", va="center", fontsize=12, color="black")
                else:
                    tot_val = cm_tot[i, j]
                    if i == cm.shape[0] and j == cm.shape[1]:
                        ax.text(j+0.5, i+0.5, f"{tot_val}", ha="center", va="center", fontsize=14, color="black")
                    else:
                        ax.text(j+0.5, i+0.5, f"{(tot_val/cm_tot[-1,-1])*100:.1f}%", ha="center", va="center", fontsize=14, color="black")
                        ax.text(j+0.5, i+0.65, f"{tot_val}", ha="center", va="center", fontsize=12, color="black")
        
        plt.title(f"{model_name}", fontsize=20); plt.xlabel("Prediction (model output)", fontsize=16); plt.ylabel("Truth (observation)", fontsize=16)
        plt.tight_layout(); plt.savefig(f"CM_{model_name}_{dataset_type}.pdf", format='pdf', bbox_inches='tight'); plt.close()

    for n, r in res.items():
        draw_confusion_matrix(y_train, (r['ptr'] >= 0.5).astype(int), n, 'Train')
        draw_confusion_matrix(y_test, (r['pte'] >= 0.5).astype(int), n, 'Test')

    # ================================================================
    # 优化可视化系列 II: 模型性能评价折线图 (合并为1张带双子图)
    # ================================================================
    def calculate_metrics_raw(y_true, y_prob):
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
        ppv = precision_score(y_true, y_pred, zero_division=0)
        npv = cm[0,0]/(cm[0,0]+cm[1,0]) if (cm[0,0]+cm[1,0])>0 else 0
        f1 = f1_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        return [acc, sens, spec, ppv, npv, f1, kappa]

    m_names = ['accuracy', 'sensitivity', 'specificity', 'Positive predictive value', 'Negative predictive value', 'F1 score', 'Kappa score']
    dict_te = {'Metrics': m_names}; dict_tr = {'Metrics': m_names}
    for n, r in res.items():
        dict_te[n] = calculate_metrics_raw(y_test, r['pte'])
        dict_tr[n] = calculate_metrics_raw(y_train, r['ptr'])
    df_met_te = pd.DataFrame(dict_te); df_met_tr = pd.DataFrame(dict_tr)

    import matplotlib.ticker as ticker
    def plot_metrics_line_combined(df_tr, df_te, fname):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
        colors = plt.cm.get_cmap('tab20', len(df_tr.columns) - 1)
        
        for ax, df, title in zip(axes, [df_tr, df_te], ["Training Set Metrics", "Test Set Metrics"]):
            for idx, model in enumerate(df.columns[1:]): 
                ax.plot(df['Metrics'], df[model], marker='o', label=model, color=colors(idx), linewidth=2)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
            ax.set_title(title, fontsize=18, fontweight="bold")
            ax.set_ylim(0.1, 1.0)
            ax.set_yticks([0.25, 0.50, 0.75, 1.00])
            
            ax.set_xticks(range(len(df['Metrics'])))
            ax.set_xticklabels(df['Metrics'], rotation=45, ha='right', fontsize=14)
            ax.grid(which='both', linestyle='-', linewidth=0.5, color='gray')
            
        axes[1].legend(title="Models", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12)
        plt.tight_layout()
        plt.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close()

    plot_metrics_line_combined(df_met_tr, df_met_te, "Metrics_Line_Combined.pdf")

    # ================================================================
    # 优化可视化系列 III: PCA 结合动态约登指数最佳预测阈值的散点图
    # ================================================================
    best_n = max(res.keys(), key=lambda k: roc_auc_score(y_test, res[k]['pte']))
    best_obj = res[best_n]
    
    # 【核心修改】：通过训练集计算 Youden's J 来确定真正的最佳预测阈值
    fpr_train_opt, tpr_train_opt, thresholds_train_opt = roc_curve(y_train, best_obj['ptr'])
    optimal_idx = np.argmax(tpr_train_opt - fpr_train_opt)
    optimal_threshold = thresholds_train_opt[optimal_idx]

    from sklearn.decomposition import PCA
    def draw_pca_scatter(X_data, y_true, y_prob, title, fname):
        pca = PCA(n_components=1); pca_res = pca.fit_transform(X_data); expl_var = pca.explained_variance_ratio_[0] * 100
        df_pca = pd.DataFrame({'Prob_Class_1': y_prob, f'PC1 ({expl_var:.2f}%)': pca_res[:,0], 'True': y_true.values})
        colors_map = {0: '#A0D6B4', 1: '#C3A6D8'}
        colors_arr = df_pca['True'].map(colors_map)
        plt.figure(figsize=(10, 6), dpi=300)
        plt.scatter(df_pca['Prob_Class_1'], df_pca[f'PC1 ({expl_var:.2f}%)'], c=colors_arr, edgecolor='black', s=100)
        
        # 换用上面动态计算出的最佳阈值
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold ({optimal_threshold:.2f})')
        
        plt.xlabel('Predicted Probability for Positive (Class 1)'); plt.ylabel(f'PC1 ({expl_var:.2f}%)'); plt.title(title)
        
        for t_val, c_val, lab in [(0, '#A0D6B4', 'Negative'), (1, '#C3A6D8', 'Positive')]: 
            plt.scatter([], [], color=c_val, edgecolor='black', s=100, label=lab)
        plt.legend(title="Class", loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(fname, format='pdf', bbox_inches='tight'); plt.close()

    draw_pca_scatter(best_obj['xv'], y_test, best_obj['pte'], 'Test set PCA', 'PCA_Scatter_Test.pdf')
    draw_pca_scatter(best_obj['xt'], y_train, best_obj['ptr'], 'Train set PCA', 'PCA_Scatter_Train.pdf')

    # ================================================================
    # 彻底优化的 ROC 曲线（移除平滑，修复Train CI显示，修改排版边框）
    # ================================================================
    for mode in ['Clean', 'CI']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        for n, r in res.items():
            f1, t1, _ = roc_curve(y_train, r['ptr'])
            f2, t2, _ = roc_curve(y_test, r['pte'])
            
            # 直接使用原生的 fpr, tpr，移除所有 smooth 插值干扰
            ax1.plot(f1, t1, lw=2, label=f"{n} (AUC={roc_auc_score(y_train, r['ptr']):.3f})")
            ax2.plot(f2, t2, lw=2, label=f"{n} (AUC={roc_auc_score(y_test, r['pte']):.3f})")
            
            if mode == 'CI':
                # 【修改】：为 Train 补充完整的 Bootstrap CI 计算
                tprs_b_tr = []; base_f_tr = np.linspace(0, 1, 100)
                for _ in range(100):
                    idx_tr = np.random.choice(len(y_train), size=len(y_train), replace=True)
                    if len(np.unique(y_train.values[idx_tr])) < 2: continue
                    fb_tr, tb_tr, _ = roc_curve(y_train.values[idx_tr], r['ptr'][idx_tr])
                    tprs_b_tr.append(np.interp(base_f_tr, fb_tr, tb_tr))
                if tprs_b_tr: 
                    ax1.fill_between(base_f_tr, np.percentile(tprs_b_tr, 2.5, axis=0), np.percentile(tprs_b_tr, 97.5, axis=0), alpha=0.1)
                
                # 保留原有的 Test CI 计算
                tprs_b_te = []; base_f_te = np.linspace(0, 1, 100)
                for _ in range(100):
                    idx_te = np.random.choice(len(y_test), size=len(y_test), replace=True)
                    if len(np.unique(y_test.values[idx_te])) < 2: continue
                    fb_te, tb_te, _ = roc_curve(y_test.values[idx_te], r['pte'][idx_te])
                    tprs_b_te.append(np.interp(base_f_te, fb_te, tb_te))
                if tprs_b_te: 
                    ax2.fill_between(base_f_te, np.percentile(tprs_b_te, 2.5, axis=0), np.percentile(tprs_b_te, 97.5, axis=0), alpha=0.1)

        # 样式细节统一应用
        for ax, title in zip([ax1, ax2], [f'Train ROC ({mode})', f'Test ROC ({mode})']):
            ax.plot([0,1], [0,1], 'r--', linewidth=1.5, alpha=0.8) # 红色虚线表示随机分类
            ax.set_title(title, fontsize=20, fontweight="bold")
            ax.set_xlabel("False Positive Rate (1-Specificity)", fontsize=18)
            ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.legend(loc="lower right", fontsize=12)
            
            # 移除上右边框，加粗左下边框，关闭网格
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(False)

        plt.tight_layout(); plt.savefig(f'ROC_Curves_{mode}.pdf', format='pdf', bbox_inches='tight'); plt.close()

    # ---------------- 以下保留所有原版代码图表满级分析模块 ----------------
    print("🔬 执行 Nested Cross-Validation...")
    nested_summary = []
    for name, (m, p) in models.items():
        xt = X_train_s if name in scaled_models else X_train[features]
        inner_cv = GridSearchCV(m, p, cv=3, scoring='roc_auc')
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
        nested_scores = cross_val_score(inner_cv, xt, y_train, cv=outer_cv, n_jobs=-1)
        nested_summary.append({'Model': name, 'Nested_AUC_Mean': np.mean(nested_scores), 'Nested_AUC_Std': np.std(nested_scores)})

    plt.figure(figsize=(12, 6)); n_df = pd.DataFrame(nested_summary)
    sns.barplot(x='Model', y='Nested_AUC_Mean', data=n_df, palette='magma')
    plt.errorbar(x=range(len(n_df)), y=n_df['Nested_AUC_Mean'], yerr=n_df['Nested_AUC_Std'], fmt='none', c='black', capsize=5)
    plt.title('Nested Cross-Validation (Generalization Stability)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('Nested_CV_Performance.pdf'); plt.close()

    print("🔬 执行 Repeated Cross-Validation...")
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=Config.RANDOM_SEED)
    rep_scores = cross_val_score(best_obj['mod'], best_obj['xt'], y_train, cv=rkf, scoring='roc_auc', n_jobs=-1)
    plt.figure(figsize=(10, 6)); plt.plot(rep_scores, marker='o', linestyle='-', color='#6A9ACE', alpha=0.7)
    plt.axhline(np.mean(rep_scores), color='red', linestyle='--', label=f'Mean={np.mean(rep_scores):.3f}')
    plt.title(f'Repeated CV Stability ({best_n})', fontsize=14, fontweight='bold'); plt.ylabel('AUC'); plt.legend(); plt.savefig('Repeated_CV_Stability.pdf'); plt.close()

    print("🔬 执行 Bootstrap Optimism Correction...")
    def calc_optimism(X, y, model, n_boot=100):
        auc_app = roc_auc_score(y, model.predict_proba(X)[:, 1]); opts = []
        for _ in range(n_boot):
            idx = np.random.choice(len(X), size=len(X), replace=True); X_b, y_b = X.iloc[idx], y.iloc[idx]
            if len(np.unique(y_b)) < 2: continue
            model.fit(X_b, y_b); opts.append(roc_auc_score(y_b, model.predict_proba(X_b)[:, 1]) - roc_auc_score(y, model.predict_proba(X)[:, 1]))
        return auc_app, np.mean(opts)
    
    auc_apparent, optimism = calc_optimism(best_obj['xt'], y_train, best_obj['mod'])
    auc_corrected = auc_apparent - optimism
    plt.figure(figsize=(6, 8)); plt.plot(['Apparent', 'Corrected'], [auc_apparent, auc_corrected], marker='o', markersize=15, color='darkorange', linewidth=4)
    plt.text(0, auc_apparent+0.005, f'{auc_apparent:.3f}', ha='center', weight='bold'); plt.text(1, auc_corrected+0.005, f'{auc_corrected:.3f}', ha='center', weight='bold')
    plt.title('Bootstrap Optimism Correction', fontsize=14, fontweight='bold'); plt.ylim(auc_corrected-0.05, auc_apparent+0.05); plt.grid(axis='y', alpha=0.3); plt.savefig('Optimism_Correction_Slope.pdf'); plt.close()

    # DCA, Calibration 等图表
    for mode in ['Clean', 'CI']:
        fig, (dx1, dx2) = plt.subplots(1, 2, figsize=(16, 8)); thresh = np.linspace(0.01, 0.99, 100)
        nb_all_tr = calc_net_benefit(y_train.values, np.ones(len(y_train)), thresh)
        nb_all_te = calc_net_benefit(y_test.values, np.ones(len(y_test)), thresh)
        dx1.plot(thresh, nb_all_tr, color='gray', lw=2, label='Treat All'); dx2.plot(thresh, nb_all_te, color='gray', lw=2, label='Treat All')
        dx1.plot([0,1], [0,0], color='black', lw=2, linestyle='--', label='Treat None'); dx2.plot([0,1], [0,0], color='black', lw=2, linestyle='--', label='Treat None')
        for n, r in res.items():
            dx1.plot(thresh, calc_net_benefit(y_train.values, r['ptr'], thresh), lw=2, label=n)
            dx2.plot(thresh, calc_net_benefit(y_test.values, r['pte'], thresh), lw=2, label=n)
        dx1.set_ylim([-0.05, max(nb_all_tr)+0.05]); dx2.set_ylim([-0.05, max(nb_all_te)+0.05])
        dx1.set_xlabel('Threshold Probability'); dx1.set_ylabel('Net Benefit'); dx1.set_title(f'Train DCA ({mode})'); dx2.set_title(f'Test DCA ({mode})')
        dx1.legend(loc='upper right'); dx2.legend(loc='upper right'); plt.tight_layout(); plt.savefig(f'DCA_Curves_{mode}.pdf'); plt.close()

        fig, (c1, c2) = plt.subplots(1, 2, figsize=(16, 7))
        for n in res.keys():
            ytr_p, yte_p = res[n]['ptr'], res[n]['pte']; pt1, pp1 = calibration_curve(y_train, ytr_p, n_bins=10)
            pt2, pp2 = calibration_curve(y_test, yte_p, n_bins=10); b1, l1, u1 = brier_score_confidence_interval(y_train, ytr_p)
            b2, l2, u2 = brier_score_confidence_interval(y_test, yte_p)
            c1.plot(pp1, pt1, marker='o', label=f"{n} ({b1:.3f})"); c2.plot(pp2, pt2, marker='o', label=f"{n} ({b2:.3f})")
        c1.plot([0,1],[0,1],'k--'); c2.plot([0,1],[0,1],'k--'); c1.set_title(f'Calibration Train ({mode})'); c2.set_title(f'Calibration Test ({mode})')
        c1.legend(fontsize=8); c2.legend(fontsize=8); plt.tight_layout(); plt.savefig(f'Calibration_Curves_{mode}.pdf'); plt.close()

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(14, 6))
    PrecisionRecallDisplay.from_estimator(best_obj['mod'], best_obj['xt'], y_train, plot_chance_level=True, name=best_n, ax=p1)
    PrecisionRecallDisplay.from_estimator(best_obj['mod'], best_obj['xv'], y_test, plot_chance_level=True, name=best_n, ax=p2)
    p1.set_title("PR Curve (Train)"); p2.set_title("PR Curve (Test)"); plt.tight_layout(); plt.savefig('PR_Curves_Best_Model.pdf'); plt.close()

    # SHAP 图表
    explainer = shap.TreeExplainer(best_obj['mod']) if best_n in ['Random Forest', 'XGBoost', 'Extra Trees', 'LightGBM', 'Gradient Boosting'] else shap.KernelExplainer(best_obj['mod'].predict_proba, shap.kmeans(best_obj['xv'], 10))
    sv = explainer.shap_values(best_obj['xv'])
    if isinstance(sv, list): sv = sv[1]
    if hasattr(sv, 'shape') and len(sv.shape) == 3: sv = sv[:, :, 1]
    
    plt.figure(); shap.summary_plot(sv, best_obj['xv'], show=False); plt.savefig('SHAP_Summary_Dot.pdf', bbox_inches='tight'); plt.close()
    plt.figure(); shap.summary_plot(sv, best_obj['xv'], plot_type='bar', show=False); plt.savefig('SHAP_Importance_Bar.pdf', bbox_inches='tight'); plt.close()
    
    shap_df = pd.DataFrame(np.abs(sv).mean(axis=0), index=features, columns=['Mean_SHAP']).reset_index().rename(columns={'index': 'Feature'})
    
    shap_df_l = pd.DataFrame(sv, columns=features); rows_l = int(np.ceil(len(features) / 3))
    fig, axes = plt.subplots(rows_l, 3, figsize=(15, 5 * rows_l))
    for i, f in enumerate(features):
        ax = axes.ravel()[i]; ax.scatter(best_obj['xv'][f], shap_df_l[f], s=15, alpha=0.5, color="#6A9ACE")
        lw_f = lowess(shap_df_l[f], best_obj['xv'][f], frac=0.3); ax.plot(lw_f[:, 0], lw_f[:, 1], color='red', lw=2); ax.set_xlabel(f)
    for i in range(len(features), len(axes.ravel())): axes.ravel()[i].axis('off')
    plt.tight_layout(); plt.savefig('SHAP_LOWESS_Best_Model.pdf'); plt.close()

    # 输出完整的 9 个子表 Excel
    met_list = []
    for n, r in res.items():
        met_list.append({
            'Model': n,
            'AUC_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'AUC'), 'AUC_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'AUC'),
            'ACC_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'ACC'), 'ACC_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'ACC'),
            'SENS_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'SENS'), 'SENS_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'SENS'),
            'SPEC_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'SPEC'), 'SPEC_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'SPEC'),
            'F1_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'F1'), 'F1_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'F1')
        })
    p_comp = []; names_list = list(res.keys())
    for m1, m2 in itertools.combinations(names_list, 2):
        pval = bootstrap_auc_pvalue(y_test, res[m1]['pte'], res[m2]['pte'], seed=Config.RANDOM_SEED); sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        p_comp.append({'Comparison': f'{m1} vs {m2}', 'P_Value': pval, 'Sig': sig})

    with pd.ExcelWriter('Final_Tables.xlsx') as writer:
        generate_baseline_table(df, Config.TARGET_COL, 'ID').to_excel(writer, sheet_name='Baseline_Characteristics', index=False)
        pd.DataFrame(met_list).to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame(p_comp).to_excel(writer, sheet_name='Model_Comparison', index=False)
        pd.DataFrame(best_params_list).to_excel(writer, sheet_name='Hyperparameters', index=False)
        pd.DataFrame([{'Model': n, 'Brier_Test': brier_score_loss(y_test, r['pte'])} for n, r in res.items()]).to_excel(writer, sheet_name='Brier_Score', index=False)
        shap_df.to_excel(writer, sheet_name='SHAP_Importance', index=False)
        pd.DataFrame(nested_summary).to_excel(writer, sheet_name='Nested_CV_Results', index=False)
        pd.DataFrame({'Repetition': range(len(rep_scores)), 'AUC': rep_scores}).to_excel(writer, sheet_name='Repeated_CV_Results', index=False)
        pd.DataFrame([{'Apparent_AUC': auc_apparent, 'Optimism': optimism, 'Corrected_AUC': auc_corrected}]).to_excel(writer, sheet_name='Optimism_Correction', index=False)

    # ================= APP部署与模型保存 =================
    app_str = f"""import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title("Clinic Predictor")
model = joblib.load('saved_models/{{best_n}}_best.pkl')
FEATURES = {features!r}
df = pd.read_excel('Final_Cleaned_Data.xlsx')
X_f = df.drop(columns=['{Config.TARGET_COL}', 'ID'], errors='ignore')

# 保护机制：为Streamlit后台可解释性背景数据进行快速填补，防止SHAP报错
for c in X_f.columns:
    if pd.api.types.is_numeric_dtype(X_f[c]):
        X_f[c].fillna(X_f[c].median(), inplace=True)
    else:
        X_f[c].fillna(X_f[c].mode()[0], inplace=True)

input_vals = []
for f in FEATURES:
    if pd.api.types.is_numeric_dtype(X_f[f]):
        v = st.number_input(f"{{f}}", float(X_f[f].min()), float(X_f[f].max()), float(X_f[f].median()))
    else:
        v = st.selectbox(f"{{f}}", X_f[f].unique().tolist())
    input_vals.append(v)

if st.button("Predict"):
    X_in = pd.DataFrame([input_vals], columns=FEATURES)
    prob = model.predict_proba(X_in)[0][1]
    st.write(f"### Probability: {{prob*100:.2f}}%")
    
    explainer = shap.Explainer(model, X_f[FEATURES])
    sv_in = explainer(X_in)
    fig = plt.figure()
    shap.plots.waterfall(sv_in[0], show=False)
    st.pyplot(fig)
"""
    with open('APP.py', 'w', encoding='utf-8') as f:
        f.write(app_str.replace('{{best_n}}', best_n))

    os.makedirs('saved_models', exist_ok=True)
    for n, r in res.items(): 
        joblib.dump(r['mod'], f'saved_models/{{n}}_best.pkl')

    print("🎉 All analysis tasks completed!")

if __name__ == '__main__': main()
