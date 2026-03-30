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
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, recall_score,
                             f1_score, brier_score_loss, PrecisionRecallDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from statsmodels.nonparametric.smoothers_lowess import lowess

# Imputation modules for leakage-safe preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'DejaVu Sans'

class Config:
    DATA_FILE = 'Final_Cleaned_Data.xlsx'
    TARGET_COL = 'status'
    PSA_COL = 'tPSA'
    RANDOM_SEED = 197
    TEST_SIZE = 0.3
    RFE_MODE = 'top_n'
    RFE_N_FEATURES = 4

os.chdir(r'/home/lin/hz/翻修')

def generate_baseline_table(df, target_col, id_col):
    stats_list = []
    groups = sorted(df[target_col].dropna().unique())
    df_groups = {g: df[df[target_col] == g] for g in groups}
    is_bin = len(groups) == 2  # Binary grouping flag for group-wise hypothesis testing

    for col in df.columns:
        if col in [target_col, id_col]:
            continue

        # Group comparison p-value for binary outcome settings
        pval_str = ""
        if is_bin:
            g1 = df_groups[groups[0]][col].dropna()
            g2 = df_groups[groups[1]][col].dropna()
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
                if len(g1) > 0 and len(g2) > 0:
                    _, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test for continuous variables
                    pval_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
            else:
                try:
                    _, p, _, _ = stats.chi2_contingency(pd.crosstab(df[col], df[target_col]))  # Chi-square test for categorical variables
                    pval_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                except:
                    pval_str = "NaN"

        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5:
            m_all, s_all = df[col].mean(), df[col].std()
            q1_all, med_all, q3_all = df[col].quantile([0.25, 0.5, 0.75])
            row_mean = {
                'Feature': f'{col} (Mean ± SD)',
                'Overall': f'{m_all:.2f} ± {s_all:.2f}',
                'P_Value': pval_str
            }
            row_med = {
                'Feature': f'{col} (Median [IQR])',
                'Overall': f'{med_all:.2f} [{q1_all:.2f}-{q3_all:.2f}]',
                'P_Value': ''
            }
            for g in groups:
                row_mean[f'Group {g}'] = f'{df_groups[g][col].mean():.2f} ± {df_groups[g][col].std():.2f}'
                q1g, medg, q3g = df_groups[g][col].quantile([0.25, 0.5, 0.75])
                row_med[f'Group {g}'] = f'{medg:.2f} [{q1g:.2f}-{q3g:.2f}]'
            stats_list.extend([row_mean, row_med])
        else:
            vc = df[col].value_counts(dropna=False)
            first_val = True
            for val in vc.index:
                c_all = vc[val]
                p_all = c_all / len(df) * 100
                row_cat = {
                    'Feature': f'{col} = {val}',
                    'Overall': f'{c_all} ({p_all:.1f}%)',
                    'P_Value': pval_str if first_val else ''
                }
                first_val = False
                for g in groups:
                    c_g = df_groups[g][col].value_counts(dropna=False).get(val, 0)
                    p_g = (c_g / len(df_groups[g]) * 100) if len(df_groups[g]) > 0 else 0
                    row_cat[f'Group {g}'] = f'{c_g} ({p_g:.1f}%)'
                stats_list.append(row_cat)
    return pd.DataFrame(stats_list)

def get_bootstrap_metrics_ci(y_true, y_prob, metric_func, n_bootstraps=1000):
    values = []
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        yt, yp = y_true[idx], y_prob[idx]
        ypr = (yp >= 0.5).astype(int)

        if metric_func == 'AUC':
            values.append(roc_auc_score(yt, yp))
        elif metric_func == 'ACC':
            values.append(accuracy_score(yt, ypr))
        elif metric_func == 'SENS':
            values.append(recall_score(yt, ypr))
        elif metric_func == 'SPEC':
            values.append(recall_score(yt, ypr, pos_label=0))
        elif metric_func == 'F1':
            values.append(f1_score(yt, ypr))

    if not values:
        return "NaN"
    return f"{np.mean(values):.3f} ({np.percentile(values, 2.5):.3f}-{np.percentile(values, 97.5):.3f})"

def brier_score_confidence_interval(y_true, y_pred_prob, n_bootstraps=1000):
    scores = []
    yt = np.array(y_true)
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(yt), size=len(yt), replace=True)
        scores.append(np.mean((y_pred_prob[idx] - yt[idx]) ** 2))
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def bootstrap_auc_pvalue(y_true, p_a, p_b, seed=42, n_boot=2000):
    rng = np.random.RandomState(seed)
    y_true, p_a, p_b = np.array(y_true), np.array(p_a), np.array(p_b)
    orig_diff = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    return float(np.mean(np.abs(diffs) >= np.abs(orig_diff)))

def calc_net_benefit(y_true, y_prob, thresholds):
    nb = []
    for t in thresholds:
        tp = np.sum((y_true == 1) & (y_prob >= t))
        fp = np.sum((y_true == 0) & (y_prob >= t))
        nb.append((tp / len(y_true)) - (fp / len(y_true)) * (t / (1 - t)))
    return np.array(nb)

def robust_impute(X_train, X_test, seed):
    """
    Leakage-safe imputation:
    fit imputers on the training set only, then transform training and test sets separately.
    """
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

def smooth_curve(fpr, tpr):
    if len(fpr) < 4:
        return fpr, tpr
    x = np.linspace(0, 1, 300)
    u_fpr, idx = np.unique(fpr, return_index=True)
    return x, np.clip(PchipInterpolator(u_fpr, tpr[idx])(x), 0, 1)

def main():
    print("🚀 Golden Egg Science-Pro Script Running...")

    df = pd.read_excel(Config.DATA_FILE)
    y_all = df[Config.TARGET_COL]
    X_all = df.drop(columns=[Config.TARGET_COL, 'ID'])

    # Data splitting is performed before imputation to avoid information leakage.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y_all
    )

    # Imputation is fitted on the training set and applied to both partitions.
    X_train, X_test = robust_impute(X_train_raw, X_test_raw, Config.RANDOM_SEED)
    X_imputed, y = X_train, y_train

    print(f"⏳ 正在运行 RFE 递归筛选 (模式: {Config.RFE_MODE})...")
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=Config.RANDOM_SEED)
    selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_ROC'])
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
    n_splits = kf.get_n_splits()
    fold_columns = [f'Fold_{i+1}_ROC' for i in range(n_splits)]

    rfe = RFE(estimator=rf_clf, n_features_to_select=1, step=1)
    rfe.fit(X_imputed, y)
    rfe_features = pd.DataFrame({'Feature': X_imputed.columns, 'Ranking': rfe.ranking_}).sort_values(by='Ranking')

    selected_features = []
    for i in range(len(rfe_features)):
        current_feature = rfe_features.iloc[i]['Feature']
        selected_features.append(current_feature)
        fold_roc_scores = []
        for train_idx, val_idx in kf.split(X_imputed):
            model = RandomForestClassifier(n_jobs=-1, random_state=Config.RANDOM_SEED)
            model.fit(X_imputed.iloc[train_idx][selected_features], y.iloc[train_idx])
            fold_roc_scores.append(
                roc_auc_score(
                    y.iloc[val_idx],
                    model.predict_proba(X_imputed.iloc[val_idx][selected_features])[:, 1]
                )
            )

        mean_roc_score = np.mean(fold_roc_scores)
        model.fit(X_imputed[selected_features], y)
        row_data = {
            'Feature': current_feature,
            'Importance': model.feature_importances_[-1],
            'Mean_ROC': mean_roc_score
        }
        for j, score in enumerate(fold_roc_scores):
            row_data[fold_columns[j]] = score
        selection_results = pd.concat([selection_results, pd.DataFrame([row_data])], ignore_index=True)

    for index, row in selection_results.iterrows():
        fold_scores = [row[col] for col in fold_columns]
        std_err = stats.sem(fold_scores)
        t_val = stats.t.ppf(0.975, df=n_splits - 1)
        selection_results.at[index, 'CI_Lower'] = row['Mean_ROC'] - t_val * std_err
        selection_results.at[index, 'CI_Upper'] = row['Mean_ROC'] + t_val * std_err

    n_highlight = Config.RFE_N_FEATURES
    rfe_selected_cols = selection_results['Feature'].head(n_highlight).tolist()

    fig, ax1 = plt.subplots(figsize=(16, 8))
    norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
    ax1.bar(
        selection_results['Feature'],
        selection_results['Importance'],
        color=plt.cm.Blues(norm(selection_results['Importance'].values.astype(float))),
        alpha=0.8
    )

    x_labels = selection_results['Feature'].tolist()
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=90, fontsize=9)
    for i, label in enumerate(ax1.get_xticklabels()):
        if x_labels[i] in rfe_selected_cols:
            label.set_color('red')

    ax2 = ax1.twinx()
    ax2.plot(selection_results['Feature'][:n_highlight], selection_results['Mean_ROC'][:n_highlight], color="#E53935", marker='o', label="Selected Area")
    ax2.plot(selection_results['Feature'][n_highlight-1:], selection_results['Mean_ROC'][n_highlight-1:], color="black", marker='o')
    ax2.fill_between(
        selection_results['Feature'],
        selection_results['CI_Lower'].astype(float),
        selection_results['CI_Upper'].astype(float),
        color='#E53935',
        alpha=0.15
    )

    plt.title(f"RFE Performance: Top {n_highlight} Features Red-Highlighted", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('RFE_Plot.pdf', dpi=300)
    FINAL_SELECTED_FEATURES = rfe_selected_cols

    features = FINAL_SELECTED_FEATURES
    print(f"🏆 Final Selected Features: {features}")
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features)
    X_test_s = pd.DataFrame(scaler.transform(X_test[features]), columns=features)

    models = {
        'LR':  (LogisticRegression(max_iter=3000, random_state=Config.RANDOM_SEED), {'C': [0.1, 1, 10]}),
        'RF':  (RandomForestClassifier(random_state=Config.RANDOM_SEED), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
        'XGB': (XGBClassifier(random_state=Config.RANDOM_SEED, eval_metric='logloss'), {'learning_rate': [0.01, 0.1]}),
        'SVM': (SVC(probability=True, random_state=Config.RANDOM_SEED), {'C': [1, 10]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5]}),
        'MLP': (MLPClassifier(random_state=Config.RANDOM_SEED, max_iter=3000), {'hidden_layer_sizes': [(64,), (64, 32)], 'alpha': [1e-4, 1e-3]})
    }
    scaled_models = {'LR', 'SVM', 'KNN', 'MLP'}
    res = {}
    best_params_list = []

    for name, (m, p) in models.items():
        xt, xv = (X_train_s, X_test_s) if name in scaled_models else (X_train[features], X_test[features])
        g = GridSearchCV(m, p, cv=3, scoring='roc_auc', n_jobs=-1).fit(xt, y_train)
        best_mod = g.best_estimator_
        res[name] = {
            'mod': best_mod,
            'ptr': best_mod.predict_proba(xt)[:, 1],
            'pte': best_mod.predict_proba(xv)[:, 1],
            'xt': xt,
            'xv': xv
        }
        best_params_list.append({'Model': name, **g.best_params_})

    print("🔬 执行 Nested Cross-Validation (内层调参, 外层评估)...")
    nested_summary = []
    for name, (m, p) in models.items():
        xt = X_train_s if name in scaled_models else X_train[features]
        inner_cv = GridSearchCV(m, p, cv=3, scoring='roc_auc')
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
        nested_scores = cross_val_score(inner_cv, xt, y_train, cv=outer_cv, n_jobs=-1)
        nested_summary.append({
            'Model': name,
            'Nested_AUC_Mean': np.mean(nested_scores),
            'Nested_AUC_Std': np.std(nested_scores)
        })

    plt.figure(figsize=(10, 6))
    n_df = pd.DataFrame(nested_summary)
    sns.barplot(x='Model', y='Nested_AUC_Mean', data=n_df, palette='magma')
    plt.errorbar(x=range(len(n_df)), y=n_df['Nested_AUC_Mean'], yerr=n_df['Nested_AUC_Std'], fmt='none', c='black', capsize=5)
    plt.title('Nested Cross-Validation (Generalization Stability)', fontsize=14, fontweight='bold')
    plt.savefig('Nested_CV_Performance.pdf')

    print("🔬 执行 Repeated Cross-Validation (5-Fold, 10次重复)...")
    best_n = max(res.keys(), key=lambda k: roc_auc_score(y_test, res[k]['pte']))
    best_obj = res[best_n]
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=Config.RANDOM_SEED)
    rep_scores = cross_val_score(best_obj['mod'], best_obj['xt'], y_train, cv=rkf, scoring='roc_auc', n_jobs=-1)

    plt.figure(figsize=(10, 6))
    plt.plot(rep_scores, marker='o', linestyle='-', color='#6A9ACE', alpha=0.7)
    plt.axhline(np.mean(rep_scores), color='red', linestyle='--', label=f'Mean={np.mean(rep_scores):.3f}')
    plt.title(f'Repeated CV Stability ({best_n})', fontsize=14, fontweight='bold')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig('Repeated_CV_Stability.pdf')

    print("🔬 执行 Bootstrap Optimism Correction (AUC 校正)...")
    def calc_optimism(X, y, model, n_boot=100):
        auc_app = roc_auc_score(y, model.predict_proba(X)[:, 1])
        opts = []
        for _ in range(n_boot):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_b, y_b = X.iloc[idx], y.iloc[idx]
            if len(np.unique(y_b)) < 2:
                continue
            model.fit(X_b, y_b)
            auc_boot = roc_auc_score(y_b, model.predict_proba(X_b)[:, 1])
            auc_orig = roc_auc_score(y, model.predict_proba(X)[:, 1])
            opts.append(auc_boot - auc_orig)
        return auc_app, np.mean(opts)

    auc_apparent, optimism = calc_optimism(best_obj['xt'], y_train, best_obj['mod'])
    auc_corrected = auc_apparent - optimism

    plt.figure(figsize=(6, 8))
    plt.plot(['Apparent', 'Corrected'], [auc_apparent, auc_corrected], marker='o', markersize=15, color='darkorange', linewidth=4)
    plt.text(0, auc_apparent + 0.005, f'{auc_apparent:.3f}', ha='center', weight='bold')
    plt.text(1, auc_corrected + 0.005, f'{auc_corrected:.3f}', ha='center', weight='bold')
    plt.title('Bootstrap Optimism Correction', fontsize=14, fontweight='bold')
    plt.ylim(auc_corrected - 0.05, auc_apparent + 0.05)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('Optimism_Correction_Slope.pdf')

    for mode in ['Clean', 'CI']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        for n, r in res.items():
            f1, t1, _ = roc_curve(y_train, r['ptr'])
            x1, y1 = smooth_curve(f1, t1)
            f2, t2, _ = roc_curve(y_test, r['pte'])
            x2, y2 = smooth_curve(f2, t2)

            ax1.plot(x1, y1, lw=2, label=f"{n} (AUC={roc_auc_score(y_train, r['ptr']):.3f})")
            ax2.plot(x2, y2, lw=2, label=f"{n} (AUC={roc_auc_score(y_test, r['pte']):.3f})")

            if mode == 'CI':
                tprs_b = []
                base_f = np.linspace(0, 1, 100)
                for _ in range(200):
                    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
                    if len(np.unique(y_test.values[idx])) < 2:
                        continue
                    fb, tb, _ = roc_curve(y_test.values[idx], r['pte'][idx])
                    tprs_b.append(np.interp(base_f, fb, tb))
                if tprs_b:
                    ax2.fill_between(base_f, np.percentile(tprs_b, 2.5, axis=0), np.percentile(tprs_b, 97.5, axis=0), alpha=0.1)

                tprs_tr = []
                for _ in range(200):
                    idx_tr = np.random.choice(len(y_train), size=len(y_train), replace=True)
                    if len(np.unique(y_train.values[idx_tr])) < 2:
                        continue
                    fb_tr, tb_tr, _ = roc_curve(y_train.values[idx_tr], r['ptr'][idx_tr])
                    tprs_tr.append(np.interp(base_f, fb_tr, tb_tr))
                if tprs_tr:
                    ax1.fill_between(base_f, np.percentile(tprs_tr, 2.5, axis=0), np.percentile(tprs_tr, 97.5, axis=0), alpha=0.1)

        ax1.plot([0,1],[0,1],'k--')
        ax2.plot([0,1],[0,1],'k--')
        ax1.set_title(f'Train ROC ({mode})')
        ax2.set_title(f'Test ROC ({mode})')
        ax1.legend(loc='lower right')
        ax2.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'ROC_Curves_{mode}.pdf')

        fig, (dx1, dx2) = plt.subplots(1, 2, figsize=(16, 8))
        thresh = np.linspace(0.01, 0.99, 100)
        for n, r in res.items():
            dx1.plot(thresh, calc_net_benefit(y_train, r['ptr'], thresh), lw=2, label=n)
            dx2.plot(thresh, calc_net_benefit(y_test, r['pte'], thresh), lw=2, label=n)
            if mode == 'CI':
                nbs_b = []
                for _ in range(50):
                    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
                    nbs_b.append(calc_net_benefit(y_test.values[idx], r['pte'][idx], thresh))
                if nbs_b:
                    dx2.fill_between(thresh, np.percentile(nbs_b, 2.5, axis=0), np.percentile(nbs_b, 97.5, axis=0), alpha=0.1)

        dx1.plot([0,1],[0,0],'k--')
        dx2.plot([0,1],[0,0],'k--')
        dx1.set_title(f'Train DCA ({mode})')
        dx2.set_title(f'Test DCA ({mode})')
        dx1.legend()
        dx2.legend()
        plt.tight_layout()
        plt.savefig(f'DCA_Curves_{mode}.pdf')

        fig, (c1, c2) = plt.subplots(1, 2, figsize=(16, 7))
        for n in res.keys():
            if not hasattr(res[n]['mod'], 'predict_proba'):
                continue
            ytr_p, yte_p = res[n]['ptr'], res[n]['pte']
            pt1, pp1 = calibration_curve(y_train, ytr_p, n_bins=10)
            pt2, pp2 = calibration_curve(y_test, yte_p, n_bins=10)
            b1, l1, u1 = brier_score_confidence_interval(y_train, ytr_p)
            b2, l2, u2 = brier_score_confidence_interval(y_test, yte_p)
            c1.plot(pp1, pt1, marker='o', label=f"{n} ({b1:.3f} CI[{l1:.3f}-{u1:.3f}])")
            c2.plot(pp2, pt2, marker='o', label=f"{n} ({b2:.3f} CI[{l2:.3f}-{u2:.3f}])")

        c1.plot([0,1],[0,1],'k--')
        c2.plot([0,1],[0,1],'k--')
        c1.set_title(f'Calibration Train ({mode})')
        c2.set_title(f'Calibration Test ({mode})')
        c1.legend(fontsize=8)
        c2.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f'Calibration_Curves_{mode}.pdf')

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(14, 6))
    PrecisionRecallDisplay.from_estimator(best_obj['mod'], best_obj['xt'], y_train, plot_chance_level=True, name=best_n, ax=p1)
    PrecisionRecallDisplay.from_estimator(best_obj['mod'], best_obj['xv'], y_test, plot_chance_level=True, name=best_n, ax=p2)
    p1.set_title("PR Curve (Train)")
    p2.set_title("PR Curve (Test)")
    plt.tight_layout()
    plt.savefig('PR_Curves_Best_Model.pdf')

    print("🔬 执行 PSA 灰区分析...")
    gray_mask = (df[Config.PSA_COL] >= 4.0) & (df[Config.PSA_COL] <= 10.0)
    df_gray = df[gray_mask]
    if len(df_gray) > 0:
        xg_raw, yg = df_gray[features], df_gray[Config.TARGET_COL]
        xg_s = pd.DataFrame(scaler.transform(xg_raw), columns=features)
        fig_g, ax_g = plt.subplots(figsize=(8, 8))
        gray_mets = []
        for n, r in res.items():
            xi_g = xg_s if n in scaled_models else xg_raw
            p_g = r['mod'].predict_proba(xi_g)[:, 1]
            auc_g = roc_auc_score(yg, p_g)
            f_g, t_g, _ = roc_curve(yg, p_g)

            # Empirical ROC curves are plotted without smoothing.
            ax_g.plot(f_g, t_g, lw=2.5, label=f"{n} (AUC={auc_g:.3f})")

            # Bootstrap confidence band for ROC in the gray-zone subset
            tprs_g = []
            base_f_g = np.linspace(0, 1, 100)
            for _ in range(200):
                idx = np.random.choice(len(yg), size=len(yg), replace=True)
                if len(np.unique(yg.values[idx])) < 2:
                    continue
                fb, tb, _ = roc_curve(yg.values[idx], p_g[idx])
                tprs_g.append(np.interp(base_f_g, fb, tb))
            if tprs_g:
                ax_g.fill_between(base_f_g, np.percentile(tprs_g, 2.5, axis=0), np.percentile(tprs_g, 97.5, axis=0), alpha=0.1)

            gray_mets.append({
                'Model': n,
                'AUC': get_bootstrap_metrics_ci(yg, p_g, 'AUC'),
                'ACC': get_bootstrap_metrics_ci(yg, p_g, 'ACC'),
                'SENS': get_bootstrap_metrics_ci(yg, p_g, 'SENS'),
                'SPEC': get_bootstrap_metrics_ci(yg, p_g, 'SPEC'),
                'F1': get_bootstrap_metrics_ci(yg, p_g, 'F1')
            })

        ax_g.plot([0,1],[0,1],'k--')
        ax_g.legend(loc='lower right')
        plt.title('PSA Gray Zone ROC (with 95% CI)')
        plt.tight_layout()
        plt.savefig('ROC_PSA_Gray_Zone.pdf')
        pd.DataFrame(gray_mets).to_excel('PSA_Gray_Zone_Metrics.xlsx', index=False)

    explainer = shap.TreeExplainer(best_obj['mod']) if best_n in ['RF', 'XGB'] else shap.KernelExplainer(best_obj['mod'].predict_proba, shap.kmeans(best_obj['xv'], 10))
    sv = explainer.shap_values(best_obj['xv'])
    if isinstance(sv, list):
        sv = sv[1]
    if hasattr(sv, 'shape') and len(sv.shape) == 3:
        sv = sv[:, :, 1]

    plt.figure()
    shap.summary_plot(sv, best_obj['xv'], show=False)
    plt.savefig('SHAP_Summary_Dot.pdf', bbox_inches='tight')

    plt.figure()
    shap.summary_plot(sv, best_obj['xv'], plot_type='bar', show=False)
    plt.savefig('SHAP_Importance_Bar.pdf', bbox_inches='tight')

    shap_df = pd.DataFrame(np.abs(sv).mean(axis=0), index=features, columns=['Mean_SHAP']).reset_index().rename(columns={'index': 'Feature'})

    shap_df_l = pd.DataFrame(sv, columns=features)
    rows_l = int(np.ceil(len(features) / 3))
    fig, axes = plt.subplots(rows_l, 3, figsize=(15, 5 * rows_l))
    for i, f in enumerate(features):
        ax = axes.ravel()[i]
        ax.scatter(best_obj['xv'][f], shap_df_l[f], s=15, alpha=0.5, color="#6A9ACE")
        lw_f = lowess(shap_df_l[f], best_obj['xv'][f], frac=0.3)
        ax.plot(lw_f[:, 0], lw_f[:, 1], color='red', lw=2)
        ax.set_xlabel(f)
    for i in range(len(features), len(axes.ravel())):
        axes.ravel()[i].axis('off')
    plt.tight_layout()
    plt.savefig('SHAP_LOWESS_Best_Model.pdf')

    met_list = []
    for n, r in res.items():
        met_list.append({
            'Model': n,
            'AUC_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'AUC'),
            'AUC_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'AUC'),
            'ACC_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'ACC'),
            'ACC_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'ACC'),
            'SENS_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'SENS'),
            'SENS_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'SENS'),
            'SPEC_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'SPEC'),
            'SPEC_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'SPEC'),
            'F1_Train': get_bootstrap_metrics_ci(y_train, r['ptr'], 'F1'),
            'F1_Test': get_bootstrap_metrics_ci(y_test, r['pte'], 'F1')
        })

    p_comp = []
    names_list = list(res.keys())
    for m1, m2 in itertools.combinations(names_list, 2):
        pval = bootstrap_auc_pvalue(y_test, res[m1]['pte'], res[m2]['pte'], seed=Config.RANDOM_SEED)
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
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

    app_str = f"""import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
st.title("Clinic Predictor")
model = joblib.load('saved_models/{best_n}_best.pkl')
FEATURES = {features!r}
df = pd.read_excel('Final_Cleaned_Data.xlsx')
X_f = df.drop(columns=['{Config.TARGET_COL}', 'ID'], errors='ignore')
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
    explainer = shap.Explainer(model, X_f[FEATURES]); sv_in = explainer(X_in)
    fig = plt.figure(); shap.plots.waterfall(sv_in[0], show=False); st.pyplot(fig)
"""
    with open('APP.py', 'w') as f:
        f.write(app_str)

    os.makedirs('saved_models', exist_ok=True)
    for n, r in res.items():
        joblib.dump(r['mod'], f'saved_models/{n}_best.pkl')

    print("🎉 All analysis tasks completed!")

if __name__ == '__main__':
    main()
