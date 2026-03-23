#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Diagnostic Prediction Model Pipeline
---------------------------------------------
This script performs data preprocessing, feature selection based on normalized 
feature importance, hyperparameter tuning, model evaluation (including nested CV 
and bootstrap optimism correction), and generates final visualizations and tables.
"""

import os
import warnings
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy import stats
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, 
                                     cross_val_score, RepeatedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, recall_score,
                             f1_score, brier_score_loss, PrecisionRecallDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'DejaVu Sans'

class Config:
    DATA_FILE = 'Final_Cleaned_Data.xlsx'
    TARGET_COL = 'status'
    PSA_COL = 'tPSA'
    RANDOM_SEED = 197
    TEST_SIZE = 0.3
    # Update threshold logic: select features with normalized importance > 0.1
    IMPORTANCE_THRESHOLD = 0.1
    OUTPUT_DIR = './outputs'

# Ensure output directories exist
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

def generate_baseline_table(df, target_col, id_col):
    """Generates a baseline characteristics table with stratification by target."""
    stats_list = []
    groups = sorted(df[target_col].dropna().unique())
    df_groups = {g: df[df[target_col] == g] for g in groups}
    
    for col in df.columns:
        if col in [target_col, id_col]: 
            continue
            
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
                c_all = vc[val]
                p_all = c_all / len(df) * 100
                row_cat = {'Feature': f'{col} = {val}', 'Overall': f'{c_all} ({p_all:.1f}%)'}
                for g in groups:
                    c_g = df_groups[g][col].value_counts(dropna=False).get(val, 0)
                    p_g = (c_g / len(df_groups[g]) * 100) if len(df_groups[g]) > 0 else 0
                    row_cat[f'Group {g}'] = f'{c_g} ({p_g:.1f}%)'
                stats_list.append(row_cat)
                
    return pd.DataFrame(stats_list)

def get_bootstrap_metrics_ci(y_true, y_prob, metric_func, n_bootstraps=1000):
    """Calculates evaluation metrics with 95% Confidence Intervals via Bootstrapping."""
    values = []
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: 
            continue
            
        yt, yp = y_true[idx], y_prob[idx]
        ypr = (yp >= 0.5).astype(int)
        
        if metric_func == 'AUC': values.append(roc_auc_score(yt, yp))
        elif metric_func == 'ACC': values.append(accuracy_score(yt, ypr))
        elif metric_func == 'SENS': values.append(recall_score(yt, ypr))
        elif metric_func == 'SPEC': values.append(recall_score(yt, ypr, pos_label=0))
        elif metric_func == 'F1': values.append(f1_score(yt, ypr))
        
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

def bootstrap_auc_pvalue(y_true, prob_a, prob_b, seed=42, n_boot=2000):
    """Compares two models using bootstrap tests for AUC."""
    rng = np.random.RandomState(seed)
    y_true, prob_a, prob_b = np.array(y_true), np.array(prob_a), np.array(prob_b)
    orig_diff = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    diffs = []
    
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2: 
            continue
        diff_boot = roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        diffs.append(diff_boot)
        
    return float(np.mean(np.abs(diffs) >= np.abs(orig_diff)))

def calc_net_benefit(y_true, y_prob, thresholds):
    """Calculates Net Benefit for Decision Curve Analysis (DCA)."""
    nb = []
    for t in thresholds:
        tp = np.sum((y_true == 1) & (y_prob >= t))
        fp = np.sum((y_true == 0) & (y_prob >= t))
        nb.append((tp / len(y_true)) - (fp / len(y_true)) * (t / (1 - t)))
    return np.array(nb)

def smooth_curve(fpr, tpr):
    """Applies PCHIP interpolation to smooth ROC curves."""
    if len(fpr) < 4: 
        return fpr, tpr
    x = np.linspace(0, 1, 300)
    u_fpr, idx = np.unique(fpr, return_index=True)
    return x, np.clip(PchipInterpolator(u_fpr, tpr[idx])(x), 0, 1)

def main():
    print("[INFO] Initializing predictive modeling pipeline...")
    
    # Data Loading
    df = pd.read_excel(Config.DATA_FILE)
    y_all = df[Config.TARGET_COL]
    X_all = df.drop(columns=[Config.TARGET_COL, 'ID'])
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_SEED, stratify=y_all
    )
    
    X_imputed, y = X_train, y_train

    print("[INFO] Executing Feature Selection based on Normalized Importance...")
    
    # 1. Calculate Normalized Feature Importance
    rf_base = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=Config.RANDOM_SEED)
    rf_base.fit(X_imputed, y)
    
    raw_importances = rf_base.feature_importances_
    norm_importances = raw_importances / np.sum(raw_importances)
    
    importance_df = pd.DataFrame({
        'Feature': X_imputed.columns,
        'Norm_Importance': norm_importances
    }).sort_values(by='Norm_Importance', ascending=False).reset_index(drop=True)
    
    # Filter features based on the defined threshold (> 0.1)
    final_selected_features = importance_df[importance_df['Norm_Importance'] > Config.IMPORTANCE_THRESHOLD]['Feature'].tolist()
    
    # 2. Evaluate sequential addition of features (Forward evaluation for plotting)
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
    n_splits = kf.get_n_splits()
    
    selection_results = []
    current_eval_features = []
    
    for i, row in importance_df.iterrows():
        current_feature = row['Feature']
        current_eval_features.append(current_feature)
        fold_roc_scores = []
        
        for train_idx, val_idx in kf.split(X_imputed):
            model = RandomForestClassifier(n_jobs=-1, random_state=Config.RANDOM_SEED)
            model.fit(X_imputed.iloc[train_idx][current_eval_features], y.iloc[train_idx])
            prob = model.predict_proba(X_imputed.iloc[val_idx][current_eval_features])[:, 1]
            fold_roc_scores.append(roc_auc_score(y.iloc[val_idx], prob))
            
        mean_roc = np.mean(fold_roc_scores)
        std_err = stats.sem(fold_roc_scores)
        t_val = stats.t.ppf(0.975, df=n_splits - 1)
        
        selection_results.append({
            'Feature': current_feature,
            'Norm_Importance': row['Norm_Importance'],
            'Mean_ROC': mean_roc,
            'CI_Lower': mean_roc - t_val * std_err,
            'CI_Upper': mean_roc + t_val * std_err
        })
        
    results_df = pd.DataFrame(selection_results)

    # Plotting Feature Importance and Selection Criteria
    fig, ax1 = plt.subplots(figsize=(16, 8))
    norm = plt.Normalize(results_df['Norm_Importance'].min(), results_df['Norm_Importance'].max())
    colors = plt.cm.Blues(norm(results_df['Norm_Importance'].values.astype(float)))
    
    bars = ax1.bar(results_df['Feature'], results_df['Norm_Importance'], color=colors, alpha=0.8)
    
    ax1.set_xticks(range(len(results_df['Feature'])))
    ax1.set_xticklabels(results_df['Feature'], rotation=90, fontsize=9)
    ax1.axhline(y=Config.IMPORTANCE_THRESHOLD, color='r', linestyle='--', linewidth=1.5, label='Importance Threshold (0.1)')
    ax1.set_ylabel('Normalized Feature Importance', fontsize=12)
    
    # Highlight selected features
    for i, label in enumerate(ax1.get_xticklabels()):
        if results_df['Feature'].iloc[i] in final_selected_features: 
            label.set_color('red')
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(1.5)

    ax2 = ax1.twinx()
    ax2.plot(results_df['Feature'], results_df['Mean_ROC'], color="black", marker='o', label="Cumulative Mean ROC")
    
    # Highlight the ROC segment corresponding to selected features
    idx_selected = len(final_selected_features)
    ax2.plot(results_df['Feature'][:idx_selected], results_df['Mean_ROC'][:idx_selected], color="#E53935", marker='o', label="Selected Feature Space")
    ax2.fill_between(results_df['Feature'], results_df['CI_Lower'], results_df['CI_Upper'], color='gray', alpha=0.15)
    
    plt.title(f"Feature Selection: Normalized Importance > {Config.IMPORTANCE_THRESHOLD}", fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'Feature_Selection_Plot.pdf'), dpi=300)

    features = final_selected_features
    print(f"[INFO] Final Selected Features subset: {features}")
    
    # Data scaling for distance-based models
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features)
    X_test_s = pd.DataFrame(scaler.transform(X_test[features]), columns=features)

    # Model definition and Hyperparameter grid
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
    
    # Core Training Pipeline
    print("[INFO] Executing hyperparameter tuning and model training...")
    for name, (m, p) in models.items():
        xt = X_train_s if name in scaled_models else X_train[features]
        xv = X_test_s if name in scaled_models else X_test[features]
        
        grid = GridSearchCV(m, p, cv=3, scoring='roc_auc', n_jobs=-1).fit(xt, y_train)
        best_mod = grid.best_estimator_
        
        res[name] = {
            'mod': best_mod, 
            'ptr': best_mod.predict_proba(xt)[:, 1], 
            'pte': best_mod.predict_proba(xv)[:, 1], 
            'xt': xt, 
            'xv': xv
        }
        best_params_list.append({'Model': name, **grid.best_params_})

    # Validation Strategy 1: Nested Cross-Validation
    print("[INFO] Executing Nested Cross-Validation for generalization evaluation...")
    nested_summary = []
    for name, (m, p) in models.items():
        xt = X_train_s if name in scaled_models else X_train[features]
        inner_cv = GridSearchCV(m, p, cv=3, scoring='roc_auc')
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
        nested_scores = cross_val_score(inner_cv, xt, y_train, cv=outer_cv, n_jobs=-1)
        nested_summary.append({'Model': name, 'Nested_AUC_Mean': np.mean(nested_scores), 'Nested_AUC_Std': np.std(nested_scores)})
    
    plt.figure(figsize=(10, 6))
    n_df = pd.DataFrame(nested_summary)
    sns.barplot(x='Model', y='Nested_AUC_Mean', data=n_df, palette='magma')
    plt.errorbar(x=range(len(n_df)), y=n_df['Nested_AUC_Mean'], yerr=n_df['Nested_AUC_Std'], fmt='none', c='black', capsize=5)
    plt.title('Nested Cross-Validation (Generalization Stability)', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'Nested_CV_Performance.pdf'))

    # Validation Strategy 2: Repeated Cross-Validation
    print("[INFO] Executing Repeated Cross-Validation (5-Fold, 10 Repeats)...")
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
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'Repeated_CV_Stability.pdf'))

    # Validation Strategy 3: Bootstrap Optimism Correction
    print("[INFO] Executing Bootstrap Optimism Correction...")
    def calc_optimism(X_opt, y_opt, model, n_boot=100):
        auc_app = roc_auc_score(y_opt, model.predict_proba(X_opt)[:, 1])
        opts = []
        for _ in range(n_boot):
            idx = np.random.choice(len(X_opt), size=len(X_opt), replace=True)
            X_b, y_b = X_opt.iloc[idx], y_opt.iloc[idx]
            if len(np.unique(y_b)) < 2: 
                continue
            model.fit(X_b, y_b)
            auc_boot = roc_auc_score(y_b, model.predict_proba(X_b)[:, 1])
            auc_orig = roc_auc_score(y_opt, model.predict_proba(X_opt)[:, 1])
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
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'Optimism_Correction_Slope.pdf'))

    # Visualizations: ROC, DCA, Calibration
    print("[INFO] Generating performance evaluation figures...")
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
                tprs_b, base_f = [], np.linspace(0, 1, 100)
                for _ in range(200):
                    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
                    if len(np.unique(y_test.values[idx])) < 2: continue
                    fb, tb, _ = roc_curve(y_test.values[idx], r['pte'][idx])
                    tprs_b.append(np.interp(base_f, fb, tb))
                if tprs_b: 
                    ax2.fill_between(base_f, np.percentile(tprs_b, 2.5, axis=0), np.percentile(tprs_b, 97.5, axis=0), alpha=0.1)
                
                tprs_tr = []
                for _ in range(200):
                    idx_tr = np.random.choice(len(y_train), size=len(y_train), replace=True)
                    if len(np.unique(y_train.values[idx_tr])) < 2: continue
                    fb_tr, tb_tr, _ = roc_curve(y_train.values[idx_tr], r['ptr'][idx_tr])
                    tprs_tr.append(np.interp(base_f, fb_tr, tb_tr))
                if tprs_tr: 
                    ax1.fill_between(base_f, np.percentile(tprs_tr, 2.5, axis=0), np.percentile(tprs_tr, 97.5, axis=0), alpha=0.1)
                    
        ax1.plot([0,1],[0,1],'k--'); ax2.plot([0,1],[0,1],'k--')
        ax1.set_title(f'Train ROC ({mode})'); ax2.set_title(f'Test ROC ({mode})')
        ax1.legend(loc='lower right'); ax2.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'ROC_Curves_{mode}.pdf'))

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
                if nbs_b: dx2.fill_between(thresh, np.percentile(nbs_b, 2.5, axis=0), np.percentile(nbs_b, 97.5, axis=0), alpha=0.1)
        
        dx1.plot([0,1],[0,0],'k--'); dx2.plot([0,1],[0,0],'k--')
        dx1.set_title(f'Train DCA ({mode})'); dx2.set_title(f'Test DCA ({mode})')
        dx1.legend(); dx2.legend(); plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'DCA_Curves_{mode}.pdf'))

        fig, (c1, c2) = plt.subplots(1, 2, figsize=(16, 7))
        for n in res.keys():
            if not hasattr(res[n]['mod'], 'predict_proba'): continue
            ytr_p, yte_p = res[n]['ptr'], res[n]['pte']
            pt1, pp1 = calibration_curve(y_train, ytr_p, n_bins=10)
            pt2, pp2 = calibration_curve(y_test, yte_p, n_bins=10)
            b1, l1, u1 = brier_score_confidence_interval(y_train, ytr_p)
            b2, l2, u2 = brier_score_confidence_interval(y_test, yte_p)
            
            c1.plot(pp1, pt1, marker='o', label=f"{n} ({b1:.3f} 95% CI: [{l1:.3f}-{u1:.3f}])")
            c2.plot(pp2, pt2, marker='o', label=f"{n} ({b2:.3f} 95% CI: [{l2:.3f}-{u2:.3f}])")
            
        c1.plot([0,1],[0,1],'k--'); c2.plot([0,1],[0,1],'k--')
        c1.set_title(f'Calibration Train ({mode})'); c2.set_title(f'Calibration Test ({mode})')
        c1.legend(fontsize=8); c2.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, f'Calibration_Curves_{mode}.pdf'))

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(14, 6))
    PrecisionRecallDisplay.from_estimator(best_obj['mod'], best_obj['xt'], y_train, plot_chance_level=True, name=best_n, ax=p1)
    PrecisionRecallDisplay.from_estimator(best_obj['mod'], best_obj['xv'], y_test, plot_chance_level=True, name=best_n, ax=p2)
    p1.set_title("PR Curve (Train)"); p2.set_title("PR Curve (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'PR_Curves_Best_Model.pdf'))

    # PSA Gray Zone Analysis
    print("[INFO] Performing sub-cohort analysis (PSA Gray Zone)...")
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
            x_sg, y_sg = smooth_curve(f_g, t_g)
            ax_g.plot(x_sg, y_sg, lw=2.5, label=f"{n} (AUC={auc_g:.3f})")
            gray_mets.append({'Model': n, 'AUC': auc_g, 'SENS': recall_score(yg, (p_g>=0.5).astype(int)), 'SPEC': recall_score(yg, (p_g>=0.5).astype(int), pos_label=0)})
            
        ax_g.plot([0,1],[0,1],'k--'); ax_g.legend()
        plt.title('PSA Gray Zone ROC Analysis')
        plt.savefig(os.path.join(Config.OUTPUT_DIR, 'ROC_PSA_Gray_Zone.pdf'))
        pd.DataFrame(gray_mets).to_excel(os.path.join(Config.OUTPUT_DIR, 'PSA_Gray_Zone_Metrics.xlsx'), index=False)

    # Explainable AI: SHAP analysis
    print("[INFO] Generating interpretability analyses (SHAP)...")
    if best_n in ['RF', 'XGB']:
        explainer = shap.TreeExplainer(best_obj['mod'])
    else:
        explainer = shap.KernelExplainer(best_obj['mod'].predict_proba, shap.kmeans(best_obj['xv'], 10))
        
    sv = explainer.shap_values(best_obj['xv'])
    if isinstance(sv, list): sv = sv[1]
    if hasattr(sv, 'shape') and len(sv.shape) == 3: sv = sv[:, :, 1]
    
    plt.figure()
    shap.summary_plot(sv, best_obj['xv'], show=False)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'SHAP_Summary_Dot.pdf'), bbox_inches='tight')
    
    plt.figure()
    shap.summary_plot(sv, best_obj['xv'], plot_type='bar', show=False)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'SHAP_Importance_Bar.pdf'), bbox_inches='tight')
    
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
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'SHAP_LOWESS_Best_Model.pdf'))

    # Compiling Results to Excel
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

    excel_path = os.path.join(Config.OUTPUT_DIR, 'Final_Statistical_Tables.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        generate_baseline_table(df, Config.TARGET_COL, 'ID').to_excel(writer, sheet_name='Baseline_Characteristics', index=False)
        pd.DataFrame(met_list).to_excel(writer, sheet_name='Metrics', index=False)
        pd.DataFrame(p_comp).to_excel(writer, sheet_name='Model_Comparison', index=False)
        pd.DataFrame(best_params_list).to_excel(writer, sheet_name='Hyperparameters', index=False)
        pd.DataFrame([{'Model': n, 'Brier_Test': brier_score_loss(y_test, r['pte'])} for n, r in res.items()]).to_excel(writer, sheet_name='Brier_Score', index=False)
        shap_df.to_excel(writer, sheet_name='SHAP_Importance', index=False)
        pd.DataFrame(nested_summary).to_excel(writer, sheet_name='Nested_CV_Results', index=False)
        pd.DataFrame({'Repetition': range(len(rep_scores)), 'AUC': rep_scores}).to_excel(writer, sheet_name='Repeated_CV_Results', index=False)
        pd.DataFrame([{'Apparent_AUC': auc_apparent, 'Optimism': optimism, 'Corrected_AUC': auc_corrected}]).to_excel(writer, sheet_name='Optimism_Correction', index=False)

    # Streamlit Deployment Code Generation
    app_str = f"""import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title("Clinical Diagnostic Prediction Model")

# Load model artifacts
model = joblib.load('saved_models/{best_n}_best.pkl')
FEATURES = {features!r}

# Reference schema
df = pd.read_excel('Final_Cleaned_Data.xlsx')
X_f = df.drop(columns=['{Config.TARGET_COL}', 'ID'], errors='ignore')

input_vals = []
st.sidebar.header("Patient Characteristics")

for f in FEATURES:
    if pd.api.types.is_numeric_dtype(X_f[f]):
        v = st.sidebar.number_input(f"{{f}}", float(X_f[f].min()), float(X_f[f].max()), float(X_f[f].median()))
    else:
        v = st.sidebar.selectbox(f"{{f}}", X_f[f].unique().tolist())
    input_vals.append(v)

if st.button("Generate Prediction"):
    X_in = pd.DataFrame([input_vals], columns=FEATURES)
    prob = model.predict_proba(X_in)[0][1]
    
    st.markdown(f"### Predicted Probability of Malignancy: **{{prob*100:.2f}}%**")
    
    st.subheader("Model Interpretability (Local SHAP Explanation)")
    explainer = shap.Explainer(model, X_f[FEATURES])
    sv_in = explainer(X_in)
    fig = plt.figure()
    shap.plots.waterfall(sv_in[0], show=False)
    st.pyplot(fig)
"""
    with open('APP.py', 'w', encoding='utf-8') as f: 
        f.write(app_str)
        
    for n, r in res.items(): 
        joblib.dump(r['mod'], f'saved_models/{n}_best.pkl')
        
    print("[INFO] Analytical pipeline executed successfully. Check the 'outputs' directory for results.")

if __name__ == '__main__': 
    main()
