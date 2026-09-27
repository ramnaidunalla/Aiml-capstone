from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / 'charts'
RANDOM_STATE = 42

CLASS_NUMERIC = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CLASS_CATEGORICAL = ['sex', 'embarked']
CLASS_FEATURES = CLASS_NUMERIC + CLASS_CATEGORICAL


def classification_preprocessor():
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric, CLASS_NUMERIC),
        ('cat', categorical, CLASS_CATEGORICAL),
    ])


def model_pipelines():
    return {
        'Logistic Regression': Pipeline([
            ('preprocess', classification_preprocessor()),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        'Decision Tree': Pipeline([
            ('preprocess', classification_preprocessor()),
            ('model', DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ]),
        'Random Forest': Pipeline([
            ('preprocess', classification_preprocessor()),
            ('model', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'AUC': roc_auc_score(y_test, proba),
        'Confusion Matrix': cm.tolist(),
        'fpr': roc_curve(y_test, proba)[0],
        'tpr': roc_curve(y_test, proba)[1],
    }


def main():
    # Continues from the one committed fallback produced by 01_eda.py.
    raw = pd.read_csv(ROOT / 'titanic.csv')
    # Reproduce the single cleaning pass from 01_eda so this notebook/script can be run independently after that artifact exists.
    df = raw.copy()
    df = df.dropna(subset=['embarked', 'embark_town'])
    df['age'] = df['age'].fillna(df['age'].median())
    df['deck'] = df['deck'].fillna('Missing')

    X = df[CLASS_FEATURES]
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print('Class balance full:', y.value_counts(normalize=True).sort_index().to_dict())
    print('Class balance train:', y_train.value_counts(normalize=True).sort_index().to_dict())
    print('Class balance test:', y_test.value_counts(normalize=True).sort_index().to_dict())

    models = model_pipelines()
    results = []
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, X_test, y_test))

    # Decision-tree visualization with labeled transformed feature names/classes.
    tree = fitted['Decision Tree']
    names = tree.named_steps['preprocess'].get_feature_names_out()
    plt.figure(figsize=(20, 10))
    plot_tree(tree.named_steps['model'], feature_names=names, class_names=['not survived', 'survived'], filled=True, max_depth=4, fontsize=7)
    plt.title('Decision Tree (displayed to depth 4)')
    plt.savefig(CHARTS / 'decision_tree.png', dpi=160, bbox_inches='tight')
    plt.close()

    # ROC curves.
    plt.figure(figsize=(8, 6))
    for r in results:
        plt.plot(r['fpr'], r['tpr'], label=f"{r['Model']} (AUC={r['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curves')
    plt.legend(); plt.tight_layout(); plt.savefig(CHARTS / 'roc_curves.png', dpi=160, bbox_inches='tight'); plt.close()

    # Single comparison table for the three classifiers.
    comparison = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Confusion Matrix', 'fpr', 'tpr']} for r in results])
    comparison.to_csv(ROOT / 'classification_metrics.csv', index=False)
    print('\n=== CLASSIFIER METRICS ===')
    print(comparison.to_string(index=False))
    for r in results:
        print(f"{r['Model']} confusion matrix:\n{np.array(r['Confusion Matrix'])}")

    # Imbalance handling: same Logistic Regression model, train-only transformations.
    imbalance = []
    baseline = Pipeline([
        ('preprocess', classification_preprocessor()),
        ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    balanced = Pipeline([
        ('preprocess', classification_preprocessor()),
        ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)),
    ])
    smote_pipe = ImbPipeline([
        ('preprocess', classification_preprocessor()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    for label, pipe in [('baseline/no handling', baseline), ('class_weight=balanced', balanced), ('SMOTE train-only', smote_pipe)]:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        imbalance.append({
            'Variant': label,
            'Precision': precision_score(y_test, pred, zero_division=0),
            'Recall': recall_score(y_test, pred, zero_division=0),
            'F1': f1_score(y_test, pred, zero_division=0),
        })
    imbalance_df = pd.DataFrame(imbalance)
    imbalance_df.to_csv(ROOT / 'imbalance_comparison.csv', index=False)
    print('\n=== IMBALANCE COMPARISON ===\n', imbalance_df.to_string(index=False))

    # Random Forest grid search; OOB score is explicitly enabled.
    rf_grid = Pipeline([
        ('preprocess', classification_preprocessor()),
        ('model', RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    grid = GridSearchCV(
        rf_grid,
        {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 5, 10],
            'model__max_features': ['sqrt', 'log2'],
        }, cv=5, scoring='roc_auc', n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    oob = best_rf.named_steps['model'].oob_score_
    print('\n=== RF GRID SEARCH ===')
    print('Best params:', grid.best_params_)
    print('Best CV AUC:', grid.best_score_)
    print('OOB score:', oob)
    with open(ROOT / 'grid_search_results.json', 'w') as f:
        json.dump({'best_params': grid.best_params_, 'best_cv_auc': grid.best_score_, 'oob_score': oob}, f, indent=2)

    # Regression side-task. Predict fare from all other available, non-identifier fields; keep survived as a legitimate predictor.
    reg_df = df.drop(columns=['fare', 'alive'])
    Xr = reg_df.drop(columns=[])
    yr = df['fare']
    xr_train, xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.20, random_state=RANDOM_STATE)
    num_cols = Xr.select_dtypes(include=np.number).columns.tolist()
    cat_cols = Xr.select_dtypes(exclude=np.number).columns.tolist()
    reg_pre = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
    ])
    reg_pipe = Pipeline([('preprocess', reg_pre), ('model', LinearRegression())])
    reg_pipe.fit(xr_train, yr_train)
    pred_fare = reg_pipe.predict(xr_test)
    mae = mean_absolute_error(yr_test, pred_fare)
    rmse = (mean_squared_error(yr_test, pred_fare) ** 0.5)
    r2 = r2_score(yr_test, pred_fare)
    p = len(reg_pipe.named_steps['preprocess'].get_feature_names_out())
    n = len(yr_test)
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1) if n > p + 1 else np.nan
    residuals = yr_test - pred_fare
    plt.figure(figsize=(8, 5))
    plt.scatter(pred_fare, residuals, alpha=.65)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted fare'); plt.ylabel('Residual (actual - predicted)')
    plt.title('Fare regression residual plot')
    plt.tight_layout(); plt.savefig(CHARTS / 'fare_residuals.png', dpi=160, bbox_inches='tight'); plt.close()
    regression_metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Adjusted_R2': adj_r2, 'n_test': n, 'p_features_after_encoding': p}
    with open(ROOT / 'regression_metrics.json', 'w') as f: json.dump(regression_metrics, f, indent=2)
    print('\n=== REGRESSION ===', regression_metrics)

    # Final deployable model: choose highest test F1 among the three classifiers, then fit its complete pipeline on all cleaned data.
    best_result = max(results, key=lambda r: r['F1'])
    best_name = best_result['Model']
    full_pipeline = models[best_name]
    full_pipeline.fit(X, y)
    artifact = ROOT / 'titanic_classifier_pipeline.joblib'
    joblib.dump(full_pipeline, artifact)
    reloaded = joblib.load(artifact)
    raw_sample_pred = reloaded.predict(raw[CLASS_FEATURES].head(5))
    print(f'Best classifier by test F1: {best_name}; reload predictions on raw, unpreprocessed CSV rows: {raw_sample_pred.tolist()}')

    # Final combined table with distinct metric groups; regression numbers are not ranked against classification metrics.
    class_table = comparison.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
    final_table = class_table.copy()
    final_table['Regression_MAE'] = mae
    final_table['Regression_RMSE'] = rmse
    final_table['Regression_R2'] = r2
    final_table['Regression_Adjusted_R2'] = adj_r2
    final_table.to_csv(ROOT / 'model_comparison.csv')
    print('\n=== FINAL MODEL COMPARISON ===\n', final_table.to_string())

    print('\nFINAL RECOMMENDATION DATA POINTS:')
    print(best_result)

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()
