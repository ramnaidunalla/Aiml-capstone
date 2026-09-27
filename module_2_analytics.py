from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / 'charts'
CHARTS.mkdir(exist_ok=True)


def load_once():
    """Load raw Titanic data exactly once through seaborn; fall back to committed CSV offline."""
    try:
        df = sns.load_dataset('titanic')
    except Exception as exc:
        fallback = ROOT / 'titanic.csv'
        if not fallback.exists():
            raise RuntimeError('Seaborn load failed and titanic.csv fallback is missing.') from exc
        df = pd.read_csv(fallback)
    # Required immediate offline fallback save.
    df.to_csv(ROOT / 'titanic.csv', index=False)
    return df


def profile(df):
    print('=== INFO ===')
    df.info()
    print('\n=== DESCRIBE ===')
    print(df.describe(include='all').transpose().to_string())
    print('\n=== SHAPE ===')
    print(df.shape)
    missing = (df.isna().mean() * 100).loc[lambda s: s.gt(0)]
    print('\n=== MISSING % ===')
    print(missing.to_string())
    return missing


def clean(df):
    out = df.copy()
    missing = out.isna().mean() * 100
    decisions = []
    for col, pct in missing[missing > 0].items():
        if pct < 5:
            out = out.dropna(subset=[col])
            decisions.append((col, pct, 'drop rows', 'Below 5% threshold.'))
        elif pct <= 30:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].fillna(out[col].median())
                decisions.append((col, pct, 'median impute', 'Between 5% and 30%; numeric median is robust to skew.'))
            else:
                out[col] = out[col].fillna(out[col].mode(dropna=True).iloc[0])
                decisions.append((col, pct, 'mode impute', 'Between 5% and 30%; categorical mode preserves an observed category.'))
        else:
            # High-missing deck is retained as an explicit Missing category.
            out[col] = out[col].fillna('Missing')
            decisions.append((col, pct, 'encode Missing category', 'Above 30%; imputation would be unreliable, so missingness is preserved explicitly.'))
    return out, decisions


def savefig(name):
    plt.tight_layout()
    plt.savefig(CHARTS / name, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    df = load_once()
    missing = profile(df)
    clean_df, decisions = clean(df)
    print('\n=== CLEANING DECISIONS ===')
    for row in decisions:
        print(f'{row[0]}: {row[1]:.3f}% -> {row[2]} | {row[3]}')
    clean_df.to_csv(ROOT / 'cleaned_titanic.csv', index=False)

    # Univariate: age/fare distributions and IQR outliers.
    for col in ['age', 'fare']:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(clean_df[col].dropna(), bins=30)
        axes[0].set_title(f'{col.title()} distribution')
        axes[0].set_xlabel(col)
        axes[1].boxplot(clean_df[col].dropna(), vert=False)
        axes[1].set_title(f'{col.title()} box plot')
        axes[1].set_xlabel(col)
        plt.tight_layout()
        fig.savefig(CHARTS / f'univariate_{col}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

        q1, q3 = clean_df[col].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((clean_df[col] < lo) | (clean_df[col] > hi)).sum())
        print(f'{col} IQR fences: [{lo:.4f}, {hi:.4f}], outliers={count}')

    fare_mean = clean_df['fare'].mean()
    fare_median = clean_df['fare'].median()
    fare_mode = clean_df['fare'].mode().iloc[0]
    print(f'fare mean={fare_mean:.4f}, median={fare_median:.4f}, mode={fare_mode:.4f}')

    # Bivariate survival rates, using boolean masks for sex/pclass combinations.
    sex_rates = clean_df.groupby('sex', observed=True)['survived'].mean()
    class_rates = clean_df.groupby('pclass')['survived'].mean()
    combo_rates = clean_df.groupby(['sex', 'pclass'], observed=True)['survived'].mean()
    print('\n=== SURVIVAL RATES BY SEX ===\n', sex_rates)
    print('\n=== SURVIVAL RATES BY PCLASS ===\n', class_rates)
    print('\n=== SURVIVAL RATES BY SEX + PCLASS ===\n', combo_rates)
    # Explicit boolean masking example required by the brief.
    female_first = clean_df[(clean_df['sex'] == 'female') & (clean_df['pclass'] == 1)]['survived'].mean()
    male_third = clean_df[(clean_df['sex'] == 'male') & (clean_df['pclass'] == 3)]['survived'].mean()
    print(f'Boolean-mask checks: female & first={female_first:.4f}; male & third={male_third:.4f}')

    corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr = clean_df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0, square=True)
    plt.title('Titanic numeric correlation matrix')
    savefig('correlation_heatmap.png')
    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i+1:]:
            pairs.append((abs(corr.loc[a, b]), corr.loc[a, b], a, b))
    top2 = sorted(pairs, reverse=True)[:2]
    print('\n=== TOP 2 ABSOLUTE OFF-DIAGONAL CORRELATIONS ===')
    for _, value, a, b in top2:
        print(f'{a} vs {b}: r={value:.4f}')

    # Four+ multivariate charts.
    plt.figure(figsize=(8, 5))
    rates = clean_df.groupby(['pclass','sex'], observed=True)['survived'].mean().unstack()
    rates.plot(kind='bar', ax=plt.gca())
    plt.ylabel('Survival rate')
    plt.title('Survival rate by class and sex')
    savefig('story_1_survival_class_sex.png')

    plt.figure(figsize=(8, 5))
    groups = [clean_df.loc[(clean_df.pclass == c) & (clean_df.survived == s), 'fare'] for c in [1,2,3] for s in [0,1]]
    plt.boxplot(groups, labels=['1/0','1/1','2/0','2/1','3/0','3/1'])
    plt.xlabel('pclass / survived')
    plt.ylabel('fare')
    plt.title('Fare distributions by class and survival')
    savefig('story_2_fare_class_survival.png')

    plt.figure(figsize=(8, 5))
    for sex, marker in [('female','o'), ('male','^')]:
        for surv, linestyle in [(0,'--'), (1,'-')]:
            sub = clean_df[(clean_df.sex == sex) & (clean_df.survived == surv)]
            plt.scatter(sub.age, sub.fare, marker=marker, alpha=.55, label=f'{sex}, survived={surv}')
    plt.xlabel('age'); plt.ylabel('fare'); plt.legend(fontsize=7); plt.title('Age vs fare by survival and sex')
    savefig('story_3_age_fare_survival.png')

    plt.figure(figsize=(9, 5))
    rates2 = clean_df.groupby(['pclass','embarked'], observed=True)['survived'].mean().unstack()
    rates2.plot(kind='bar', ax=plt.gca())
    plt.ylabel('Survival rate')
    plt.title('Survival rate by class and embarkation port')
    savefig('story_4_class_embarked_survival.png')

    plt.figure(figsize=(9, 6))
    groups2 = [clean_df.loc[(clean_df.sex == sex) & (clean_df.survived == surv), 'age'] for sex in ['female','male'] for surv in [0,1]]
    plt.boxplot(groups2, labels=['female/0','female/1','male/0','male/1'])
    plt.ylabel('age'); plt.title('Age distribution by sex and survival')
    savefig('story_5_age_sex_survival.png')

    # Exploratory z-score check on full cleaned data only.
    standardized = clean_df[['age', 'fare']].copy()
    before = standardized.agg(['mean', 'std']).transpose()
    for col in standardized:
        standardized[col] = (standardized[col] - standardized[col].mean()) / standardized[col].std(ddof=1)
    after = standardized.agg(['mean', 'std']).transpose()
    print('\n=== STANDARDIZATION BEFORE ===\n', before)
    print('\n=== STANDARDIZATION AFTER ===\n', after)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col, ax in zip(['age', 'fare'], axes):
        ax.hist(clean_df[col], bins=30, density=True, alpha=.45, label='before')
        ax.hist(standardized[col], bins=30, density=True, alpha=.45, label='after z-score')
        ax.set_title(f'{col}: before vs after standardization')
        ax.legend()
    plt.tight_layout()
    fig.savefig(CHARTS / 'standardization_before_after.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    print(f'Cleaned shape: {clean_df.shape}')

if __name__ == '__main__':
    main()
