'''
Exploratory Data Anlaysis Module

    Univariate  Analysis :  Categorical -> barplots  ,  Numerical -> histogram + Kde
    Bivariate Analysis :  categorical -> cross tab barplot  , numerical -> boxplot + violin plot
    Multivariate Analysis :  correlation matrix + pairplot

'''


import os 
import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis


#_________________________________________________________

# Univariate Analysis 
# ________________________________________________________________ 

# Categorical univariate

def categorical_univariate(df: pd.DataFrame, outputdir: str = None) -> None:

    non_numeric_col = [col for col in df.columns if df[col].dtype == 'object']

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    for ax, col in zip(axes.flatten(), non_numeric_col):
        vc = df[col].value_counts()
        pct = vc / len(df) * 100
        bars = ax.bar(vc.index, vc.values, color='steelblue', alpha=0.8)
        for bar, (cnt, p) in zip(bars, zip(vc.values, pct.values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f'{cnt:,}\n({p:.1f}%)', ha='center', fontsize=8)
        ax.set_title(col, fontweight='bold')

    plt.tight_layout()

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'categorical_univariate.jpg'))


# Numerical Univariate  

def numerical_univariate(df: pd.DataFrame, numeric_columns: list[str], outputdir: str = None) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numeric_columns):
        df[col].hist(ax=ax, bins=20, density=True,
                     color='steelblue', alpha=0.7, edgecolor='black')

        df[col].plot.kde(ax=ax, color='black', lw=2)
        # mean v line
        ax.axvline(df[col].mean(), color='red', ls='--', lw=1.5,
                   label=f'Mean: {df[col].mean():.1f}')
        # median v line
        ax.axvline(df[col].median(), color='green', ls=':', lw=1.5,
                   label=f'Median: {df[col].median():.1f}')
        sk = skew(df[col])
        ax.set_title(f'{col} | Skew: {sk:+.2f}')

    ax.legend(fontsize=8)
    plt.tight_layout()

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'numerical_univariate.png'))




#_________________________________________________________________

# Bivariate Analysis 
#_________________________________________________________________

# CroosTab barplot

def categorical_bivariate(df: pd.DataFrame, non_numeric_col: list[str], outputdir: str = None) -> None:

    fig, axes = plt.subplots(8, 2, figsize=(18, 40))
    axes = axes.flatten()

    for ax, feat in zip(axes, non_numeric_col):
        ct = pd.crosstab(df[feat], df['Churn'], normalize='index') * 100

        # Sort by churn for
        ct = ct.sort_values(by='Yes')

        # Plot
        ct.plot(kind='bar',
                stacked=True,
                ax=ax,
                color=['#2ECC71', '#E74C3C'],
                edgecolor='none',
                width=0.7)

        # Add churn % labels
        for i, (idx, row) in enumerate(ct.iterrows()):
            churn = row['Yes']
            retained = row['No']

            if churn > 5:
                ax.text(i,
                        retained + churn / 2,
                        f"{churn:.1f}%",
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='white',
                        fontweight='bold')

        ax.set_title(feat, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_ylim(0, 100)

        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Remove top right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ['Retained', 'Churned'],
               loc='upper center',
               ncol=2,
               fontsize=11,
               frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'categorical_bivariate.jpg'))
    plt.show()



def box_plot_bivariate(df: pd.DataFrame, numeric_columns: list[str], outputdir: str = None) -> None:

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    for ax, col in zip(axs.flatten(), numeric_columns):

        groups = [df[df['Churn'] == 'No'][col],
                  df[df['Churn'] == 'Yes'][col]]

        bp = ax.boxplot(groups,
                        patch_artist=True,
                        tick_labels=['Retained', 'Churned'],
                        medianprops=dict(color='white', lw=2))

        colors = ['#2ecc71', '#e67e22']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for i, group in enumerate(groups, 1):
            ax.plot(i, group.mean(), marker='D', color='white',
                    markersize=6, zorder=5, markeredgecolor='black')

        ax.set_xlabel('Churn Status')
        ax.set_ylabel(col)
        ax.set_title(f'{col} by Churn Status')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Numeric Features vs. Churn Status',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'numeric_feature_churn_status.png'))
    plt.show()



def violin_plot_bivariate(df: pd.DataFrame, outputdir: str = None) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    contracts = df['Contract'].unique()
    for ax, contract in zip(axes, contracts):
        sub = df[df['Contract'] == contract]
        sns.violinplot(data=sub, x='Churn', y='MonthlyCharges', hue='Churn',
                       palette={'No': '#1E8449', 'Yes': '#E67E22'},
                       inner='box', ax=ax, legend=False)
        ax.set_title(contract, fontweight='bold')
        ax.set_xticklabels(['Retained', 'Churned'])

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'violinplot.png'))



#_____________________________________________________________________________

# Multivariate Analysis 
#_____________________________________________________________________________


# Corelation Matrix 

def correlation_matrix(df: pd.DataFrame, outputdir: str = None) -> None:

    corr_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    temp_df_for_corr = df[corr_cols].copy()
    temp_df_for_corr['SeniorCitizen'] = temp_df_for_corr['SeniorCitizen'].map({'yes': 1, 'no': 0})

    correlation = temp_df_for_corr.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'correlationmatrix.jpg'))


def pairplot(df: pd.DataFrame, outputdir: str = None) -> None:

    # Pair plot

    pair_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].copy()
    pair_df['Churn'] = pair_df['Churn'].map({'No': 'Retained', 'Yes': 'Churned'})
    g = sns.pairplot(pair_df, hue='Churn',
                     palette={'Retained': 'skyblue', 'Churned': 'orange'},
                     diag_kind='kde',
                     plot_kws={'alpha': 0.25, 's': 10},
                     diag_kws={'alpha': 0.6})
    g.fig.suptitle('Scatter Matrix — Numeric Features × Churn',
                   y=1.02, fontsize=12, fontweight='bold')

    if outputdir:
        os.makedirs(outputdir, exist_ok=True)
        plt.savefig(os.path.join(outputdir, 'Pairplot.jpg'))
