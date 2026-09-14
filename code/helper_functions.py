### multivariable Cox regression
def cox_regression(X, time_col='Time', event_col='Event'):
    import pandas as pd
    import numpy as np
    from lifelines import CoxPHFitter

    X_cox = X.copy()

    X_cox = X_cox.dropna()

    cph = CoxPHFitter()
    cph.fit(X_cox, duration_col=time_col, event_col=event_col, formula = ' + '.join(X_cox.columns.difference([time_col, event_col])))

    summary = cph.summary
    df = pd.DataFrame({
        "hazard_ratio": summary["exp(coef)"],
        "ci_lower_95": summary["exp(coef) lower 95%"],
        "ci_upper_95": summary["exp(coef) upper 95%"],
        "p_value": summary["p"],
        "log2_p_value": np.log2(summary["p"])
    })
    #round everything exept p_value to 4 decimals
    df[["hazard_ratio", "ci_lower_95", "ci_upper_95", "log2_p_value"]] = df[["hazard_ratio", "ci_lower_95", "ci_upper_95", "log2_p_value"]].round(4)

    return df.sort_values(by='hazard_ratio', ascending=False)

### Univariable Cox regression
def univariable_cox(df, time_col='Time', event_col='Event', reference = None):
    """
    Perform univariable Cox regression for each feature column (one at a time).
    Parameters:
    - df: DataFrame including time, event, and covariates
    - time_col: column name for survival time
    - event_col: column name for event flag (1 = event occurred, 0 = censored)
    
    Returns:
    - DataFrame with one row per covariate: coef, HR, p-value, CI
    """
    from lifelines import CoxPHFitter
    import pandas as pd
    from tqdm import tqdm
    import numpy as np

    results = []

    covariates = [col for col in df.columns if col not in [time_col, event_col]]

    for cov in covariates:
        data = df[[time_col, event_col, cov]].copy()

        if data[cov].dtype == 'object' or str(data[cov].dtype).startswith('category'):
            if reference is not None and reference in data[cov].unique():
                # Set reference first
                new_categories = [reference] + [cat for cat in data[cov].unique() if cat != reference]
                data[cov] = pd.Categorical(data[cov], categories=new_categories, ordered=True)
            # Encode category
            data[cov] = data[cov].astype('category').cat.codes

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col=time_col, event_col=event_col)
            summary = cph.summary.loc[cov]
            results.append({
                'covariate': cov,
                'coef': round(summary['coef'], 4),
                'HR': round(summary['exp(coef)'], 4),
                'p': summary['p'],
                'CI_lower': round(summary['exp(coef) lower 95%'], 4),
                'CI_upper': round(summary['exp(coef) upper 95%'], 4)
            })
        except Exception as e:
            results.append({
                'covariate': cov,
                'coef': None,
                'HR': None,
                'p': None,
                'CI_lower': None,
                'CI_upper': None,
                'error': str(e)
            })

    return pd.DataFrame(results).set_index('covariate').sort_values('p')

import numpy as np

def plot_time_dependent_auc_from_scores(risk_score_list, model_names, y_train, y_test, times=np.arange(12, 121, 12), figsize=(7, 5), ylim = None, legend_out = False, grid = True, title = None, save = None, palette = None, fontsize_small = False):
    """
    This function uses as input only the risk scores, generated with model.predict(X_test)
    Plot time-dependent AUCs for multiple models using precomputed risk scores.

    Parameters:
    - risk_score_list: list of risk score arrays (each shape: [n_samples] or [n_samples, len(times)])
    - model_names: list of names for each model (same length as risk_score_list)
    - y_train: structured training labels
    - y_test: structured test labels
    - times: array of time points (e.g., months)
    - figsize: size of the matplotlib figure
    - ylim: tuple specifying y-axis limits (optional)
    - legend_out: boolean to place legend outside the plot (optional)
    - grid: boolean to show grid (optional)
    - save: file path to save the plot (optional)
    - palette = dictionary of colors for each model (optional)
    - fontsize_small: boolean to use small font size for labels and legend (optional)
    """
    from sksurv.metrics import cumulative_dynamic_auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    
    for scores, name in zip(risk_score_list, model_names):
        aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, scores, times)
        if palette:
            plt.plot(times, aucs, marker="o", label=f"{name} (mean AUC: {mean_auc:.2f})", color=palette.get(name))
        else:
            plt.plot(times, aucs, marker="o", label=f"{name} (mean AUC: {mean_auc:.2f})")

    if fontsize_small:
        plt.xlabel("Time (months)", fontsize='small')
        plt.ylabel("Area Under the Curve (AUC)", fontsize='small')
    else:
        plt.xlabel("Time (months)")
        plt.ylabel("Area Under the Curve (AUC)")
    plt.xticks(times)
    if grid:
        plt.grid(True)
    else: 
        plt.grid(False)
    if legend_out:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize='small' if fontsize_small else None)
    else:
        plt.legend(frameon=False, fontsize='small' if fontsize_small else None)
    if fontsize_small:
        plt.tick_params(labelsize='small')
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title, fontweight='bold')
    plt.tight_layout()
    if save:
        import matplotlib as mpl
        mpl.rcParams['pdf.fonttype'] = 42
        plt.savefig(save, bbox_inches='tight', dpi=300)
    plt.show()



def y_forSurv(data):
    '''
    Transform the y variable into the format required by sksurv
    data ---> DataFrame containing 'Event' and 'Time' columns and IDs as index
    return ---> y variable in the format required by sksurv
    '''
    from sksurv.util import Surv
    return Surv.from_dataframe("Event", "Time", data.reset_index(drop=True))



def plot_kaplan_meier_risktable(X, category=None, upper_limit = 60, step = 12, figsize=(6, 4), ylabel = '', title = '', save=None, colors=None, only_risk = False, conf_intervals = False, time = 'Time', event = 'Event', legend_out = None, fontsize_small = False):
    '''
    Plot Kaplan Meier curves for each risk group with at-risk table
    X --> pd.DataFrame with 'Time' (duration), 'Event' (1/0), and a categorical column for stratification
    category --> the column to use for stratification, do a dictionary of colors and groups, e.g., colors={'Low':'blue', 'High':'red'}
    upper_limit --> the upper limit for the x-axis
    step --> the step for the x-axis
    figsize --> the size of the figure
    ylabel --> the label for the y-axis
    title --> the title of the plot
    save --> the path to save the plot, default None
    colors --> dictionary with colors for each group, default None
    only_risk --> if True, only show the at-risk table without the censored marks, default False
    conf_intervals --> if True, show confidence intervals, default False
    time --> the column name for the time variable, default 'Time'
    event --> the column name for the event variable, default 'Event'
    fontsize_small --> if True, use smaller font size for the plot, default False
    '''
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    import seaborn as sns

    kmfs = []
    fig, ax = plt.subplots(figsize=figsize)

    for group in X[category].unique():
        kmf = KaplanMeierFitter()
        mask = X[category] == group
        kmf.fit(X.loc[mask, time], X.loc[mask, event], label=str(group))
        if colors is not None and group in colors:
            if fontsize_small:
                #reduce width of line 
                kmf.plot(ax=ax, ci_show=conf_intervals, show_censors=True, color=colors[group], censor_styles={'ms': 5}, linewidth=0.8)
            else:
                kmf.plot(ax=ax, ci_show=conf_intervals, show_censors=True, color=colors[group], censor_styles={'ms': 8})
        else:
            if fontsize_small:
                kmf.plot(ax=ax, ci_show=conf_intervals, show_censors=True, censor_styles={'ms': 5}, linewidth=0.8)
            else:
                kmf.plot(ax=ax, ci_show=conf_intervals, show_censors=True, censor_styles={'ms': 8})
        kmfs.append(kmf)

    ax.set_xlim(0, upper_limit)
    ax.set_xticks(range(0, upper_limit + 1, step))
    ax.set_ylim(0, 1)
    if fontsize_small:
        ax.set_xlabel('Time (months)', fontsize='small')
    else:
        ax.set_xlabel('Time (months)')
    if fontsize_small:
        ax.set_title(title, fontweight='bold', fontsize='small')
    else:
        ax.set_title(title, fontweight='bold')
    if fontsize_small:
        ax.set_ylabel(ylabel, fontsize='small')
    else:
        ax.set_ylabel(ylabel)
    if fontsize_small:
        ax.legend(frameon = False, fontsize = 'small')
    else:
        ax.legend(frameon = False)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(bottom = True, left = True)
    if fontsize_small:
        ax.tick_params(labelsize = 'small')
    sns.despine()

    if legend_out:
        ax.legend(bbox_to_anchor=(1.20, 1), loc=legend_out, borderaxespad=0., frameon=False)

    # Add at-risk table
    if only_risk:
        add_at_risk_counts(*kmfs, ax=ax, rows_to_show = ['At risk'], fontsize='small' if fontsize_small else None)
    else:     
        add_at_risk_counts(*kmfs, ax=ax, fontsize='small' if fontsize_small else None)
    if save:
        import matplotlib as mpl
        mpl.rcParams['pdf.fonttype'] = 42
        fig.savefig(save, bbox_inches='tight')
    plt.show()
