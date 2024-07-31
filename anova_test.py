import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

def evaluate_anova_test(file_name,fdr_corrected = False, save_results = False): 
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Define the sleep stages you're interested in
    sleep_stages = ['W', '1', '2']

    region_columns = df.columns[4:]  

    # Store the results
    anova_results = []

    for region in region_columns:
        # Get the data for each sleep stage
        sleep_stage_data = []
        for stage in sleep_stages:
            sleep_stage_data.append(df[df['sleep_stage'] == stage][region])
            
        # Perform one-way ANOVA
        f_stat, p_val = stats.f_oneway(*sleep_stage_data)
        
        # Append results to the list
        anova_results.append({
            'Region': region,
            'F-statistic': f_stat,
            'p-value': p_val
        })

    # Create DataFrame from the list of results
    results_anova_df = pd.DataFrame(anova_results)

    # Save results to CSV file
    if(save_results == True):
        directory = os.path.dirname(file_name)
        t_test_path = os.path.join(directory, 'anova_test.csv')
        results_anova_df.to_csv(t_test_path, index=False)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2)
    axs = gs.subplots(sharex=True, sharey=True)

    fig.suptitle('ANOVA of Wake vs N1 vs N2')

    # t-score plot
    positive_indices = results_anova_df['F-statistic'] > 0
    negative_indices = results_anova_df['F-statistic'] <= 0
    results_anova_df.reset_index(drop=True, inplace=True)

    axs[0].plot(results_anova_df['Region'], results_anova_df['F-statistic'], color='blue')
    axs[0].plot(results_anova_df[positive_indices]['Region'], results_anova_df[positive_indices]['F-statistic'],
            linestyle='None', marker='o', color='green', label='Positive t-scores')
    axs[0].plot(results_anova_df[negative_indices]['Region'], results_anova_df[negative_indices]['F-statistic'],
            linestyle='None', marker='x', color='red', label='Negative t-scores')
    axs[0].set(ylabel='F-statistic')
    axs[0].legend()
    axs[0].grid()

    # p-score plot
    if(fdr_corrected):
        results_anova_df['p-value'] = stats.false_discovery_control(results_anova_df['p-value'], axis=0, method='bh')

    large_p_indices = results_anova_df['p-value'] > 0.1
    low_p_indices = results_anova_df['p-value'] <= 0.1

    axs[1].plot(results_anova_df['Region'], results_anova_df['p-value'], color='blue')
    axs[1].plot(results_anova_df[large_p_indices]['Region'], results_anova_df[large_p_indices]['p-value'],
            linestyle='None', marker='x', color='red', label='High p-score')
    axs[1].plot(results_anova_df[low_p_indices]['Region'], results_anova_df[low_p_indices]['p-value'],
            linestyle='None', marker='o', color='green', label='Low p-score')
    axs[1].set(ylabel='t-scores')
    axs[1].set(ylabel='p-value')
    axs[1].legend()
    axs[1].grid()


    fig.tight_layout()

    for ax in axs.flat:
        ax.set(xlabel='Regions')
        ax.tick_params(axis='x', rotation=90)
        
    for ax in axs:
        ax.label_outer()

