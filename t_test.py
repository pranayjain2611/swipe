import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

def evaluate_t_test(file_name,fdr_corrected = False, save_results = False):    
    df = pd.read_csv(file_name)

    df['sleep_stage'] = df['sleep_stage'].replace({'W': 'W', '1': '1/2', '2': '1/2','3':'1/2'})
    sleep_stages = ['W', '1/2']

    region_columns = df.columns[4:]  
    t_test_results = []

    for region in region_columns:
        # Get the data for wakefulness
        wake_data = df[df['sleep_stage'] == 'W'][region]


        for stage in sleep_stages[1:]:
            stage_data = df[df['sleep_stage'] == stage][region]

            
            t_stat, p_val = stats.ttest_ind(wake_data,stage_data)
            
            # Append results to the list
            t_test_results.append({
                'Region': region,
                'Comparison': ' Wake vs Sleep',
                't-statistic': t_stat,
                'p-value': p_val
            })
    
    
    # Create DataFrame from the list of results
    results_tscore_df = pd.DataFrame(t_test_results)


    # Save results to CSV file if selected
    if(save_results == True):
        directory = os.path.dirname(file_name)
        t_test_path = os.path.join(directory, 't_test.csv')
        results_tscore_df.to_csv(t_test_path, index=False)

    # #Plotting the results
    # fig = plt.figure(figsize=(20, 10))
    # gs = fig.add_gridspec(2)
    # axs = gs.subplots(sharex=True, sharey=True)

    # fig.suptitle('t-testing of Wake vs Sleep')

    # # t-score plot
    # positive_indices = results_tscore_df['t-statistic'] > 0
    # negative_indices = results_tscore_df['t-statistic'] <= 0
    # results_tscore_df.reset_index(drop=True, inplace=True)

    # axs[0].plot(results_tscore_df['Region'], results_tscore_df['t-statistic'], color='blue')
    # axs[0].plot(results_tscore_df[positive_indices]['Region'], results_tscore_df[positive_indices]['t-statistic'],
    #         linestyle='None', marker='o', color='green', label='Positive t-scores')
    # axs[0].plot(results_tscore_df[negative_indices]['Region'], results_tscore_df[negative_indices]['t-statistic'],
    #         linestyle='None', marker='x', color='red', label='Negative t-scores')
    # axs[0].set(ylabel='t-scores')
    # axs[0].legend()
    # axs[0].grid()

    # # p-score plot
    
    # # if fdr_corrected:
    # #     results_tscore_df['p-value'] = stats.false_discovery_control(results_tscore_df['p-value'], axis=0, method='bh')

    # if fdr_corrected:
    #     _, corrected_p_values, _, _ = multipletests(results_tscore_df['p-value'], alpha=0.05, method='fdr_bh')
    #     results_tscore_df['p-value'] = corrected_p_values
    
    # large_p_indices = results_tscore_df['p-value'] > 0.05
    # low_p_indices = results_tscore_df['p-value'] <= 0.05

    # axs[1].plot(results_tscore_df['Region'], results_tscore_df['p-value'], color='blue')
    # axs[1].plot(results_tscore_df[large_p_indices]['Region'], results_tscore_df[large_p_indices]['p-value'],
    #         linestyle='None', marker='x', color='red', label='High p-score')
    # axs[1].plot(results_tscore_df[low_p_indices]['Region'], results_tscore_df[low_p_indices]['p-value'],
    #         linestyle='None', marker='o', color='green', label='Low p-score')
    # axs[1].set(ylabel='t-scores')
    # axs[1].set(ylabel='p-value')
    # axs[1].legend()
    # axs[1].grid()


    # fig.tight_layout()

    # for ax in axs.flat:
    #     ax.set(xlabel='Regions')
    #     ax.tick_params(axis='x', rotation=90)
        
    # for ax in axs:
    #     ax.label_outer()

    # Plot for t-scores
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    fig1.suptitle('t-testing of Wake vs Sleep - t-scores')

    # Separate positive and negative indices for t-scores
    positive_indices = results_tscore_df['t-statistic'] <= 0
    negative_indices = results_tscore_df['t-statistic'] > 0
    results_tscore_df.reset_index(drop=True, inplace=True)

    # Plotting t-scores
    ax1.plot(results_tscore_df['Region'], results_tscore_df['t-statistic'], color='blue')
    ax1.plot(results_tscore_df[positive_indices]['Region'], results_tscore_df[positive_indices]['t-statistic'],
            linestyle='None', marker='o', color='green', label='Suitable t-scores')
    ax1.plot(results_tscore_df[negative_indices]['Region'], results_tscore_df[negative_indices]['t-statistic'],
            linestyle='None', marker='x', color='red', label='Unsuitable t-scores')
    ax1.set_ylabel('t-scores')
    ax1.legend()
    ax1.grid()

    ax1.set_xlabel('Regions')
    ax1.tick_params(axis='x', rotation=90)

    # Plot for p-values
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    fig2.suptitle('t-testing of Wake vs Sleep - p-values')

    # Applying FDR correction if flagged
    if fdr_corrected:
        _, corrected_p_values, _, _ = multipletests(results_tscore_df['p-value'], alpha=0.05, method='fdr_bh')
        results_tscore_df['p-value'] = corrected_p_values

    # Separate indices based on p-value threshold
    large_p_indices = results_tscore_df['p-value'] > 0.05
    low_p_indices = results_tscore_df['p-value'] <= 0.05

    # Plotting p-values
    ax2.plot(results_tscore_df['Region'], results_tscore_df['p-value'], color='blue')
    ax2.plot(results_tscore_df[large_p_indices]['Region'], results_tscore_df[large_p_indices]['p-value'],
            linestyle='None', marker='x', color='red', label='Unsuitable p-score')
    ax2.plot(results_tscore_df[low_p_indices]['Region'], results_tscore_df[low_p_indices]['p-value'],
            linestyle='None', marker='o', color='green', label='Suitable p-score')
    ax2.set_ylabel('p-value')
    ax2.legend()
    ax2.grid()

    ax2.set_xlabel('Regions')
    ax2.tick_params(axis='x', rotation=90)

    # Adjust layout to prevent label overlap
    fig1.tight_layout()
    fig2.tight_layout()

