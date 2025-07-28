import os
import pandas as pd
import numpy as np

# Set the main directory path
main_directory = 'C:/Users/dtubiana/PycharmProjects/chargeSystem/eval_model_V6/run_results'
output_path = 'C:/Users/dtubiana/PycharmProjects/chargeSystem/eval_model_V6/run results data'


main_directory = 'G:/My Drive/Systems engineering Studies/Final project/files submissions/Final Report/Reuslts/last run/run_results'
output_path = 'G:/My Drive/Systems engineering Studies/Final project/files submissions/Final Report/Reuslts/last run/run_results_summery'


# Create an empty dataframe to store results
columns = [
    "Run Name", "Mean total reward", "STD total reward", "Mean Power Delta",
    "Charge power variance", "Number cases 'not enough energy'",
    "Number cases SOC smaller than ENonD", "Mean soc deviation", 'min_soc_val'
]
results_df = pd.DataFrame(columns=columns)

# Loop through each subdirectory in the main directory
for subdir, _, _ in os.walk(main_directory):
    os.chdir(subdir)
    files_in_subdir = os.listdir(subdir)

    # Filter for Excel files (both .xls and .xlsx formats)
    excel_files = [file for file in files_in_subdir if file.endswith(('.xls', '.xlsx'))]

    # Continue if no Excel files found
    if not excel_files:
        continue

    xls_file = pd.ExcelFile(excel_files[0])

    sheets_data = {}
    for sheet in xls_file.sheet_names:
        sheets_data[sheet] = pd.read_excel(xls_file, sheet_name=sheet)

    run_name = (files_in_subdir[0].split('.'))[0]
    charge_pwr_df = sheets_data['charge_pwr']
    charge_pwr_df = charge_pwr_df.iloc[:-1, :]
    surplus_pwr = charge_pwr_df['surplusPWR']
    charge_pwr_df['mean'] = charge_pwr_df.drop(columns='surplusPWR').mean(axis=1)
    charge_pwr_df['std'] = charge_pwr_df.drop(columns='surplusPWR').std(axis=1)

    soc_mat_df = sheets_data['SOC_mat']
    soc_mat_df['mean'] = soc_mat_df.mean(axis=1)
    soc_mat_df['std'] = soc_mat_df.std(axis=1)
    soc_summary_df = soc_mat_df[['mean', 'std']]
    df_evs_out = sheets_data['df_evs_out']
    df_ev_summery = pd.concat([df_evs_out, soc_summary_df], axis=1)
    df_ev_summery.drop(columns=['Arrival_time[h]', 'TuD (int)'], inplace=True)
    min_soc = 0.2
    df_ev_summery['SOC target'] = min_soc + df_ev_summery['ENonD'] / df_ev_summery['Battery capacity [KWh]']
    total_reward_df = sheets_data['total_reward']
    mean_power_delta = (surplus_pwr - charge_pwr_df['mean']).mean()
    mean_total_reward = total_reward_df.mean().values[0]  # Assuming it's a single column dataframe
    std_total_reward = total_reward_df.std().values[0]
    charge_pwr_variance = charge_pwr_df['mean'].var()
    n_evs_deviations = ((df_ev_summery['SOC'] - df_ev_summery['SOC target']) < 0).sum()
    mean_soc_deviation = ((df_ev_summery['SOC'] - df_ev_summery['SOC target']) < 0).mean()
    SOC_smaller_than_ENonD = (
                (df_ev_summery['SOC'] - (df_ev_summery['ENonD'] / df_ev_summery['Battery capacity [KWh]'])) < 0).sum()
    min_soc_val = df_ev_summery['SOC'].min()

    # Create a dictionary with the results
    data_dict = {
        "Run Name": run_name,
        "Mean total reward": mean_total_reward,
        "STD total reward": std_total_reward,
        "Mean Power Delta": mean_power_delta,
        "Charge power variance": charge_pwr_variance,
        "Number cases 'not enough energy'": n_evs_deviations,
        "Number cases SOC smaller than ENonD": SOC_smaller_than_ENonD,
        "Mean soc deviation": mean_soc_deviation
    }

    # Append the results to the dataframe
    results_df = pd.concat([results_df, pd.DataFrame([data_dict])], ignore_index=True)


# At the end of the loop, you'll have a dataframe with all the results
#print(results_df)

os.chdir(output_path)
# Optionally, save the dataframe to a CSV file
results_df.to_csv('summary_results.csv', index=False)
