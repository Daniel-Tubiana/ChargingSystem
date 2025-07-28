import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io


# Set the main directory path
#main_directory = 'C:/Users/dtubiana/PycharmProjects/chargeSystem/eval_model_V6/run_results'
#output_path = 'C:/Users/dtubiana/PycharmProjects/chargeSystem/eval_model_V6/run results pictures'

main_directory = 'G:/My Drive/Systems engineering Studies/Final project/files submissions/Final Report/Reuslts/last run/run_results'
output_path = 'G:/My Drive/Systems engineering Studies/Final project/files submissions/Final Report/Reuslts/last run/run_results_pictures'
# TODO: count how much evs were not charged at all
# Loop through each subdirectory in the main directory
for subdir, _, _ in os.walk(main_directory):
    os.chdir(subdir)
    files_in_subdir = os.listdir(subdir)

    # Filter for Excel files (both .xls and .xlsx formats)
    excel_files = [file for file in files_in_subdir if file.endswith(('.xls', '.xlsx'))]

    # If an Excel file is found in the subdirectory
    if excel_files:
        xls_file = pd.ExcelFile(excel_files[0])

        # Get the names of all the sheets in the file
        sheet_names = xls_file.sheet_names

        sheets_data = {}
        for sheet in sheet_names:
            sheets_data[sheet] = pd.read_excel(xls_file, sheet_name=sheet)

        run_name = (files_in_subdir[0].split('.'))[0]
        charge_pwr_df = sheets_data['charge_pwr']
        charge_pwr_df = charge_pwr_df.iloc[:-1, :]
        # Calculating the mean and std for each row excluding the 'surplusPWR' column
        surplus_pwr = charge_pwr_df['surplusPWR']
        charge_pwr_df['mean'] = charge_pwr_df.drop(columns='surplusPWR').mean(axis=1)
        charge_pwr_df['std'] = charge_pwr_df.drop(columns='surplusPWR').std(axis=1)
        # Creating a new dataframe with the specified columns
        charge_pwr_df = charge_pwr_df[['mean', 'std', 'surplusPWR']]
        # Load the 'SOC_mat' sheet into a dataframe
        soc_mat_df = pd.read_excel(xls_file, sheet_name='SOC_mat')

        # Calculate the mean and std for each row
        soc_mat_df['mean'] = soc_mat_df.mean(axis=1)
        soc_mat_df['std'] = soc_mat_df.std(axis=1)

        # Create a new dataframe with the specified columns
        soc_summary_df = soc_mat_df[['mean', 'std']]
        # Load the 'df_evs_out' sheet into a dataframe
        df_evs_out = pd.read_excel(xls_file, sheet_name='df_evs_out')

        # Concatenate the soc_summary_df to df_evs_out
        df_ev_summery = pd.concat([df_evs_out, soc_summary_df], axis=1)
        # Drop specified columns
        df_ev_summery.drop(columns=['Arrival_time[h]', 'TuD (int)'], inplace=True)
        # Calculate 'SOC target' column
        min_soc = 0.2
        df_ev_summery['SOC target'] = min_soc + df_ev_summery['ENonD'] / df_ev_summery['Battery capacity [KWh]']
        total_reward_df = pd.read_excel(xls_file, sheet_name='total_reward')
        mean_power_delta = (surplus_pwr - charge_pwr_df['mean']).mean()
        mean_total_reward = total_reward_df.mean()
        std_total_reward = total_reward_df.std()
        charge_pwr_variance = charge_pwr_df['mean'].var()
        n_evs_deviations = ((df_ev_summery['SOC'] - df_ev_summery['SOC target']) < 0).sum()
        mean_soc_deviation = ((df_ev_summery['SOC'] - df_ev_summery['SOC target']) < 0).mean()



        SOC_smaller_than_ENonD = ((df_ev_summery['SOC'] - (df_ev_summery['ENonD'] / df_ev_summery['Battery capacity [KWh]'])) < 0).sum()
        #median_soc_deviation = ((df_ev_summery['SOC'] - df_ev_summery['SOC target']) < 0).median()
        min_soc_val = df_ev_summery['SOC'].min()
        from PIL import ImageFont



        # Efont = ImageFont.truetype("C:/Users/dtubiana/PycharmProjects/chargeSystem/font/Roboto-Italic.ttf", size= 20)

        # Helper function to convert plot to image
        def plot_to_img():
            time_h = list(range(23))
            plt.figure(figsize=(12, 7))

            # Step plot for mean power with the std
            plt.step(time_h, charge_pwr_df['mean'], label='Mean Power', where='mid', color='blue')
            plt.fill_between(time_h, charge_pwr_df['mean'] - charge_pwr_df['std'],
                             charge_pwr_df['mean'] + charge_pwr_df['std'], color='blue', alpha=0.2,
                             label='Std Power')

            # Step plot for surplusPWR
            plt.step(time_h, charge_pwr_df['surplusPWR'], label='Surplus Power', where='mid', color='red')

            # Setting labels, title and legend
            plt.xlabel('Time [h]')
            plt.ylabel('Power [Kw]')
            plt.title('Charge Power vs Time of day')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            plt.tight_layout()

            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            return img


        def format_value(value):
            """Format the value to a string without showing its type."""
            # Check for pandas Series or DataFrame
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value = value.squeeze()  # Convert to scalar if single value
                if isinstance(value, (float, int)):
                    return "{:.2f}".format(value)
                return str(value)
            elif isinstance(value, float):
                return "{:.2f}".format(value)
            elif isinstance(value, int):
                return str(value)
            else:
                return str(value)  # For other types, just convert to string


        def combine_outputs(data_dict, title=None, plot_offset=0,
                            font_path="C:/Users/dtubiana/PycharmProjects/chargeSystem/font/Roboto-Italic.ttf",
                            text_size=20):
            # Convert the plot to an image
            plot_img = plot_to_img()

            # Load the custom font
            font = ImageFont.truetype(font_path, text_size)

            # Determine image width and height
            width = max(plot_img.width, 800)  # Minimum width of 800 or plot width
            height = plot_img.height + 200 + (
                50 if title else 0) + plot_offset  # 200 for strings, 50 for title, plot_offset for moving plot down

            # Create a white background image
            img = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(img)

            # Draw title if provided
            y_offset = 10
            if title:
                title_width, title_height = draw.textsize(title, font=font)
                draw.text(((width - title_width) // 2, y_offset), title, font=font, fill="black")
                y_offset += title_height + 10

            # Draw data from the dictionary
            # Draw data from the dictionary
            for key, value in data_dict.items():
                text = f"{key}:  {format_value(value)}"
                draw.text((10, y_offset), text, font=font, fill="black")
                y_offset += text_size + 5  # Adjusting vertical spacing based on text size

            # Adjust for plot offset
            y_offset += plot_offset

            # Paste the plot
            img.paste(plot_img, ((width - plot_img.width) // 2, y_offset))

            return img


        # Sample dictionary with string names and values for demonstration
        data_dict = {
            "Mean total reward": mean_total_reward,
            "STD total reward": std_total_reward,
            "Mean Power Delta": mean_power_delta,
            "Charge power variance": charge_pwr_variance,
            "Number cases 'not enough energy'": n_evs_deviations,
            "Number cases SOC smaller than ENonD": SOC_smaller_than_ENonD,
            "Mean soc deviation": mean_soc_deviation

        }


        def plot_SOC_delta_histogram(data, run_name):
            """
            Plot a histogram from a dataframe with the title based on the run_name.

            Args:
            - data (pd.DataFrame): DataFrame containing the data.
            - run_name (str): Name of the run to be used as title.

            Returns:
            - None (But saves the plot as an image.)
            """
            plt.figure(figsize=(10, 6))

            # Plotting the histogram
            plt.hist(data, bins=30, edgecolor='k', alpha=0.7)

            # Setting the titles and labels
            fig_name = run_name + ' SOC delta in %'
            plt.title(fig_name)
            plt.xlabel('Delta SOC % from target')
            plt.ylabel('Count of EVs')

            # Save the figure
            filename = run_name.replace(" ", "_") + "_histogram.png"
            plt.savefig(filename)
            plt.close()

            return filename  # Return the saved filename for reference

            # Sample data for testing


        SOC_delta = (df_ev_summery['SOC'] - df_ev_summery['SOC target'])

        os.chdir(output_path)
        filename = plot_SOC_delta_histogram(SOC_delta, run_name)
        output_img = combine_outputs(data_dict, title=f"Power Analysis: {run_name}", plot_offset=0, text_size=20)
        output_img.save(run_name+".png") # save power fig and stats




