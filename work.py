"""
This script is designed to manage the workflow of polling data analysis for political candidates. It includes functionality to download and preprocess polling data, calculate daily and state-level weighted averages, simulate electoral vote outcomes, and visualize the results through various plots and maps.

Functions included:
- get_polling_data: Downloads and preprocesses raw polling data.
- create_national_polling_averages: Calculates national daily weighted averages and win probabilities.
- create_state_polling_averages: Computes state-level polling averages and probabilities, adjusting for historical election results.
- simulate_electoral_votes: Simulates electoral vote outcomes using state win probabilities to model possible election results.
- generate_plots: Generates visual representations of polling averages and electoral probabilities.
- generate_map: Creates a choropleth map visualizing state-specific win probabilities.

This script supports extensive data analysis workflows, making it suitable for use in political campaign strategies, academic research, or news analysis. The results provide insights into the current political landscape and potential election outcomes based on polling data.
"""


import io
import os
import time
import warnings  # Import the warnings module
from datetime import timedelta

import csv
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import logging

# Configure logging to capture all output and warnings at the DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('work_log_output.log'),
        logging.StreamHandler()
    ]
)

logging.debug("Script execution initiated.")

# Replace all print statements with logging.info calls throughout the script
# to ensure all output is captured in the log file as well as the console.

# Configure warnings to display all warnings
warnings.simplefilter("default")

# Constants
LAMBDA = 0.0619
STANDARD_DEVIATION = 5.356


def get_polling_data(url, output_file):
    """
    Download the CSV file from the specified URL and save it locally.

    Parameters:
        url (str): The URL of the CSV file to download.
        output_file (str): The path to save the downloaded CSV file.
    """
    print("Entering get_polling_data function.")
    logging.debug("Entering get_polling_data function.")
    # Get the polling data from the URL.
    response = requests.get(url)

    # Decode the content of the response.
    content = response.content.decode('utf-8')

    # Read the polling data into a DataFrame.
    polling_data = pd.read_csv(io.StringIO(content))

    # Identify non-candidate columns.
    non_candidate_columns = [
        col for col in polling_data.columns if col not in ('candidate_name', 'pct')
    ]

    # Drop duplicate rows based on the poll_id column.
    unique_poll_details = polling_data[non_candidate_columns].drop_duplicates('poll_id')

    # Pivot the candidate names and percentages to columns.
    candidate_percentages = polling_data.pivot_table(
        index='poll_id', columns='candidate_name', values='pct', aggfunc='first'
    ).reset_index()

    # Merge the unique poll details with the candidate percentages.
    merged_poll_data = pd.merge(unique_poll_details, candidate_percentages, on='poll_id', how='left')

    # Fill missing values with 0 and infer the most appropriate data types automatically.
    merged_poll_data.fillna(0, downcast='infer', inplace=True)
    logging.info(f"After fillna and downcast='infer', 'end_date' has {merged_poll_data['end_date'].isna().sum()} 'NaT' values.")

    # Rename the columns for consistency.
    merged_poll_data = merged_poll_data[
        ['poll_id', 'display_name', 'state', 'end_date', 'sample_size', 'url'] +
        [col for col in candidate_percentages.columns if col != 'poll_id']
    ]

    # Filter out rows where both candidates have 0 percentage.
    merged_poll_data = merged_poll_data[
        (merged_poll_data['Joe Biden'] != 0) & (merged_poll_data['Donald Trump'] != 0)
    ]

    # Create the output directory if it does not exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the processed polling data to a CSV file.
    merged_poll_data.to_csv(output_file, index=False)
    print("Exiting get_polling_data function.")


def create_national_polling_averages(input_file, output_file):
    print("Entering create_national_polling_averages function.")
    """
    Process polling data from a CSV file to calculate and save daily weighted averages for the candidates along with their win probabilities.

    This function reads polling data, computes daily averages based on weights, and estimates win probabilities for each candidate.

    Parameters:
        input_file (str): Path to the input CSV file containing the polling data.
        output_file (str): Path to save the output CSV file with daily averages.
    """
    # Read polling data from the input file.
    polling_data = pd.read_csv(input_file)

    # Filter to include only national polls.
    polling_data = polling_data[polling_data['state'] == '0']

    # Clean 'end_date' column to ensure consistent date format
    # Add leading zeros to single-digit months/days and convert two-digit years to four-digit years
    # Assuming any year below 30 should be treated as 2000s, otherwise 1900s
    polling_data['end_date'] = polling_data['end_date'].str.replace(r'(?<!\d)(\d{1})/(?=\d{1}/\d{2})', r'0\1/', regex=True)
    polling_data['end_date'] = polling_data['end_date'].str.replace(r'(?<!\d)(\d{1})/(?=\d{2}/\d{2})', r'0\1/', regex=True)
    polling_data['end_date'] = polling_data['end_date'].str.replace(r'(?<=/\d{2}/)(\d{2})(?!\d)', lambda x: '20' + x.group(0) if int(x.group(0)) < 30 else '19' + x.group(0), regex=True)

    # Log the state of the 'end_date' column before conversion
    logging.info(f"'end_date' column before conversion:\n{polling_data['end_date'].head()}")

    # Convert 'end_date' to datetime objects, coercing errors to 'NaT'
    polling_data['end_date'] = pd.to_datetime(polling_data['end_date'], errors='coerce', format='%m/%d/%y')
    logging.info(f"Converted 'end_date' to datetime, resulting in {polling_data['end_date'].isna().sum()} 'NaT' values before removal.")

    # Log details of 'NaT' values for debugging
    nat_values = polling_data[polling_data['end_date'].isna()]
    logging.debug(f"'NaT' values found in the following rows:\n{nat_values}")

    # Remove rows with 'NaT' values in 'end_date'
    polling_data = polling_data.dropna(subset=['end_date'])
    logging.info(f"Removed 'NaT' values, resulting in {polling_data['end_date'].isna().sum()} 'NaT' values after removal.")

    # Ensure there are valid dates for creating the date range
    if polling_data['end_date'].isna().any():
        logging.error("Cannot create date range with 'NaT' values for 'end_date'")
        return "ERROR: Invalid date data in 'end_date' column after removal of 'NaT' values."

    # Ensure that the date range is created from valid dates only
    first_end_date = polling_data['end_date'].min()
    last_end_date = polling_data['end_date'].max()

    # Log the min and max dates to ensure they are not NaT
    logging.info(f"First end date: {first_end_date}, Last end date: {last_end_date}")

    # If either the first or last end date is NaT, log an error and do not attempt to create a date range
    if pd.isna(first_end_date) or pd.isna(last_end_date):
        logging.error("Cannot create date range with NaT values for start or end date.")
        return "Error: Cannot create date range with NaT values for start or end date."

    # Log the rows where 'end_date' is at the min and max to ensure they are valid
    logging.debug(f"Row with first end date:\n{polling_data[polling_data['end_date'] == first_end_date]}")
    logging.debug(f"Row with last end date:\n{polling_data[polling_data['end_date'] == last_end_date]}")

    # Additional logging to check if 'first_end_date' or 'last_end_date' is NaT
    if pd.isna(first_end_date):
        logging.debug(f"First end date is NaT. Unable to create date range.")
    if pd.isna(last_end_date):
        logging.debug(f"Last end date is NaT. Unable to create date range.")

    # Create the date range only if both first and last end dates are valid
    dates = pd.date_range(start=first_end_date, end=last_end_date)
    logging.info(f"Created date range from {first_end_date} to {last_end_date}")

    # Initialize output file and write the header.
    header = "Date,Joe Biden,Donald Trump,Joe Biden Win Probability,Donald Trump Win Probability\n"
    with open(output_file, 'w') as file:
        file.write(header)

    # Calculate daily weighted averages and probabilities.
    for day in dates:
        # Filter polling data based on end date.
        polls = polling_data[polling_data['end_date'] < day].copy()
        polls['days_diff'] = (day - polls['end_date']).dt.days

        # Calculate weights and normalize them.
        polls['weight'] = np.exp(-LAMBDA * polls['days_diff']) * np.sqrt(polls['sample_size'])
        polls['weight'] /= polls['weight'].sum()

        # Calculate weighted averages for each candidate.
        biden_avg = (polls['Joe Biden'] * polls['weight']).sum()
        trump_avg = (polls['Donald Trump'] * polls['weight']).sum()

        # Calculate the z-score and win probabilities.
        margin = biden_avg - trump_avg
        z_score = margin / STANDARD_DEVIATION
        biden_win_prob = norm.cdf(z_score)
        trump_win_prob = 1 - biden_win_prob

        # Append results to the output file.
        results = f"{day.strftime('%Y-%m-%d')},{biden_avg},{trump_avg},{biden_win_prob},{trump_win_prob}\n"
        with open(output_file, 'a') as file:
            file.write(results)

    print("Exiting create_national_polling_averages function.")
    return f"Biden is currently polling at {biden_avg / 100:.2%}, while Trump is at {trump_avg / 100:.2%}."


def create_state_polling_averages():
    """
    Calculate state-level polling averages and win probabilities based on national and state polls.
    This function adjusts shares and boost factors according to past election results and saves the outputs to CSV files.
    """
    logging.debug("Entering create_state_polling_averages function.")
    # Load data from CSV files
    past_results = pd.read_csv('raw_data/raw_past_results.csv')
    national_polling = pd.read_csv('processed_data/president_polls_daily.csv')
    state_polling = pd.read_csv('processed_data/processed_polls.csv')

    # Convert date columns to datetime objects only once, with error handling for parsing
    try:
        national_polling['Date'] = pd.to_datetime(national_polling['Date'], format='%Y-%m-%d', errors='coerce')
    except ValueError as e:
        warnings.warn(f"Date parsing error in national_polling: {e}")

    # Clean 'end_date' column to ensure consistent date format
    # Add leading zeros to single-digit months/days and convert two-digit years to four-digit years
    # Assuming any year below 30 should be treated as 2000s, otherwise 1900s
    state_polling['end_date'] = state_polling['end_date'].str.replace(r'(?<!\d)(\d{1})/(?=\d{1}/\d{2})', r'0\1/', regex=True)
    state_polling['end_date'] = state_polling['end_date'].str.replace(r'(?<!\d)(\d{1})/(?=\d{2}/\d{2})', r'0\1/', regex=True)
    state_polling['end_date'] = state_polling['end_date'].str.replace(r'(?<=/\d{2}/)(\d{2})(?!\d)', lambda x: '20' + x.group(0) if int(x.group(0)) < 30 else '19' + x.group(0), regex=True)

    # Convert 'end_date' to datetime objects, coercing errors to 'NaT'
    state_polling['end_date'] = pd.to_datetime(state_polling['end_date'], errors='coerce', format='%m/%d/%Y')
    # Log the state of the 'end_date' column after conversion
    logging.info(f"'end_date' column after conversion:\n{state_polling['end_date'].head()}")
    logging.info(f"Number of 'NaT' values after conversion: {state_polling['end_date'].isna().sum()}")

    # Remove rows with 'NaT' values in 'end_date'
    state_polling = state_polling.dropna(subset=['end_date'])
    logging.info(f"Removed 'NaT' values, resulting in {state_polling['end_date'].isna().sum()} 'NaT' values after removal.")

    # Define date range for averaging, ensuring no NaT values are used
    if state_polling['end_date'].isna().any():
        logging.error("NaT values present after dropping from 'end_date'. Cannot proceed with date range creation.")
        return "Error: NaT values present after dropping from 'end_date'. Cannot proceed with date range creation."

    # Calculate the start and end dates for the date range
    start_date = national_polling['Date'].min() + timedelta(days=14)
    end_date = state_polling['end_date'].max()

    # Log the min and max dates to ensure they are not NaT
    logging.info(f"Start date for averaging: {start_date}, End date for averaging: {end_date}")

    # If either the start or end date is NaT, log an error and do not attempt to create a date range
    if pd.isna(start_date) or pd.isna(end_date):
        logging.error("Cannot create date range with NaT values for start or end date.")
        return "Error: Cannot create date range with NaT values for start or end date."

    date_range = pd.date_range(start=start_date, end=end_date)
    logging.info(f"Date range created from {start_date} to {end_date}")

    # Extract states excluding national results
    states = past_results.loc[past_results['Location'] != 'National', 'Location'].unique()

    # Pre-calculate national past results for optimization
    national_past_results = past_results.loc[past_results['Location'] == 'National']
    biden_past_national_share = national_past_results['Biden Share'].values[0]
    trump_past_national_share = national_past_results['Trump Share'].values[0]

    # Prepare DataFrames to hold results
    biden_averages = pd.DataFrame(index=date_range, columns=states)
    trump_averages = pd.DataFrame(index=date_range, columns=states)
    biden_win_probabilities = pd.DataFrame(index=date_range, columns=states)

    # Process state data and compute averages and probabilities
    for state in states:
        state_polls = state_polling.loc[state_polling['state'] == state]
        state_past_results = past_results.loc[past_results['Location'] == state]
        biden_past_share = state_past_results['Biden Share'].values[0]
        trump_past_share = state_past_results['Trump Share'].values[0]

        for date in date_range:
            # Calculate boost factors from national polls
            national_polls_to_date = national_polling.loc[national_polling['Date'] <= date]
            current_total = national_polls_to_date.iloc[-1]['Joe Biden'] + national_polls_to_date.iloc[-1]['Donald Trump']
            biden_boost = (national_polls_to_date.iloc[-1]['Joe Biden'] / current_total) / biden_past_national_share
            trump_boost = (national_polls_to_date.iloc[-1]['Donald Trump'] / current_total) / trump_past_national_share

            biden_estimated_share = biden_boost * biden_past_share * current_total
            trump_estimated_share = trump_boost * trump_past_share * current_total

            # Aggregate state-specific polling data up to current date
            state_polls_to_date = state_polls.loc[state_polls['end_date'] <= date]
            national_poll_date = date - timedelta(days=14)

            # Add national polling estimate as an additional "poll"
            national_poll_entry = pd.DataFrame({
                'Joe Biden': [biden_estimated_share],
                'Donald Trump': [trump_estimated_share],
                'end_date': [national_poll_date],
                'sample_size': [1000]
            })
            state_polls_to_date = pd.concat([state_polls_to_date, national_poll_entry], ignore_index=True)

            state_polls_to_date['weight'] = np.exp(-LAMBDA * (date - state_polls_to_date['end_date']).dt.days) * np.sqrt(state_polls_to_date['sample_size'])
            state_polls_to_date['weight'] /= state_polls_to_date['weight'].sum()  # Normalize weights
            biden_state_avg = (state_polls_to_date['Joe Biden'] * state_polls_to_date['weight']).sum()
            trump_state_avg = (state_polls_to_date['Donald Trump'] * state_polls_to_date['weight']).sum()

            # Save daily averages and win probabilities
            biden_averages.loc[date, state] = biden_state_avg
            trump_averages.loc[date, state] = trump_state_avg
            margin = biden_state_avg - trump_state_avg
            z_score = margin / STANDARD_DEVIATION
            biden_win_prob = norm.cdf(z_score)
            biden_win_probabilities.loc[date, state] = biden_win_prob

    # Save results to CSV files
    for df, filename in zip([biden_averages, trump_averages, biden_win_probabilities],
                            ['biden_state_averages.csv', 'trump_state_averages.csv', 'biden_win_probabilities.csv']):
        df.reset_index().rename(columns={'index': 'Date'}).to_csv(f'processed_data/{filename}', index=False)

    # for all the states where biden is between 5% and 95% chance of winning, print them out and each candidates' win probability
    biden_win_probabilities = biden_win_probabilities.iloc[-1]
    trump_win_probabilities = 1 - biden_win_probabilities
    closest_states = biden_win_probabilities[(biden_win_probabilities > 0.05) & (biden_win_probabilities < 0.95)]
    closest_states = closest_states.sort_values()
    # should include the win probability of each candidate
    closest_states_string = ', '.join([f"{state} ({biden_win_probabilities[state]:.2%} Biden, {trump_win_probabilities[state]:.2%} Trump)" for state in closest_states.index])

    return f"The closest states are {closest_states_string}."


def simulate_electoral_votes():
    """
    Simulate the electoral vote outcomes using Biden's state win probabilities, accounting for correlations between states.

    The results are saved to a CSV file.
    """
    print("Entering simulate_electoral_votes function.")
    # Load the electoral votes data and set the index
    electoral_votes = pd.read_csv('raw_data/raw_electoral_votes.csv')
    electoral_votes.set_index('Location', inplace=True)

    # Load Biden's win probabilities
    biden_win_probs = pd.read_csv('processed_data/biden_win_probabilities.csv')

    # Define the correlation matrix for the states
    states = electoral_votes.index.tolist()
    num_states = len(states)
    correlation_matrix = np.full((num_states, num_states), 0.5)
    np.fill_diagonal(correlation_matrix, 1)

    # Set the number of simulations
    num_simulations = 10000

    results = []

    # Simulate electoral vote outcomes for each date
    for date in biden_win_probs['Date'].unique():
        daily_data = biden_win_probs[biden_win_probs['Date'] == date]

        state_indices = [states.index(state) for state in states if state in daily_data.columns]
        win_probs = daily_data[states].iloc[0, state_indices]

        # Generate correlated random outcomes based on win probabilities
        mean = np.arcsin(2 * win_probs - 1)  # Sin transformation for mean
        correlated_normals = np.random.multivariate_normal(mean, correlation_matrix, size=num_simulations)
        correlated_outcomes = (np.sin(correlated_normals) + 1) / 2 > 0.5  # Convert back to determine win/loss

        # Calculate electoral votes for each simulation
        electoral_votes_array = electoral_votes.loc[states, 'Electoral Votes'].values[state_indices]
        simulated_electoral_votes = (correlated_outcomes * electoral_votes_array).sum(axis=1)

        # Calculate probabilities of outcomes
        biden_wins = (simulated_electoral_votes > 269).mean()
        tie = (simulated_electoral_votes == 269).mean()
        trump_wins = 1 - biden_wins - tie

        # Append results for the current date
        results.append({
            'Date': date,
            'Biden Win Probability': biden_wins,
            'Trump Win Probability': trump_wins,
            'Tie Probability': tie
        })

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('processed_data/simulated_national_election_outcomes_correlated.csv', index=False)

    print("Exiting simulate_electoral_votes function.")
    return [f"Biden has an {results[-1]['Biden Win Probability']:.2%} chance of winning the election, Trump has an {results[-1]['Trump Win Probability']:.2%} chance of winning the election, and there is an {results[-1]['Tie Probability']:.2%} chance of a tie.", f"Biden is expected to win {np.median([simulated_electoral_votes]):.0f} electoral votes, while Trump is expected to win {538 - np.median([simulated_electoral_votes]):.0f} electoral votes."]


def generate_plots(polling_data_file, probabilities_file):
    """
    Plot the national polling averages and electoral college probabilities for Biden and Trump.

    Parameters:
        polling_data_file (str): Path to the CSV file containing the polling data.
        probabilities_file (str): Path to the CSV file containing the election probabilities.
    """
    print("Entering generate_plots function.")
    # Read the data from CSV files
    polling_data = pd.read_csv(polling_data_file)
    probabilities = pd.read_csv(probabilities_file)

    # Convert 'Date' columns to datetime format
    polling_data['Date'] = pd.to_datetime(polling_data['Date'])
    probabilities['Date'] = pd.to_datetime(probabilities['Date'])

    # Filter data for events starting from 2023
    polling_data = polling_data[polling_data['Date'] >= pd.Timestamp('2023-01-01')]
    probabilities = probabilities[probabilities['Date'] >= pd.Timestamp('2023-01-01')]

    # Plot national polling averages
    plt.figure(figsize=(12, 8))
    plt.plot(polling_data['Date'], polling_data['Joe Biden'], label='Joe Biden', color='blue')
    plt.plot(polling_data['Date'], polling_data['Donald Trump'], label='Donald Trump', color='red')
    plt.title('National Polling Averages')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim([min(polling_data['Joe Biden'].min(), polling_data['Donald Trump'].min()) - 5,
              max(polling_data['Joe Biden'].max(), polling_data['Donald Trump'].max()) + 5])
    plt.savefig('static/plots/national_polling_averages.png')

    # Plot electoral college probabilities
    plt.figure(figsize=(12, 8))
    plt.plot(probabilities['Date'], probabilities['Biden Win Probability'] * 100, label='Joe Biden', color='blue')
    plt.plot(probabilities['Date'], probabilities['Trump Win Probability'] * 100, label='Donald Trump', color='red')
    plt.plot(probabilities['Date'], probabilities['Tie Probability'] * 100, label='Tie', color='yellow')
    plt.title('Electoral College Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Probability (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig('static/plots/election_win_probabilities.png')
    print("Exiting generate_plots function.")


def generate_map(state_probabilities, states_shapefile):
    """
    Generate a choropleth map visualizing the probabilities of a specified outcome by state.

    Parameters:
        state_probabilities (str): Path to the CSV file containing state probabilities.
        states_shapefile (str): Path to the shapefile of US states.
    """
    # Load probability data and the US states shapefile
    data = pd.read_csv(state_probabilities)
    states = gpd.read_file(states_shapefile)

    # Prepare the data
    last_row = data.iloc[-1].rename('Probability').to_frame().reset_index().rename(columns={'index': 'State'})
    states = states.rename(columns={'NAME': 'State'})
    states = states[~states['State'].isin([
        'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'Guam', 'United States Virgin Islands',
        'American Samoa', 'Alaska', 'Hawaii'])]

    # Merge and prepare states data
    states_data = states.merge(last_row, on='State', how='left')
    states_data['Probability'] = states_data['Probability'].fillna(0, downcast='infer')

    # Define colors
    white = '#FFFFFF'

    # Create a custom colormap based on Biden's win probability
    def custom_color(prob):
        if prob == 0.5:
            return white
        elif prob > 0.5:
            white_share = 1 - (prob - 0.5) * 2
            return plt.cm.colors.to_hex([white_share, white_share, 1])
        else:
            white_share = 1 - (0.5 - prob) * 2
            return plt.cm.colors.to_hex([1, white_share, white_share])

    states_data['Color'] = states_data['Probability'].apply(custom_color)

    # Create and configure the plot
    _, ax = plt.subplots(1, figsize=(15, 8))
    states_data.plot(color=states_data['Color'], linewidth=0.2, ax=ax, edgecolor='black')
    ax.axis('off')
    plt.tight_layout()

    # Save the map to a file
    plt.savefig('static/plots/win_probability_map.png', dpi=1000, bbox_inches='tight')
    print("Exiting generate_map function.")

def do_work():
    print("Entering do_work function.")
    try:
        logging.debug("do_work function execution started.")
        logging.info("Starting do_work function.")
        url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
        processed_file = 'processed_data/processed_polls.csv'
        output_file = 'processed_data/president_polls_daily.csv'

        storage = []

        get_polling_data(url, processed_file)
        logging.info("Polling data downloaded and processed.")

        polling_averages_string = create_national_polling_averages(processed_file, output_file)
        if "ERROR" in polling_averages_string:
            logging.error(polling_averages_string)
            return
        storage.append(polling_averages_string)
        logging.info("National polling averages calculated.")

        close_states_string = create_state_polling_averages()
        if "ERROR" in close_states_string:
            logging.error(close_states_string)
            return
        storage.append(close_states_string)
        logging.info("State polling averages calculated.")

        electoral_college_votes_list = simulate_electoral_votes()
        storage.extend(electoral_college_votes_list)
        logging.info("Electoral votes simulated.")

        generate_plots('processed_data/president_polls_daily.csv', 'processed_data/simulated_national_election_outcomes_correlated.csv')
        logging.info("Polling data plots generated.")

        generate_map('processed_data/biden_win_probabilities.csv', 'raw_data/cb_2023_us_state_500k.shp')
        logging.info("Map generated.")

        # create the file if it doesn't exist

        # save storage to a csv file
        with open('static/work_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in storage:
                writer.writerow([item])  # Write each item as its own row
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
