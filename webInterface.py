import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load and prepare data
def load_data(start_date, end_date):
    data = yf.download("AMAN.JK", start=start_date, end=end_date)
    data['return'] = data['Adj Close'].pct_change().dropna()
    return data

# Set page title
st.title('Stock Price Prediction and Simulation')

# Informasi nama emiten dan nama anggota kelompok
st.markdown("""
* *Emiten:* AMAN.JK
* *Kelompok:*  
  - Alya Nabila Muliani (1301213343)  
  - Alicia Kristina Parinussa (1301213507)  
  - Devi Zahra Aulia (1301213155)  
  - Sintia Dwi Cahya (1301213440)  
**Yang Dapat Diinputkan !!!
* *Rentang Waktu Data AMAN.JK: 13 March 2020 sampai 13 March 2024
""")

# Input tanggal dari pengguna
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

# Input field for number of simulations
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1, max_value=100, value=10)

# Button to load data
if st.button('Load Data'):
    # Load data based on user input
    data = load_data(start_date, end_date)

    # Placeholder for simulated paths and steps
    steps = int(len(data) * 0.5)  # Number of steps from earlier simulation
    simulated_paths = np.random.normal(loc=data['Adj Close'].mean(), scale=10, size=(len(data) - steps + 1))  # Mock data for paths

    # Actual stock prices from the data
    harga_act = data['Adj Close'][steps-1:].values

    # Show data in a table
    st.subheader('Stock Price Data')
    st.write(data)

    # Sidebar options
    st.sidebar.header('Simulation Parameters')
    # Add a slider for number of steps
    steps_slider = st.sidebar.slider('Number of Simulation Steps', min_value=10, max_value=len(data) // 2, value=len(data) // 2)

    # Add input fields for LSTM training parameters
    st.sidebar.subheader('LSTM Model Training Parameters')
    epochs = st.sidebar.slider('Number of Epochs', min_value=10, max_value=100, value=20)
    batch_size = st.sidebar.slider('Batch Size', min_value=1, max_value=32, value=16)

    # Plotting simulated vs actual stock prices
    st.subheader('Simulated vs Actual Stock Price Paths')
    fig, ax = plt.subplots(figsize=(10, 6))
    index = data.index[steps-1:]  # Corresponding dates
    ax.plot(index, simulated_paths, label='Predicted', marker='.', linestyle='-')
    ax.plot(index, harga_act, label='Actual', marker='.', linestyle='-')
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_title("Simulated vs Actual Stock Price Paths")
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Error analysis plots
    st.subheader('Error Analysis')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Absolute error plot
    abs_errors = [abs(i - j) for (i, j) in zip(simulated_paths, harga_act)]
    ax[0].plot(index, abs_errors, '.-', color='red')
    ax[0].set_title('Absolute Error of Predicted Price')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Price Difference")

    # Relative absolute error plot
    rel_errors = [abs(i - j) / j * 100 for (i, j) in zip(simulated_paths, harga_act)]
    ax[1].plot(index, rel_errors, '.-', color='blue')
    ax[1].set_title('Relative Absolute Error of Predicted Price (in %)')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Percentage")

    # Improving the x-axis labels readability
    _ = [ax[i].tick_params(axis='x', labelrotation=40) for i in [0, 1]]
    plt.tight_layout()

    # Show the error plots in Streamlit
    st.pyplot(fig)

    # Placeholder for additional statistics
    st.subheader('Additional Statistics')
    percentiles = np.percentile(simulated_paths, [10, 50, 90])
    st.write(f"10th Percentile: {percentiles[0]:.2f}")
    st.write(f"50th Percentile (Median): {percentiles[1]:.2f}")
    st.write(f"90th Percentile: {percentiles[2]:.2f}")

    # DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Date': index,
        'Predicted Price': simulated_paths,
        'Actual Price': harga_act,
        'Absolute Error': abs_errors,
        'Relative Error (%)': rel_errors
    })

    # Function to convert dataframe to CSV
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    # Function to convert dataframe to Excel
    def convert_df_to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    # Download buttons
    csv_data = convert_df_to_csv(predictions_df)
    st.download_button(label="Download Predictions as CSV", data=csv_data, file_name='predictions.csv', mime='text/csv')
