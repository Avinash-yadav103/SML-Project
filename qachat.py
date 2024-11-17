from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables

import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt # type: ignore


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def get_energy_management_advice(electricity_usage, solar_generation, other_energy_sources):
    question = f"How can I manage my energy consumption with the following data? Electricity Usage: {electricity_usage} kWh, Solar Generation: {solar_generation} kWh, Other Energy Sources: {other_energy_sources} kWh"
    response = get_gemini_response(question)
    return response

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")

st.header("Home Energy Manager")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and input:
    response = get_gemini_response(input)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", input))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")

# Section for managing electricity consumption
st.subheader("Manage Your Home Energy Consumption")

# Input fields for different energy sources
electricity_usage = st.number_input("Electricity Usage (kWh):", min_value=0.0, step=0.1)
solar_generation = st.number_input("Solar Generation (kWh):", min_value=0.0, step=0.1)
other_energy_sources = st.number_input("Other Energy Sources (kWh):", min_value=0.0, step=0.1)

# Button to submit energy data
submit_energy_data = st.button("Submit Energy Data")

if submit_energy_data:
    st.write("Electricity Usage:", electricity_usage, "kWh")
    st.write("Solar Generation:", solar_generation, "kWh")
    st.write("Other Energy Sources:", other_energy_sources, "kWh")
    st.success("Energy data submitted successfully!")
    
    # Get energy management advice from Gemini
    advice_response = get_energy_management_advice(electricity_usage, solar_generation, other_energy_sources)
    st.subheader("Energy Management Advice")
    for chunk in advice_response:
        st.write(chunk.text)

# Section for AI/ML data study
st.subheader("AI/ML Data Study")

# Placeholder for AI/ML dataset and analysis
st.write("This section will include the AI/ML data study using the dataset which will be added later.")

# Load the dataset
data_path = './Synthetic_Household_Electricity_Consumption_Dataset.csv'
if os.path.exists(data_path):
    data = pd.read_csv(data_path)

    # Display the dataset
    st.write("Household Electricity Consumption Data")
    st.dataframe(data.head())

    # Plotting the data
    st.write("Electricity Usage Over Time")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Power Consumption (W)'], label='Electricity Usage')
    ax.set_xlabel('Date')
    ax.set_ylabel('Electricity Usage (kWh)')
    ax.legend()
    st.pyplot(fig)

    # st.write("Solar Generation Over Time")
    # fig, ax = plt.subplots()
    # ax.plot(data.index, data['Solar_Generation'], label='Solar Generation', color='orange')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Solar Generation (kWh)')
    # ax.legend()
    # st.pyplot(fig)

    # st.write("Other Energy Sources Over Time")
    # fig, ax = plt.subplots()
    # ax.plot(data.index, data['Other_Energy_Sources'], label='Other Energy Sources', color='green')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Other Energy Sources (kWh)')
    # ax.legend()
    # st.pyplot(fig)

#     # Correlation heatmap
#     st.write("Correlation Heatmap")
#     corr = data.corr()
#     fig, ax = plt.subplots()
#     cax = ax.matshow(corr, cmap='coolwarm')
#     fig.colorbar(cax)
#     ax.set_xticks(range(len(corr.columns)))
#     ax.set_yticks(range(len(corr.columns)))
#     ax.set_xticklabels(corr.columns, rotation=90)
#     ax.set_yticklabels(corr.columns)
#     st.pyplot(fig)
# else:
#     st.error(f"Dataset not found at path: {data_path}")


import nbformat # type: ignore

# Title for the Streamlit app
st.title("Household Electricity Consumption Analysis")

# Load the preprocessed model or data from the .ipynb conversion
notebook_file = 'Household_Electricity_Consumption_Preprocessing.ipynb'  # Ensure this file exists
try:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
except FileNotFoundError:
    st.error(f"Notebook file '{notebook_file}' not found. Please make sure it is in the same directory.")
    st.stop()

# Execute the notebook and extract the variables

# ep = ExecutePreprocessor(timeout=600, kernel_name='python3')  # Ensure 'python3' is a valid kernel name
# ep.preprocess(notebook, {'metadata': {'path': './'}})

# ep = ExecutePreprocessor(timeout=600, kernel_name='python3')  
data_path = './Synthetic_Household_Electricity_Consumption_Dataset.csv'  # Adjust to the CSV path if needed
try:
    data = pd.read_csv(data_path)
    st.header("Dataset Overview")
    st.write(data.head())

    # Plotting some basic visualizations
    st.subheader("Electricity Usage Visualization")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Power Consumption (W)'], label='Electricity Usage')
    ax.set_title("Electricity Usage Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Power Consumption (W)")
    ax.legend()
    st.pyplot(fig)
except FileNotFoundError:
    st.error(f"Dataset file '{data_path}' not found. Please ensure it is in the correct directory.")

# Input fields for user data
st.header("Enter Consumption Data")

# Create simple input fields (adjust based on your model's input structure)
feature1 = st.number_input("Feature 1", min_value=0.0, max_value=1000.0, value=10.0)
feature2 = st.number_input("Feature 2", min_value=0.0, max_value=1000.0, value=20.0)
feature3 = st.number_input("Feature 3", min_value=0.0, max_value=1000.0, value=30.0)

# Prepare input data as DataFrame (adjust column names as needed)
input_df = pd.DataFrame(
    [[feature1, feature2, feature3]],
    columns=['Appliance', 'Voltage (V)', 'Power Consumption (W)']  # Replace with your actual feature names
)

# Display prediction when the button is clicked
if st.button("Predict Consumption"):
    # Assuming the model is defined in the notebook and available as 'model'
    try:
        prediction = model.predict(input_df)
        st.write("Predicted Consumption:", prediction)
    except NameError:
        st.error("Model is not defined. Please ensure the model is loaded correctly.")
# Optional: Add a simple plot if relevant
if st.button("Show Sample Plot"):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [feature1, feature2, feature3], label='Sample Data')
    ax.set_title("Sample Feature Plot")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Value")
    ax.legend()
    st.pyplot(fig)
