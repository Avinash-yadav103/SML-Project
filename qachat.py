from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai

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
