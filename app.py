import streamlit as st
import random
import time
import tensorflow as tf
import numpy as np
from llm_backend import initialize_system, process_query

# Placeholder for your LLM response function
def get_llm_response(model_name, query):
   if "chain" in st.session_state:
        response = process_query(st.session_state.chain, query)
        return response['answer']
   
def predict_result(features):
    model = tf.keras.models.load_model('my_model.keras')
    means = np.array([5.25605072, 5.2488346, 5.25248183, 5.2461366, 3.7502637, -1.24964552, -1.25013591, -1.25048227,
                      0.52429052, 0.52529141, 0.52481541, 0.52561655])
    stds = np.array([2.74287799, 2.74246835, 2.73935338, 2.7430897, 0.75333038, 0.43323333, 0.43255948, 0.43310764,
                     0.27362782, 0.27430814, 0.27440013, 0.27437782])
    data_point = np.array(features)
    normalized_data_point = (data_point - means) / stds
    normalized_data_point = normalized_data_point.reshape(1, -1)
    prediction = model.predict(normalized_data_point)[0]
    return 'Stable' if prediction > 0.5 else 'Unstable'

# Define pages
def model_prediction():
    st.title("Model Prediction")
    model_name = st.session_state.model
    st.write(f"Using model: {model_name}")
    st.session_state.chain = initialize_system(st.session_state.model)
    with st.form(key='predict_form'):
        cols = [st.columns(2) for _ in range(6)]
        input_data = []
        for i in range(12):
            with cols[i // 2][i % 2]:
                feature = st.number_input(f'Feature {i+1}', value=0.0)
                input_data.append(feature)
        submit_button = st.form_submit_button(label='Predict Stability')
        
        if submit_button:
            result = predict_result(input_data)
            prompt = f"Reaction times are {input_data[0]},{input_data[1]},{input_data[2]},{input_data[3]}. Power values are {input_data[4]},{input_data[5]},{input_data[6]},{input_data[7]}. Price Elasticity are {input_data[8]},{input_data[9]},{input_data[10]},{input_data[11]}. The result is {result}.'"
            st.write(f'Prediction: **{result}**')
            prediction_exp = get_llm_response(model_name, prompt)
            st.write(prediction_exp)

def chatbot():
    st.title("Chatbot")
    model_name = st.session_state.model

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.chain = initialize_system(st.session_state.model)
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display the assistant's response
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("asisstant"):
            llm_response = get_llm_response(model_name, prompt)
            response = st.write(llm_response)
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
        # Add the streamed assistant response to the history (note: workaround as streaming directly does not capture text)
        

# Main app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Model Prediction", "Chatbot"])
model_changed = False
if 'model' not in st.session_state:
    st.session_state.model = "Mixtral-8x7b-32768"  # default model
current_model = st.sidebar.selectbox("Select Model:", ["Llama3-70b-8192", "Mixtral-8x7b-32768", "Gemma-7b-it"], key="model")
if 'model' not in st.session_state or current_model != st.session_state.model:
    model_changed = True
    st.session_state.model = current_model

if model_changed:
    st.session_state.chain = initialize_system(st.session_state.model)
    st.session_state.messages = []
if page == "Model Prediction":
    model_prediction()
elif page == "Chatbot":
    chatbot()
