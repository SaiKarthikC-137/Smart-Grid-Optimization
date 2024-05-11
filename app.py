import streamlit as st
import random
import time

from llm_backend import initialize_system, process_query

# Placeholder for your LLM response function
def get_llm_response(model_name, query):
   if "chain" in st.session_state:
        response = process_query(st.session_state.chain, query)
        return response['answer']
# Define pages
def model_prediction():
    st.title("Model Prediction")
    model_name = st.session_state.model
    st.write(f"Using model: {model_name}")
    
    with st.form(key='predict_form'):
        cols = [st.columns(2) for _ in range(6)]
        input_data = []
        for i in range(12):
            with cols[i // 2][i % 2]:
                feature = st.number_input(f'Feature {i+1}', value=0.0)
                input_data.append(feature)
        submit_button = st.form_submit_button(label='Predict Stability')
        
        if submit_button:
            result = "Unstable"  # Placeholder for actual prediction
            prompt = f"Reaction times are 3.1, 7.6, 4.94, 9.85. Power values are 3.52, -1.12, -1.85, -0.55. Price Elasticity are 0.8, 0.45, 0.66, 0.82. The result is unstable.'"
            st.write(f'Prediction: **{result}**')
            prediction_exp = get_llm_response(model_name, prompt)
            print(prediction_exp)
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

if 'model' not in st.session_state:
    st.session_state.model = "Mixtral-8x7b-32768"  # default model
st.sidebar.selectbox("Select Model:", ["Llama3-70b-8192", "Mixtral-8x7b-32768", "Gemma-7b-it"], key="model")

if page == "Model Prediction":
    model_prediction()
elif page == "Chatbot":
    chatbot()
