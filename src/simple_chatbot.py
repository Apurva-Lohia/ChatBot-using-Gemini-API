import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import time


st.set_page_config(
    page_title="ChatBot",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  
)

st.title("ChatBot")

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')
    
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history = [])

for message in st.session_state.chat_session.history:
    with st.chat_message(message.role):
        st.markdown(message.parts[0].text)

def response_generator(prompt):
    """Generates a response (in stream) to the user's prompt"""
    response = st.session_state.chat_session.send_message(prompt)
    return response 

if prompt := st.chat_input("Say something..."):
    #st.session_state.chat_session.history.append({"role": "user", "parts": [{"text": prompt}]})
    with st.chat_message("user"):
        st.markdown(prompt)
   
    response = response_generator(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for word in response.text.split():
            full_response += word + " "
            placeholder.write(full_response + "▌")  # Update the placeholder dynamically
            time.sleep(0.05)

    #st.session_state.chat_session.history.append({"role": "assistant", "parts": [{"text": response}]})

    