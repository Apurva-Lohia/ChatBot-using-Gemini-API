import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import time
from mongodb_embeddings import search_user_uploaded_docs, generate_text_embeddings


st.set_page_config(
    page_title="Gemini Pro - ChatBot",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  
)

st.title("🤖 Chat with Gemini Pro")


#User can add files to along with their prompt
uploaded_files = st.file_uploader(
    "Attach documents here (txt or pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)
if uploaded_files:
    user_uploaded_docs = generate_text_embeddings(uploaded_files)

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

gen_ai.configure(api_key=GOOGLE_API_KEY)

# Chat model configuration
chat_generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=chat_generation_config,
)    

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history = [])

# Check if the history is empty before trying to access it
if st.session_state.chat_session.history:
    # Display only the most recent exchange in the history
    for message in st.session_state.chat_session.history[-1:]:  # Only display the latest message
        with st.chat_message(message.role):
            st.markdown(message.parts[0].text)

def response_generator(prompt):
    """Generates a response (in stream) to the user's prompt"""
    response = st.session_state.chat_session.send_message(prompt)
    return response 

def response_generator_with_user_docs(prompt, user_docs):
    """
    Generates a response using the user's prompt and relevant context
    from uploaded documents, but only returns the final answer.
    """
    # Retrieve relevant contexts from user-uploaded documents
    relevant_contexts = search_user_uploaded_docs(prompt, user_docs, top_k=1)
    combined_context = " ".join(relevant_contexts)

    # Combine prompt and context internally
    combined_input = f"{combined_context}\n{prompt}"

    response = st.session_state.chat_session.send_message(combined_input)

    return response

if prompt := st.chat_input("Say something..."):
    #st.session_state.chat_session.history.append({"role": "user", "parts": [{"text": prompt}]})
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_files:   
        response = response_generator_with_user_docs(prompt, user_uploaded_docs)
    else:
        response = response_generator(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for word in response.text.split():
            full_response += word + " "
            placeholder.write(full_response + "▌")  # Update the placeholder dynamically
            time.sleep(0.05)

    #st.session_state.chat_session.history.append({"role": "assistant", "parts": [{"text": response.text}]})

    