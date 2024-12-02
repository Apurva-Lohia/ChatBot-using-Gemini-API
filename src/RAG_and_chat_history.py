import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import time
from mongodb_embeddings import search_user_uploaded_docs, generate_text_embeddings
import json
import datetime

# Set the Streamlit page config
st.set_page_config(
    page_title="Gemini Pro - ChatBot",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  
)

st.title("🤖 Chat with Gemini Pro")

# Sidebar for managing chat history
st.sidebar.title("Chat History")

# Load the API key for the model
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Chat model configuration
chat_generation_config = {
    "temperature": 1.0,  # Default single float value
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=chat_generation_config,
)

# Add sliders for model configuration to the existing sidebar
chat_generation_config["temperature"] = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, max_value=2.0, 
    value=chat_generation_config["temperature"], 
    step=0.1
)

chat_generation_config["top_p"] = st.sidebar.slider(
    "Top-p", 
    min_value=0.0, max_value=1.0, 
    value=chat_generation_config["top_p"], 
    step=0.05
)

chat_generation_config["top_k"] = st.sidebar.slider(
    "Top-k", 
    min_value=0, max_value=100, 
    value=chat_generation_config["top_k"], 
    step=1
)

# Add input box for max_output_tokens
chat_generation_config["max_output_tokens"] = st.sidebar.number_input(
    "Max Output Tokens", 
    min_value=1, max_value=10000, 
    value=chat_generation_config["max_output_tokens"], 
    step=50
)

# Allow file upload for user documents
uploaded_files = st.file_uploader(
    "Attach documents here (txt or pdf)",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)
if uploaded_files:
    user_uploaded_docs = generate_text_embeddings(uploaded_files)



# To store chat histories
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  
 # To track the active chat session
if "current_chat_session_id" not in st.session_state:
   st.session_state.current_chat_session_id = None 
# To store the actual chat session object 
if "chat_session_object" not in st.session_state:
    st.session_state.chat_session_object = None 

# Function to create a new chat
def start_new_chat():
    # Unique chat ID
    chat_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Initialize an empty history
    st.session_state.chat_sessions[chat_id] = []  
    # Start a new chat session object
    st.session_state.chat_session_object = model.start_chat(history=[])  
    # Set as active chat
    st.session_state.current_chat_session_id = chat_id  
    return chat_id

# Handle "Start New Chat" button
if st.sidebar.button("Start New Chat"):
    selected_chat = start_new_chat()

# Load existing chats in the sidebar
existing_chats = list(st.session_state.chat_sessions.keys())
if existing_chats:
    selected_chat = st.sidebar.selectbox(
        "Select a previous chat", existing_chats
        )
else:
    selected_chat = None



# If no chat is selected, start a new one
if selected_chat is None and st.session_state.current_chat_session_id is None:
    selected_chat = start_new_chat()

# Set the current chat session ID and object
if selected_chat and st.session_state.current_chat_session_id != selected_chat:
    st.session_state.current_chat_session_id = selected_chat
    if selected_chat not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[selected_chat] = []
    st.session_state.chat_session_object = model.start_chat(
        history=st.session_state.chat_sessions[selected_chat]
    )

# Load the chat history from the session
if st.session_state.current_chat_session_id:
    history = st.session_state.chat_sessions.get(
        st.session_state.current_chat_session_id, []
        )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Display the chat history
if history:
    for message in history:
        with st.chat_message(translate_role_for_streamlit(message["role"])):
            st.markdown(message["parts"]["text"])


def response_generator(prompt):
    """Generates a response (in stream) to the user's prompt"""
    response = st.session_state.chat_session_object.send_message(prompt)
    return response

# Function to send user message and get response
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

    response = st.session_state.chat_session_object.send_message(combined_input)

    return response

if prompt := st.chat_input("Say something..."):
    # Save the user's message
    user_message = {"role": "user", "parts": {"text": prompt}}
    st.session_state.chat_sessions[selected_chat].append(user_message)
    
    # Show the user message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the response based on document uploads
    if uploaded_files:
        response = response_generator_with_user_docs(prompt, user_uploaded_docs)
    else:
        response = response_generator(prompt)

    # Display assistant's response
    assistant_message = {"role": "model", "parts": {"text": response.text}}
    st.session_state.chat_sessions[selected_chat].append(assistant_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for word in response.text.split():
            full_response += word + " "
            # Update the placeholder dynamically
            placeholder.write(full_response + "▌")  
            time.sleep(0.05)

    # Optionally, store chat history to disk in JSON format for persistence
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_sessions, f)
