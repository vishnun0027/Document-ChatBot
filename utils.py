import streamlit as st
import uuid

def write_message(role, content, save=True):
    """
    Save a message to the session state and display it in the UI.

    Args:
    - role (str): The role of the message sender (e.g., 'user' or 'assistant').
    - content (str): The content of the message.
    - save (bool): Whether to save the message to session state (default is True).
    """
    # Append to session state if save is True
    if save:
        if 'messages' not in st.session_state:
            st.session_state.messages = []  # Initialize if not present
        st.session_state.messages.append({"role": role, "content": content})

    # Write to the UI
    with st.chat_message(role):
        st.markdown(content)

def get_session_id():
    """
    Generate or retrieve a unique session identifier.

    Returns:
    - str: A unique session ID.
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    return st.session_state.session_id
