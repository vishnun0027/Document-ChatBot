import streamlit as st
import os
import tempfile
from utils import write_message
from chat import generate_response, setup_rag_pipeline
from chat import DocumentLoader

# Page Configuration
st.set_page_config(page_title="Document ChatBot", page_icon="🤖")

# Title
st.title("Document ChatBot 📰🤖")
st.sidebar.header("Document")
st.sidebar.info("A Document ChatBot retrieves and summarizes information from \
                uploaded files or URLs, answering user questions by providing relevant, context-based responses")

# Initialize vectorstore and app in session state if they don't exist
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'app' not in st.session_state:
    st.session_state.app = None

# URL input for document loading
url_input = st.sidebar.text_input("Enter a URL:")
# Upload PDF files
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Initialize the DocumentLoader
document_loader = DocumentLoader()

if st.sidebar.button("Proceed"):
    with st.spinner("Processing... Please wait."):
        try:
            # Check if either URL or uploaded file is provided
            if uploaded_file is not None or (url_input and url_input.strip()):
                # Load document from URL
                if url_input and url_input.strip():
                    document_loader.load_web(url_input)

                # Load PDF from uploaded file
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())  # Write the uploaded file's contents to the temp file
                        temp_file_path = temp_file.name

                    document_loader.load_pdf(temp_file_path)
                    os.remove(temp_file_path)  # Clean up the temp file

                # Create vector store and store it in session state
                st.session_state.vectorstore = document_loader.vector_embedding()
                st.session_state.app = setup_rag_pipeline(st.session_state.vectorstore)  # Set app here
                st.sidebar.success("File uploaded successfully!")
            else:
                st.sidebar.warning("Please upload a PDF file or enter a URL.")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Set up Session State for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm your Chatbot! How can I help you?  You can upload \
         a document or enter a URL, and I'll answer your questions based on the content.?"}
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler to process user input and generate a response.
    """
    with st.spinner('Thinking...'):
        if st.session_state.app is not None:  # Check if app is defined
            response = generate_response(st.session_state.app, message)
            write_message('assistant', response)
        else:
            write_message('assistant', "Please upload a PDF or enter a URL to start.")

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
