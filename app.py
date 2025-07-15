import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def create_chat_chain(file_path, openai_model):
    """
    Create a conversational chain for the chat interface.
    
    Args:
        file_path (list): List of PDF file paths to process
        openai_model (str): Name of the OpenAI model to use
        
    Returns:
        ConversationalRetrievalChain: Configured chat chain
    """
    # Load documents
    print("Loading documents...")
    docs = []
    for filepath in file_path:
        try:
            print(f"Loading {filepath}...")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File exists: {os.path.exists(filepath)}")
            
            # Try to open the file directly to check for permissions
            try:
                with open(filepath, 'rb') as f:
                    print("File opened successfully")
            except Exception as e:
                print(f"Failed to open file: {str(e)}")
                raise
                
            # Try to import pypdf and get version
            try:
                import pypdf
                print(f"PyPDF version: {pypdf.__version__}")
            except ImportError as e:
                print(f"PyPDF import error: {str(e)}")
                raise
                
            # Try to load the PDF
            try:
                loader = PyPDFLoader(filepath)
                loaded_docs = loader.load()
                print(f"Successfully loaded {len(loaded_docs)} pages from {filepath}")
                docs.extend(loaded_docs)
            except Exception as e:
                print(f"Error in PyPDFLoader: {str(e)}")
                raise
                
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if not docs:
        raise ValueError("No documents were loaded. Please check the file paths and try again.")

    # Split text into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_documents = text_splitter.split_documents(docs)
    print(f"Split into {len(split_documents)} chunks")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create vector store
    print("Creating vector store...")
    vectordb = FAISS.from_documents(split_documents, embeddings)
    
    # Set up retriever
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Initialize the language model
    print("Initializing language model...")
    llm = ChatOpenAI(
        model=openai_model,
        temperature=0.2,
        max_tokens=1500
    )
    
    # Create the conversational chain
    print("Creating conversation chain...")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    print("Chat chain created successfully!")
    return chain

# Global variable to store the chat chain
chat_chain = None

def process_user_input(user_message, chat_history):
    """
    Process user input and generate a response using the chat chain.
    
    Args:
        user_message (str): User's message
        chat_history: Current chat history
        
    Returns:
        str: Generated response
    """
    global chat_chain
    if chat_chain is None:
        return "Error: Chat chain not initialized. Please restart the application."
        
    try:
        response = chat_chain({"question": user_message, "chat_history": chat_history})
        return response["answer"]
    except Exception as e:
        print(f"Error processing input: {str(e)}")
        return "I'm sorry, I encountered an error processing your request. Please try again."

def handle_message(user_message, history):
    """
    Handle incoming messages and update chat history.
    
    Args:
        user_message (str): User's message
        history: Current chat history
        
    Returns:
        tuple: Empty message and updated history
    """
    if not user_message.strip():
        return "", history
    
    # Get response from the chain
    response = process_user_input(user_message, history)
    
    # Update chat history
    history.append((user_message, response))
    
    return "", history

def clear_chat():
    """
    Clear the conversation history.
    
    Returns:
        list: Empty list to clear the chat interface
    """
    global chat_chain
    if chat_chain is not None and hasattr(chat_chain, 'memory') and chat_chain.memory is not None:
        chat_chain.memory.clear()
    return []

if __name__ == "__main__":
    # Configuration
    openai_model = "gpt-3.5-turbo"  # Using a more reliable model
    file_path = ["sat-practest.pdf"]  # Make sure this file exists in the same directory
    
    try:
        print("Initializing RAGBot...")
        # Initialize the chat chain
        chat_chain = create_chat_chain(file_path, openai_model)
        
        # Create the Gradio interface
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # RAGBot with Memory
            Ask questions about your PDF document. The bot will use the content of the document to answer your questions.
            """)
            
            # Chat interface
            chatbot = gr.Chatbot(height=500, label="Chat History")
            msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            clear = gr.Button("Clear Conversation")
            
            # Example questions
            examples = [
                "What is this document about?",
                "Can you summarize the key points?",
                "What are the main topics covered?"
            ]
            
            # Add examples
            with gr.Row():
                for example in examples:
                    gr.Examples(
                        examples=[[example]],
                        inputs=msg,
                        label=f"Try: {example[:30]}..."
                    )
            
            # Event handlers
            msg.submit(
                handle_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear.click(
                clear_chat,
                outputs=chatbot
            )
        
        # Launch the app
        print("Launching the application...")
        demo.queue().launch(debug=True, share=True)
        
    except Exception as e:
        print(f"Error initializing the application: {str(e)}")
        print("Please make sure all required files exist and try again.")
