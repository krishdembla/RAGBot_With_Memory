import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variable to store the chat chain and vector store
chat_chain = None
vectordb = None

# For debugging
import sys
print("Python version:", sys.version)

def create_chat_chain(file_path):
    """
    Create a conversational chain for the chat interface.
    
    Args:
        file_path: Path to the uploaded PDF file
        
    Returns:
        tuple: (status_message, chain, vector_store)
    """
    global vectordb
    
    try:
        # Load the PDF document
        print(f"Loading document: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        if not docs:
            return "Error: No content could be extracted from the PDF.", None, None
            
        print(f"Successfully loaded {len(docs)} pages from the document")

        # Split text into chunks
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size for better context
            chunk_overlap=300,  # Increased overlap for better temporal context
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
        
        # Set up retriever with more context
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,  # Increased number of documents to retrieve
                "fetch_k": 20  # Increased fetch size for better context
            }
        )

        # Set up memory with more context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )

        # Initialize the language model
        print("Initializing language model...")
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1500
        )
        
        # Create the conversational chain with better configuration
        print("Creating conversation chain...")
        from langchain.prompts import PromptTemplate
        from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
        
        # Create a proper prompt template
        prompt_template = """Use the provided documents to answer the question.
        Be specific and accurate with temporal references.
        If you cannot find the information in the documents, say so clearly.
        
        {context}
        
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the chain with proper configuration
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PROMPT
            }
        )
        
        print("\nChain configuration:")
        print(f"LLM model: gpt-3.5-turbo")  # Hardcoded since we know the model
        print(f"Retriever config: {retriever.search_kwargs}")
        print(f"Memory type: {type(memory).__name__}")
        print(f"Prompt template: {prompt_template}")
        
        print("Chat chain created successfully!")
        return "Document processed successfully! You can now ask questions about the PDF.", chain, vectordb
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, None, None

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
        return "Please upload a PDF file first."
        
    try:
        print(f"Processing question: {user_message}")
        print(f"Current chat history: {len(chat_history)} messages")
        
        # Get response from the chain
        response = chat_chain({"question": user_message, "chat_history": chat_history})
        
        return response["answer"]
        
    except Exception as e:
        import traceback
        print("\nError details:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        
        if isinstance(e, ValueError):
            error_msg = f"Value error: {str(e)}"
        else:
            error_msg = "An unexpected error occurred. Please try again."
        
        return error_msg

def handle_message(user_message, chat_history):
    """
    Handle incoming messages and update chat history.
    
    Args:
        user_message (str): User's message
        history: Current chat history
        
    Returns:
        tuple: Empty message and updated history
    """
    if not user_message.strip():
        return "", chat_history
    
    # Get response from the chain
    response = process_user_input(user_message, chat_history)
    
    # Update chat history
    chat_history.append((user_message, response))
    
    return "", chat_history

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

def process_uploaded_file(file):
    """
    Process the uploaded PDF file and initialize the chat chain.
    
    Args:
        file: Uploaded file object from Gradio
        
    Returns:
        str: Status message
    """
    global chat_chain, vectordb
    
    if file is None:
        return "No file uploaded. Please upload a PDF file."
    
    try:
        # Get the file path from the Gradio file object
        file_path = file.name
        
        # Validate file type
        if not file_path.lower().endswith('.pdf'):
            return "Error: Only PDF files are supported. Please upload a PDF file."
            
        # Create the chat chain with the uploaded file
        status_msg, new_chain, new_vectordb = create_chat_chain(file_path)
        
        if new_chain is not None and new_vectordb is not None:
            chat_chain = new_chain
            vectordb = new_vectordb
            
        return status_msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing file: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # RAGBot with Memory
    Upload a PDF file and ask questions about its content.
    """)
    
    # File upload component
    file_output = gr.File(label="Upload a PDF file")
    upload_btn = gr.Button("Process PDF")
    status = gr.Markdown("Please upload a PDF file to begin.")
    
    # Chat interface
    chatbot = gr.Chatbot(height=400, label="Chat History")
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    
    # Action buttons
    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Conversation")
    
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
                label=f"Try: {example[:20]}..."
            )
    
    # Event handlers
    upload_btn.click(
        process_uploaded_file,
        inputs=file_output,
        outputs=status
    )
    
    def handle_click(user_message, history):
        return handle_message(user_message, history)
    
    msg.submit(
        handle_click,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    send_btn.click(
        handle_click,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=chatbot
    )

# Launch the interface
if __name__ == "__main__":
    print("Starting RAGBot application...")
    demo.launch(share=False)
