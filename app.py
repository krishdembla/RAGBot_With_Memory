import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.prompts import PromptTemplate
#from langfuse.callback import CallbackHandler

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_SK = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PK = os.getenv("LANGFUSE_PUBLIC_KEY")
HOST = os.getenv("LANGFUSE_HOST")

# langfuse_handler = CallbackHandler(
#     secret_key=LANGFUSE_SK,
#     public_key=LANGFUSE_PK,
#     host=HOST,
#     debug=True,
#     session_id="1505",
#     release="1.0.0",
#     user_id="Krish",
#     trace_name="WarehouseAst"
# )

# Function to create the chat chain
def create_chat_chain(file_path, openai_model):
    docs = []
    for filepath in file_path:
        loader = UnstructuredPDFLoader(filepath)
        docs.extend(loader.load())

    # Splitting txt
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(docs)

    #Creating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    vectordb = DocArrayInMemorySearch.from_documents(split_documents, embeddings)

    #Retriever setip
    retriever = vectordb.as_retriever(
        search_type='mmr'
    )

    #memory setup
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'  # Explicitly set the output key to avoid ambiguity 
    )

    llm_def = ChatOpenAI(model=openai_model, temperature=0, api_key=API_KEY, max_tokens=1100)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_def,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
        #callbacks=[langfuse_handler]
    )
    return chain

# Function to process user input
def process_input(chain, user_message):
    inputs = {"question": user_message}
    outputs = chain(inputs)
    return outputs['answer']

if __name__ == "__main__":
    openai_model = "gpt-3.5-turbo"
    file_path = [
        r"sat-practest.pdf"
        #r"Advantage Dashboard Functional Description.pdf",
        #r"user_guide.pdf"
    ]
    chain = create_chat_chain(file_path, openai_model)

    # Gradio UI
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("<h1 style='text-align: center; width: 100%;'>RAG with Memory</h1>")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Here's how to use this demo:**")
                gr.Markdown("""This a conversational agent responsible for summarizing, and giving key highlights about the atached pdf file (The SAT Practice Test 1) .\n
                Feel free to experiment with your own different queries and usage or accesibility requests!
                """)
            with gr.Column(scale=3):
                ex = ["Which question contains information about Navajo Nation legislator Annie Dodge Wauneka?", "What does question 7 in module 2 of reading and writing ask for?", "How many questions are there in moduule 2 of the reading and writing section?"]
                chatbot = gr.Chatbot()
                msg = gr.Textbox()
                gr.Examples(ex, msg)
                clear = gr.Button("Clear Chat History")

                def handle_message(user_message, history):
                    answer = process_input(chain, user_message)
                    history.append([user_message, answer])
                    return "", history

                msg.submit(handle_message, [msg, chatbot], [msg, chatbot])
                clear.click(lambda: chain.memory.clear(), None, chatbot)

    demo.queue()
    demo.launch()
