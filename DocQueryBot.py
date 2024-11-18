from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline, AutoTokenizer
from langchain.memory import ConversationBufferMemory
import torch 
import gradio as gr
device = 'cuda' if torch.cuda.is_available() else 'cpu'

chat_history = [
    {"role": "system", "content": "Provide answers strictly based on the document content. If the requested information is not found in the document, respond that the information is not available in the document. Do not generate or assume answers outside of the document's contents. Answer only with information directly found in the document."}
]
# Function to load and process the document
def load_and_embed_document(file_path, persist_directory="docs/chroma"):
    # Load the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print("PDF text extracted successfully.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Set up embeddings with HuggingFace
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize Chroma vector store and add documents
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
    print("Chroma collection created and embeddings stored.")

    return vectordb

# Function to create the conversational chain
def create_conversational_chain(vectordb):
    # Load the model and tokenizer
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if missing
    # Set up the Hugging Face model for response generation
    generator = pipeline(
        "text-generation",
        model=model_id, 
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float32,
        max_new_tokens=150,  
        do_sample=True,
        temperature=0.5,
        )
    # Wrap the pipeline with HuggingFacePipeline for LangChain compatibility
    llm = HuggingFacePipeline(pipeline=generator)

    # Define the conversational retrieval chain
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer",
        )
    
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = ConversationalRetrievalChain.from_llm( 
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        output_key="answer",
        )

    return qa_chain

# Gradio UI setup
def chatbot_interface(file, user_input, chat_history=[]):
    # Load the document and initialize vector store
    vectordb = load_and_embed_document(file.name)
    qa_chain = create_conversational_chain(vectordb)
    
    
    # Get the response from the conversational chain
    response = qa_chain({"question": user_input, "chat_history": chat_history})     

    # Extract only the answer text using regex, ignoring metadata or context instructions
    full_answer = response.get("answer", "").strip()

    # Remove everything up to and including "Helpful Answer:" to show only the clean response
    if "Helpful Answer:" in full_answer:
        answer_start = full_answer.index("Helpful Answer:") + len("Helpful Answer:")
        clean_answer = full_answer[answer_start:].strip()
    else:
        clean_answer = full_answer  

    chat_history.append((user_input, clean_answer))

    return chat_history, chat_history

# Gradio app setup
with gr.Blocks() as demo:
    gr.Markdown("# Document-based Chatbot")
    with gr.Row():
        file = gr.File(label="Upload a PDF Document")
    user_input = gr.Textbox(label="Enter your question")
    chat_history = gr.State([])  # Store chat history

    # Display chat messages
    chatbot = gr.Chatbot()
    user_input.submit(chatbot_interface, inputs=[file, user_input, chat_history], outputs=[chatbot, chat_history])

# Launch the Gradio app
demo.launch(server_name="0.0.0.0", server_port=8080)
