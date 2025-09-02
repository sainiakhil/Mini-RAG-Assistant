import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import tempfile
import dotenv


# Load environment variables from .env file
dotenv.load_dotenv()

# --- Configuration ---
# Pinecone
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "mini-rag" # As requested

# LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# --- Helper Functions ---
def load_and_split_document(file):
    """Loads and splits a document into chunks."""
    # Use tempfile to get a cross-platform temporary directory
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.name)

    # Save the uploaded file temporarily to be accessed by loaders
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    file_extension = os.path.splitext(file.name)[1]
    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == ".txt":
        loader = TextLoader(temp_file_path)
    else:
        st.error("Unsupported file format. Please upload a .pdf or .txt file.")
        return None

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    # Clean up the temporary file after loading
    os.remove(temp_file_path)

    return text_splitter.split_documents(documents)

def upsert_to_pinecone(chunks, index_name):
    """
    Creates a Pinecone index if it doesn't exist and upserts document chunks.
    Pinecone will handle the embedding creation internally.
    If the index exists, it will be deleted and recreated to ensure integrated embedding configuration.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if the index exists. If it does, delete it to ensure correct configuration.
    if pc.has_index(index_name):
        pc.delete_index(index_name)
        # A short wait to ensure the index is fully deleted
        time.sleep(5)

    # Create the index with integrated embedding
    st.info(f"Creating new Pinecone serverless index: {index_name}")
    pc.create_index_for_model(
          name=index_name,
          cloud="aws",
          region="us-east-1",
          embed={
                    "model":"llama-text-embed-v2",
                    "field_map":{"text": "chunk_text"}
                }
    )
    # A short wait to ensure the index is ready
    time.sleep(10)

    index = pc.Index(index_name)

    # Convert chunks into the format Pinecone accepts for serverless text embedding
    converted_chunks = []
    for i, chunk in enumerate(chunks):
      converted_chunks.append({
        "_id": f"chunk_{i}",  # Create a unique ID for each chunk
        "chunk_text": chunk.page_content,
        "metadata": str(chunk.metadata)
    })

    # Upsert the records into a namespace
    index.upsert_records("example-namespace", converted_chunks)
    return index

def retrieve_and_rerank(query, index):
    """Retrieves chunks from Pinecone and reranks them using Cohere."""
    # Pinecone's serverless search with integrated embeddings
    # Note: The query format is different for serverless text search
    # We must provide the text directly to be embedded by Pinecone
    reranked_docs = index.search(
    namespace="example-namespace",
    query={
        "top_k": 5,
        "inputs": {
            'text': query
        }
    },
    rerank={
        "model": "bge-reranker-v2-m3",
        "top_n": 3,
        "rank_fields": ["chunk_text"]
    }   
    )
    return reranked_docs

def format_docs(docs):
    """Formats retrieved documents for the LLM prompt."""
    return "\n\n".join(f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

# --- Streamlit UI ---
st.set_page_config(
    page_title="Mini RAG App", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimal, clean styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .step-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .status-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
    .status-waiting {
        background-color: #fff3cd;
        color: #856404;
    }
    .source-box {
        background: #f1f3f4;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1976d2;
    }
    .metric-box {
        background: #e3f2fd;
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .answer-container {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🔍 Mini RAG Assistant")
st.markdown("*Upload documents and ask questions with AI-powered search*")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# --- Document Upload and Processing ---
st.markdown('<div class="step-container">', unsafe_allow_html=True)

# Status indicator
if st.session_state.document_processed:
    st.markdown(f'<div class="status-indicator status-ready">✓ Document Ready: {st.session_state.document_name} ({st.session_state.chunk_count} chunks)</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-indicator status-waiting">⏳ No document loaded</div>', unsafe_allow_html=True)

st.subheader("📄 Document Input")

# Create two columns for file upload and text input
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Document", 
        type=["pdf", "txt"],
        help="Supported formats: PDF, TXT"
    )

with col2:
    pasted_text = st.text_area(
        "Or paste text directly", 
        height=100,
        placeholder="Paste your text content here..."
    )

# Process button - full width
if uploaded_file or pasted_text:
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        start_time = time.time()
        
        # Determine document name
        doc_name = uploaded_file.name if uploaded_file else "Pasted Text"
        
        with st.spinner(f"Processing {doc_name}..."):
            if uploaded_file:
                chunks = load_and_split_document(uploaded_file)
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.create_documents([pasted_text])

            if chunks:
                try:
                    st.session_state.pinecone_index = upsert_to_pinecone(chunks, PINECONE_INDEX_NAME)
                    st.session_state.document_processed = True
                    st.session_state.document_name = doc_name
                    st.session_state.chunk_count = len(chunks)
                    
                    processing_time = time.time() - start_time
                    st.success(f"✅ Document processed successfully in {processing_time:.2f}s!")
                    st.rerun()  # Refresh to show updated status
                except Exception as e:
                    st.error(f"❌ Failed to process document: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# --- Query Section ---
st.markdown('<div class="step-container">', unsafe_allow_html=True)
st.subheader("❓ Ask Questions")

if not st.session_state.document_processed:
    st.info("📝 Please upload and process a document first to enable querying.")
    query_disabled = True
else:
    query_disabled = False

query = st.text_input(
    "Enter your question", 
    placeholder="What would you like to know about the document?",
    disabled=query_disabled
)

if st.button("🔍 Get Answer", type="primary", disabled=query_disabled or not query, use_container_width=True):
    start_time = time.time()
    
    with st.spinner("Searching for answers..."):
        index = st.session_state.pinecone_index
        retrieved_docs = retrieve_and_rerank(query, index)

        if not retrieved_docs:
            st.warning("⚠️ No relevant information found for your query.")
        else:
            llm = ChatGroq(model_name=LLM_MODEL, groq_api_key=GROQ_API_KEY)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant that answers questions based on provided context. 
                    Use the following context documents to answer the user's question. Include inline citations [1], [2], etc. 
                    that refer to the document numbers provided. If you cannot find relevant information in the context, 
                    say so clearly.

                    Context Documents:
                    {context}"""),
                ("human", "{input}")
            ])

            # Chain is now simpler as context is passed directly
            rag_chain = (
                prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke({"context": retrieved_docs, "input": query})
            generation_time = time.time() - start_time

            # Display Answer
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.subheader("💡 Answer")
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)

            # Display Sources in expandable section
            with st.expander("📚 View Sources", expanded=False):
                for i, hit in enumerate(retrieved_docs['result']['hits']):
                    st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                    st.markdown(f"**Source {i+1}:**")
                    st.write(hit['fields']['chunk_text'])
                    st.markdown('</div>', unsafe_allow_html=True)

            # Performance Metrics in sidebar-style box
            with st.expander("⚡ Performance Metrics", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Response Time", f"{generation_time:.2f}s")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    input_tokens = len(query.split()) + sum(len(doc['fields']['chunk_text'].split()) for doc in retrieved_docs['result']['hits'])
                    output_tokens = len(answer.split())
                    total_tokens = input_tokens + output_tokens
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Total Tokens", f"{total_tokens:,}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    cost = (input_tokens * 0.05 + output_tokens * 0.25) / 1_000_000
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Est. Cost", f"${cost:.6f}")
                    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Powered by Pinecone • Groq • LangChain"
    "</div>", 
    unsafe_allow_html=True
)