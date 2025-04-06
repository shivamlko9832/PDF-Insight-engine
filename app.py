import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import tempfile

# Load environment
load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HuggingFace_Token")

# Page Configuration
st.set_page_config(
    page_title="PDF Insight Engine | AI-Powered Document Analysis",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Dark Professional CSS Styling
st.markdown("""
    <style>
        :root {
            --primary: #1B263B;
            --secondary: #0D6EFD;
            --accent: #FF6B6B;
            --light: #3E4C59;
            --dark: #121E28;
            --success: #2ECC71;
            --warning: #F39C12;
            --danger: #C0392B;
        }

        .stApp {
            background-color: var(--dark);
            color: #dfe6ed;
        }

        .header-container {
            background: linear-gradient(135deg, var(--primary), var(--dark));
            padding: 1.5rem 2rem;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            color: #dfe6ed;
        }

        .sidebar .sidebar-content {
            background-color: var(--primary);
            border-right: 1px solid #2c3e50;
            color: #dfe6ed;
        }

        .card {
            background-color: var(--light);
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #2c3e50;
            color: #dfe6ed;
        }

        .card-title {
            color: var(--secondary);
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }

        .btn-primary {
            background-color: var(--secondary);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }

        .btn-primary:hover {
            background-color: #0b5ed7;
        }

        .btn-danger {
            background-color: var(--danger);
            color: white;
        }

        .chat-message-user {
            background-color: #2c3e50;
            border-radius: 12px 12px 0 12px;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 80%;
            float: right;
            clear: both;
            color: white;
        }

        .chat-message-bot {
            background-color: var(--secondary);
            color: white;
            border-radius: 12px 12px 12px 0;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            max-width: 80%;
            float: left;
            clear: both;
        }

        .feature-icon {
            font-size: 2rem;
            color: var(--secondary);
            margin-bottom: 1rem;
        }

        .feature-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }

        .footer {
            text-align: center;
            margin-top: 3rem;
            padding: 1.5rem;
            color: #bdc3c7;
            font-size: 0.9rem;
            border-top: 1px solid #2c3e50;
        }

        .file-pill {
            display: inline-block;
            background-color: #2c3e50;
            color: #dfe6ed;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            margin: 0.25rem;
            font-size: 0.8rem;
        }

        /* Input fields styling */
        .stTextInput input {
            background-color: #2c3e50;
            color: white;
            border: 1px solid #3E4C59;
        }

        .stTextInput input:focus {
            border-color: var(--secondary);
            box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.25);
        }

        /* Select box styling */
        .stSelectbox select {
            background-color: #2c3e50;
            color: white;
            border: 1px solid #3E4C59;
        }

        /* Button styling */
        .stButton button {
            background-color: var(--secondary);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }

        .stButton button:hover {
            background-color: #0b5ed7;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: var(--light);
            color: #dfe6ed;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background-color: var(--secondary);
            color: white;
        }

        /* Spinner color */
        .stSpinner > div {
            color: var(--secondary);
        }

        /* Developer info in header */
        .developer-contact {
            background: rgba(255,255,255,0.1);
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .developer-contact:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .header-container a {
            color: #bdc3c7;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .header-container a:hover {
            color: var(--secondary) !important;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Session Store
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Header with Developer Information
st.markdown("""
    <div class='header-container'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h1 style='color: white; margin-bottom: 0.5rem;'>PDF Insight Engine</h1>
                <p style='color: rgba(255,255,255,0.9); margin-bottom: 0;'>AI-Powered Document Analysis & Conversational RAG</p>
            </div>
            <div class='developer-contact'>
                <p style='margin: 0; font-size: 0.9rem; color: #bdc3c7;'>Developed by</p>
                <div style='display: flex; align-items: center; gap: 1rem; margin-top: 0.5rem;'>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' viewBox='0 0 16 16'>
                            <path d='M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0zm4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4zm-1-.004c-.001-.246-.154-.986-.832-1.664C11.516 10.68 10.289 10 8 10c-2.29 0-3.516.68-4.168 1.332-.678.678-.83 1.418-.832 1.664h10z'/>
                        </svg>
                        <span style='color: white; font-weight: 500;'>Shivam Kumar</span>
                    </div>
                    <a href='mailto:shivamlko9832@gmail.com' style='display: flex; align-items: center; gap: 0.5rem;'>
                        <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='currentColor' viewBox='0 0 16 16'>
                            <path d='M.05 3.555A2 2 0 0 1 2 2h12a2 2 0 0 1 1.95 1.555L8 8.414.05 3.555ZM0 4.697v7.104l5.803-3.558L0 4.697ZM6.761 8.83l-6.57 4.027A2 2 0 0 0 2 14h12a2 2 0 0 0 1.808-1.144l-6.57-4.027L8 9.586l-1.239-.757Zm3.436-.586L16 11.801V4.697l-5.803 3.546Z'/>
                        </svg>
                        <span>Email</span>
                    </a>
                    <a href='https://www.linkedin.com/in/shivamlko9832/' target='_blank' style='display: flex; align-items: center; gap: 0.5rem;'>
                        <svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='currentColor' viewBox='0 0 16 16'>
                            <path d='M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z'/>
                        </svg>
                        <span>LinkedIn</span>
                    </a>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## Configuration", unsafe_allow_html=True)
    
    with st.expander("🔑 API Settings", expanded=True):
        api_key = st.text_input("Groq API Key", type="password", help="Required for accessing the Groq LLM service")
        st.markdown("[Get Groq API Key](https://console.groq.com/)")
    
    with st.expander("📂 Document Upload", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents for analysis"
        )
    
    with st.expander("⚙️ Session Settings", expanded=True):
        session_id = st.text_input("Session ID", value="default", help="Maintain conversation history across sessions")
        user_name = st.text_input("Your Name", value="Analyst", help="For personalizing your experience")
        
        if st.button("🔄 Reset Session", type="secondary"):
            if 'store' in st.session_state and session_id in st.session_state.store:
                del st.session_state.store[session_id]
            st.success("Session history cleared!")
    
    st.markdown("---")
    st.markdown("""
        <div class='card'>
            <h4 class='card-title'>About PDF Insight Engine</h4>
            <p>Professional document analysis using RAG architecture and state-of-the-art LLMs.</p>
            <p><strong>Version:</strong> 1.0.0</p>
            <p><strong>Last Updated:</strong> June 2024</p>
        </div>
    """, unsafe_allow_html=True)

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📋 Dashboard", "💬 Document Chat", "⚙️ Advanced"])

with tab1:
    st.markdown("## Document Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='card feature-card'>
                <h3 class='card-title'>📊 Document Summary</h3>
                <p>Upload PDF documents to begin analysis. The system will automatically:</p>
                <ul>
                    <li>Extract text content</li>
                    <li>Chunk documents for processing</li>
                    <li>Generate vector embeddings</li>
                    <li>Prepare for conversational analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if uploaded_files:
            st.markdown("""
                <div class='card feature-card'>
                    <h3 class='card-title'>📂 Uploaded Documents</h3>
            """, unsafe_allow_html=True)
            
            for file in uploaded_files:
                st.markdown(f"""
                    <div class='file-pill'>
                        {file.name} ({round(file.size/1024)} KB)
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='card feature-card'>
                <h3 class='card-title'>🚀 Key Features</h3>
                <div>
                    <div class='feature-icon'>🔍</div>
                    <h4>Context-Aware Search</h4>
                    <p>Find answers within your documents with semantic understanding.</p>
                </div>
                <div>
                    <div class='feature-icon'>🧠</div>
                    <h4>Conversational Memory</h4>
                    <p>Maintains context across questions for natural interaction.</p>
                </div>
                <div>
                    <div class='feature-icon'>⚡</div>
                    <h4>High Performance</h4>
                    <p>Leverages Groq's ultra-fast LLM inference.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("## Document Conversation Interface")
    
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to enable chat functionality.")
        st.stop()
    
    if not uploaded_files:
        st.info("📁 Please upload PDF documents in the sidebar to begin analysis.")
        st.stop()
    
    # Initialize components
    with st.spinner("Initializing document processing..."):
        # Create temporary files
        temp_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_files.append(temp_file.name)
        
        # Process documents
        documents = []
        for temp_path in temp_files:
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error processing file {temp_path}: {str(e)}")
            finally:
                os.unlink(temp_path)
        
        if not documents:
            st.error("No valid documents could be processed. Please check your PDF files.")
            st.stop()
        
        # Split and embed documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(documents)
        
        # Initialize vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM and conversation chain
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="Llama3-8b-8192",
        temperature=0.3
    )
    
    # Define prompts
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and the latest user question which might reference context in the chat history, rephrase the question to be a standalone question that contains all the needed context. Return the standalone question verbatim. Do not answer the question."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    qa_system_prompt = """You are an expert document analyst. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Be precise and professional in your responses.
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create chains
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(
        llm, qa_prompt
    )
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Chat interface
    st.markdown("### Document Conversation")
    
    # Display chat history
    history = get_session_history(session_id)
    for msg in history.messages:
        if msg.type == "human":
            st.markdown(f"""
                <div class='chat-message-user'>
                    <strong>{user_name}</strong><br>
                    {msg.content}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message-bot'>
                    <strong>AI Analyst</strong><br>
                    {msg.content}
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input(f"Ask {user_name}'s document question...")
    
    if user_input:
        # Add user message to chat
        st.markdown(f"""
            <div class='chat-message-user'>
                <strong>{user_name}</strong><br>
                {user_input}
            </div>
        """, unsafe_allow_html=True)
        
        # Get and display AI response
        with st.spinner("Analyzing documents..."):
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                
                st.markdown(f"""
                    <div class='chat-message-bot'>
                        <strong>AI Analyst</strong><br>
                        {response['answer']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Add to session history
                session_history = get_session_history(session_id)
                session_history.add_user_message(user_input)
                session_history.add_ai_message(response["answer"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab3:
    st.markdown("## Advanced Settings & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='card feature-card'>
                <h3 class='card-title'>🛠️ System Configuration</h3>
                <p><strong>Embedding Model:</strong> all-MiniLM-L6-v2</p>
                <p><strong>Vector Database:</strong> ChromaDB</p>
                <p><strong>LLM Provider:</strong> Groq</p>
                <p><strong>Chunk Size:</strong> 5000 characters</p>
                <p><strong>Chunk Overlap:</strong> 500 characters</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='card feature-card'>
                <h3 class='card-title'>📈 Performance Metrics</h3>
                <p><strong>Document Processing:</strong> {doc_count} documents</p>
                <p><strong>Total Chunks:</strong> {chunk_count} segments</p>
                <p><strong>Session History:</strong> {msg_count} messages</p>
                <p><strong>Vector Store:</strong> Active</p>
            </div>
        """.format(
            doc_count=len(uploaded_files) if uploaded_files else 0,
            chunk_count=len(splits) if 'splits' in locals() else 0,
            msg_count=len(get_session_history(session_id).messages)
        ), unsafe_allow_html=True)
    
    st.markdown("""
        <div class='card feature-card'>
            <h3 class='card-title'>🔍 Retrieval Parameters</h3>
            <p>The system retrieves the 3 most relevant document chunks for each query, using cosine similarity on the embeddings.</p>
            <p>Conversation history is used to provide context for follow-up questions while maintaining relevance to the original documents.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        <p>PDF Insight Engine v1.0 | © 2024 Professional AI Solutions</p>
    </div>
""", unsafe_allow_html=True)