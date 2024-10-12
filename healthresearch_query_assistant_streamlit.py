# RETRIEVAL-AUGMENTED GENERATION (RAG) 

# streamlit run healthresearch_query_assistant_streamlit.py
# Chat Q&A Framework for RAG Apps

# Imports 
import streamlit as st
from dotenv import load_dotenv
import yaml
import os
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import uuid
import fitz  # PyMuPDF for PDF text extraction
from fpdf import FPDF  # For generating PDF files

# Initialize Streamlit app

openai_api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Arxiv AI in Healthcare Copilot", layout="wide")

# Custom Header for Branding
st.markdown(
    """
    <div style="text-align:center; margin-bottom:20px;">
        <h1>🤖 AI in Healthcare Research Assistant</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the API Key securely
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize session state for message history
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Display initial message if there's no history
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hey! I'm here to help you with writing research papers in AI and healthcare. What paper would you like to write today?")

# Sidebar for Customization
with st.sidebar:
    st.header("🔧 Customization Settings")
    st.markdown(
        """
        - Adjust the temperature for more creative answers.
        - Choose if you'd like a .tex or .pdf download of your research paper.
        - Select the type of report: Detailed or Summary.
        """
    )
    temperature = st.slider("Response Creativity (Temperature)", 0.0, 1.0, 0.8)
    file_format = st.selectbox("Download Format", ['.tex', '.pdf'])
    report_type = st.selectbox("Report Type", ['Detailed', 'Summary'])

# Upload PDF files
uploaded_files = st.file_uploader("Upload your own PDF documents for analysis", type=["pdf"], accept_multiple_files=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        return text
    except fitz.FileDataError:
        st.error(f"Error: Failed to open {pdf_path}. The file might be corrupt or invalid.")
        return None
    except Exception as e:
        st.error(f"Unexpected error occurred while processing {pdf_path}: {e}")
        return None

# Function to format citations in APA and MLA styles
def format_citation(author, title, year, style='APA'):
    if style == 'APA':
        return f"{author}. ({year}). {title}."
    elif style == 'MLA':
        return f"{author}. \"{title}.\" {year}."
    return ""

# Function to create the RAG chain
# Function to create the RAG chain
def create_rag_chain(api_key):
        embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)
        vectorstore = Chroma(persist_directory="data/chroma_store", embedding_function=embedding_function)
        
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, api_key=api_key, max_tokens=3500)

        # Rest of the setup as before...
    


def create_rag_chain(api_key):
    try:
        embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)
        vectorstore = Chroma(persist_directory="data/chroma_store", embedding_function=embedding_function)
    
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, api_key=api_key, max_tokens=3500)
    
        # Contextualizing question based on history
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
    
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answering the user's question
        qa_system_prompt = """You are an AI in healthcare research writing expert for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Add subsections and categorize it into sections based on the user prompt. \
        Use the provided context to write a detailed response. \
        Add references and citations while writing. \
        Make sure you add refferences and citations to the generated research document. \
        Add them using the 
        Use bullet points if necessary. \
        Add a conclusion at the end of the document. \
        Use as many words as needed to make it comprehensive. \
        Avoid plagiarism and add references and citations. \
        Make sure it is comprehensive and detailed like a research paper as standard as that of NeurIPS or ICML accepted papers. \
        If you don't know any answer, don't try to make up an answer. Just say that you don't know and to contact the owner of the chatbot @chisomruth1212@gmail.com. \
        Also give the option of downloading the generated research document written as a .tex document. \
        If the person agrees to downloading the document as a .tex file, just convert everything you have written into a .tex file and put it up for download. \
        If the question asked is not related to AI in healthcare, tell the user you are not capable of generating research contents in that domain. \
        Don't be overconfident and don't hallucinate. Ask follow-up questions if necessary or if there are several offerings related to the user's query. Provide answers with complete details in a properly formatted manner. \
        Context: {context}"""  # Ensure that context is included
    
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
        # Combine RAG + Message History
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None
    
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

rag_chain = create_rag_chain(OPENAI_API_KEY)

# Display conversation
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Input box for user queries
if question := st.chat_input("Enter your question about AI in Healthcare Research"):
    with st.spinner("Writing your research section... please wait..."):
        st.chat_message("human").write(question)  # Display the user query

        # Get the response from RAG
        response = rag_chain.invoke(
            {"input": question}, 
            config={"configurable": {"session_id": "session_{}".format(uuid.uuid4())}}
        )
        # Display AI's response
        st.chat_message("ai").write(response['answer'])

        # Option to download generated research as a .tex file or .pdf
        if st.button("Download the generated research document"):
            generated_content = response['answer']
            if file_format == '.tex':
                # Generate and download a .tex file
                st.download_button(label="Download .tex file", data=generated_content, file_name="research_paper.tex")
            elif file_format == '.pdf':
                # Generate and download a .pdf file
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, generated_content)
                pdf_file_path = "research_paper.pdf"
                pdf.output(pdf_file_path)
                with open(pdf_file_path, "rb") as f:
                    st.download_button(label="Download .pdf file", data=f, file_name=pdf_file_path)

# Debugging - view messages
with st.expander("🔍 View Message History"):
    st.write("Message history initialized with `langchain_messages`:")
    st.json(st.session_state.langchain_messages)

