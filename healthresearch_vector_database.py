""" # RETRIEVAL-AUGMENTED GENERATION (RAG) BBUILDER

######
# - Langchain Document Loaders
# - Text Embeddings
# - Vector Databases

# LIBRARIES 

# from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import pandas as pd
import yaml
from pprint import pprint

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

import chromadb

from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer, AutoModel
import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# pip install PyMuPDF solves fitz problem
# OPENAI API SETUP pip install PyMuPDF

OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']


# 1.0 DATA PREPARATION ----

pdf_directory = "Data/pdf folder"
chroma_db_path = "Data/chroma_store"

# Ensure the ChromaDB path exists and has write permissions
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path)
    os.chmod(chroma_db_path, 0o775)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Load PDFs and extract text
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        documents.append({"text": text, "source": pdf_path})

# Define the HuggingFace model for embeddings
# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
# model = AutoModel.from_pretrained(embedding_model_name)

# # Define a function to create embeddings
# def create_embeddings(texts):
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#     outputs = model(**inputs)
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings.detach().numpy()

# Initialize the HuggingFaceEmbeddings
# hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

embedding = OpenAIEmbeddings(model='text-embedding-ada-002',api_key=OPENAI_API_KEY)

# Use a text splitter to handle large documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split texts and create embeddings
split_texts = [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]
embeddings = embedding.embed_documents(split_texts)

# Create and store embeddings in ChromaDB
chroma = Chroma(embedding_function=embedding, persist_directory=chroma_db_path)
chroma.add_texts(texts=split_texts, embeddings=embeddings, metadatas=[{'source': doc['source']} for doc in documents])
chroma.persist()




result = chroma.similarity_search("How can  AI impact healthcare?", k = 4)

pprint(result[0].page_content)

# # youtube_df = pd.read_csv('data/youtube_videos.csv')

# # youtube_df.head()

# # # * Text Preprocessing

# # youtube_df['page_content'] = youtube_df['page_content'].str.replace('\n\n', '\n', regex=False)

# # * Document Loaders
# #   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

# loader = DataFrameLoader(youtube_df, page_content_column='page_content')

# documents = loader.load()

# documents[0].metadata
# documents[0].page_content

# pprint(documents[0].page_content)

# len(documents)

# # * Text Splitting
# #   https://python.langchain.com/docs/modules/data_connection/document_transformers

# CHUNK_SIZE = 1000

# # Character Splitter: Splits on simple default of 
# text_splitter = CharacterTextSplitter(
#     chunk_size=CHUNK_SIZE, 
#     # chunk_overlap=100,
#     separator="\n"
# )

# docs = text_splitter.split_documents(documents)

# docs[0].metadata

# len(docs)

# # Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
# text_splitter_recursive = RecursiveCharacterTextSplitter(
#     chunk_size = CHUNK_SIZE,
#     chunk_overlap=100,
# )

# docs_recursive = text_splitter_recursive.split_documents(documents)

# len(docs_recursive)

# # * Text Embeddings

# # OpenAI Embeddings
# # - See Account Limits for models: https://platform.openai.com/account/limits
# # - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview

# embedding_function = OpenAIEmbeddings(
#     model='text-embedding-ada-002',
#     api_key=OPENAI_API_KEY
# )

# # Open Source Alternative:
# # Requires Torch and SentenceTransformer packages:
# # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# # * Langchain Vector Store: Chroma DB
# # https://python.langchain.com/docs/integrations/vectorstores/chroma

# # Creates a sqlite database called vector_store.db
# vectorstore = Chroma.from_documents(
#     docs, 
#     embedding=embedding_function, 
#     persist_directory="data/chroma_2.db"
# )

# vectorstore


# # * Similarity Search: The whole reason we did this

# result = vectorstore.similarity_search("How to create a social media strategy", k = 4)

# pprint(result[0].page_content)


 """
import arxiv
import requests
import os
import yaml
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from pprint import pprint
import fitz  # PyMuPDF for PDF text extraction

# Load credentials
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup paths
pdf_directory = "Data/ai_healthcare_papers"
chroma_db_path = "Data/chroma_store"

# Ensure the ChromaDB path exists and has write permissions
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path, 0o775)

# Function to extract text from PDF (optimized for multi-threading)
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

# Load and process PDF files in parallel
def load_pdfs_in_parallel(pdf_directory, max_workers=4):
    documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_text_from_pdf, os.path.join(pdf_directory, filename)): filename
                   for filename in os.listdir(pdf_directory) if filename.endswith(".pdf")}
        
        for future in as_completed(futures):
            filename = futures[future]
            try:
                text = future.result()
                if text:
                    documents.append({"text": text, "source": os.path.join(pdf_directory, filename)})
                else:
                    print(f"Skipped file {filename} due to extraction issues.")
            except Exception as exc:
                print(f"Exception occurred while processing {filename}: {exc}")
    
    return documents

# Load PDFs (parallelized)
documents = load_pdfs_in_parallel(pdf_directory)

# Setup OpenAI Embeddings and text splitter
embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# Split the documents into chunks
def split_documents(documents):
    return [chunk for doc in documents for chunk in splitter.split_text(doc["text"])]

split_texts = split_documents(documents)

# Batch embeddings for faster processing
def embed_in_batches(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embedding.embed_documents(batch))
    return embeddings

embeddings = embed_in_batches(split_texts)

# Initialize Chroma for vector storage
chroma = Chroma(embedding_function=embedding, persist_directory=chroma_db_path)

# Function to split data into smaller batches
def batch_data(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Set the maximum batch size allowed by ChromaDB
MAX_BATCH_SIZE = 5461

# Split texts, embeddings, and metadata into batches
text_batches = list(batch_data(split_texts, MAX_BATCH_SIZE))
embedding_batches = list(batch_data(embeddings, MAX_BATCH_SIZE))
metadata = [{'source': doc['source']} for doc in documents]
metadata_batches = list(batch_data(metadata, MAX_BATCH_SIZE))

# Add the batches to ChromaDB incrementally
for text_batch, embedding_batch, metadata_batch in zip(text_batches, embedding_batches, metadata_batches):
    chroma.add_texts(texts=text_batch, embeddings=embedding_batch, metadatas=metadata_batch)

# Persist the database after all batches have been added
chroma.persist()

# Function to query public citation databases (e.g., arXiv)
def get_public_citations(query, max_results=7):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    citations = []
    for result in search.results():
        citation = f"{result.title} by {result.authors[0].name}, published in {result.published.year}."
        citations.append(citation)
    return citations

#  Retrieval-Augmented Generation (Retrieve + Generate + Cite)
def rag_with_citations(question):
    # Step 1: Retrieve relevant documents from ChromaDB
    results = chroma.similarity_search(question, k=7)
    
    # Step 2: Generate content with a language model (OpenAI GPT-3)
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    retrieved_content = " ".join([result.page_content for result in results])
    
    prompt = f"Using the following context, generate a detailed paragraph:\n\n{retrieved_content}\n\nQuestion: {question}\nAnswer:"
    generated_content = openai_llm(prompt)
    
    # Step 3: Retrieve public citations (e.g., from arXiv)
    citations = get_public_citations(query=question)
    
    # Step 4: Append citations to the generated content
    final_content = f"{generated_content}\n\nCitations:\n"
    for citation in citations:
        final_content += f"- {citation}\n"
    
    return final_content

# Example Usage: Generate a response with citations
""" question = "How can AI impact healthcare?"
response = rag_with_citations(question)
print(response) """