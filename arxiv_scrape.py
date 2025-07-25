import arxiv
import os
import requests
import time
time.sleep(1)
import re


# Function to sanitize file names
def sanitize_filename(title):
    # Remove or replace any invalid characters (e.g., ':', '/', '\', etc.)
    title = re.sub(r'[\\/*?:"<>|]', "", title)  
    return title.replace(' ', '_').replace(':', '_') 

# Set up the query for AI in healthcare
query = '(all:Healthcare OR all:Medical) AND (all:Artificial intelligence OR all:Machine learning OR all:Deep learning)'

# Create a directory to save the PDFs
if not os.path.exists('Data/ai_healthcare_papers'):
    os.makedirs('Data/ai_healthcare_papers')

# Function to download PDF
def download_pdf(paper):
    try:
        pdf_url = paper.pdf_url
        paper_title = sanitize_filename(paper.title)  # Sanitize the title for the filename
        pdf_filename = f"Data/ai_healthcare_papers/{paper_title}.pdf"
        
        # Download the PDF file
        response = requests.get(pdf_url)
        with open(pdf_filename, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded: {pdf_filename}")
    except Exception as e:
        print(f"Failed to download {paper.title}: {e}")

search = arxiv.Search(
    query=query,
    max_results=100,  
    sort_by=arxiv.SortCriterion.Relevance
)

# Iterate over results and download PDFs
for result in search.results():
    print(f"Processing paper: {result.title}")
    download_pdf(result)
