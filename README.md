This project is a Streamlit-based Gen AI application that enables users to:

- Upload PDFs and extract text, tables, and OCR data.
- Ask questions using a RAG (Retrieval-Augmented Generation) pipeline with Qdrant + Azure OpenAI.
- Fallback to web/GPT search when no answer is found in PDFs.
- Generate presentation slides with images (uploaded or AI-generated via Stable Diffusion) and captions (manual, AI-generated, or RAG-derived).
- Preview & Rearrange slides.
- Download slides as PNG or PDF.

## 🚀 Features
- Upload PDF file
- Extract:
  - Text via **PyPDF2**
  - Tables via **Camelot**
  - OCR text via **pytesseract**
- Convert PDF pages to images using **pdf2image**
- Generate embeddings with **text-embedding-3-large**
- Ask questions with **Azure gpt-4o-mini**
- Multimodal support (Text + Tables + OCR)

## ⚙️ Tech Stack
- **Streamlit**: UI framework  
- **PyPDF2**: PDF text extraction  
- **Camelot**: Table extraction  
- **pdf2image**: Convert PDF to images  
- **pytesseract**: OCR engine   
- **Qdrant(in memory)**: Vector database
- **Stable Diffusion**: AI image creation
- **Azure OpenAI(gpt-4o-mini, text-embedding-3-large)**: LLM & Embeddings
- **ReportLab** Document Export
- **BeautifulSoup** (web scraping), **NLTK** (sentence tokenization)
  
## 📂 Project Structure
```plaintext
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # Documentation
```

## ⚙️ Setup & Installation
1. Clone the repository
2. ```bash
   git clone https://github.com/your-username/genai-assignment.git
   cd genai-assignment
   ```
   
3. Create virtual environment & install dependencies
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   pip install -r requirements.txt
   
4. Configure Environment Variables
   Create a .env file in the project root (sample is given in .env.example)

5. Run the Streamlit App
   streamlit run app.py



