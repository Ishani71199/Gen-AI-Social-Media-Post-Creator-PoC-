# Gen AI Social Media style post creation PoC
This project is a Streamlit-based Gen AI application that enables users to:

- Upload PDFs and extract text, tables, and OCR data.
- Ask questions using a RAG (Retrieval-Augmented Generation) pipeline with Qdrant + Azure OpenAI.
- Fallback to web/GPT search when no answer is found in PDFs.
- Generate presentation slides with images (uploaded or AI-generated via Stable Diffusion) and captions (manual, AI-generated, or RAG-derived).
- Preview & Rearrange slides.
- Download slides as PNG or PDF.

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

## 🖼️ App Walkthrough

### Upload & Ask  
- **Upload a PDF**  
- The app automatically extracts **text, tables, and OCR**  
- Ask a question → **RAG retrieves context → GPT answers**  
- If context not found → **GPT fallback**

### Create Post  
- **Decide number of slides**  
- For each slide:  
  - Upload image / Generate via **AI**  
  - Add caption: **Manually / AI / RAG**  
- Slides are stored in **session state**

### Preview & Download  
- Navigate slides one by one  
- Reorder slides
- Export options:    
  - Individual slide → **PNG**  
  - All slides → **PDF** 

## ⚙️ Setup & Installation
1. Clone the repository
 ```bash
   git clone https://github.com/your-username/genai-assignment.git
   cd genai-assignment
   ```
   
3. Create virtual environment & install dependencies
```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   pip install -r requirements.txt
   ```
   
5. Configure Environment Variables <br>
   Create a .env file in the project root (sample is given in .env.example)

6. Run the Streamlit App
 ```bash
   streamlit run app.py
   ```

## 🤝 Contributing
Pull requests are welcome! Please fork this repo and create a PR for review.

## 👩‍💻 Author
### Ishani Biswas 
🔗 **Connect with me**   
- 💼 [LinkedIn](https://www.linkedin.com/in/ishanibiswas/)  
- 📧 [Email](mailto:biswasishani71199@gmail.com) 


