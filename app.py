# --- Import statements ---
import streamlit as st
from PIL import Image
import io
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
import os
import tempfile
import requests
from bs4 import BeautifulSoup
import camelot
from pdf2image import convert_from_bytes
import pytesseract
import torch
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as RLImage
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from diffusers import StableDiffusionPipeline
from openai import AzureOpenAI
import torch

# --- Load environment variables ---
load_dotenv()
os.environ['CURL_CA_BUNDLE'] = ''

#  --- Azure OpenAI setup
AZURE_API_KEY = os.getenv("AZURE_GPT_KEY")
AZURE_API_VERSION = os.getenv("AZURE_GPT_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

if not (AZURE_API_KEY and AZURE_API_VERSION and AZURE_ENDPOINT):
    st.error("Azure OpenAI credentials not found in .env")
    st.stop()

azure_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)
#  --- LLM configuration ---
llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    temperature=0,
    openai_api_version=AZURE_API_VERSION,
    openai_api_base=AZURE_ENDPOINT,
    openai_api_key=AZURE_API_KEY,
    max_tokens=512
)

# --- Qdrant setup ---
if "qdrant" not in st.session_state:
    st.session_state.qdrant = QdrantClient(":memory:")
qdrant = st.session_state.qdrant

COLLECTION_NAME = "rag_content"
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
    )

# --- PDF helpers ---

# --- Extract text from pdf using PyPDF ---
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text += p + "\n"
    return text.strip()

# --- Extract tables from pdf using camelot
def extract_tables_from_pdf(pdf_file):
    pdf_file.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name
    tables = camelot.read_pdf(tmp_file_path, pages='all')
    table_texts = [f"Table {i+1}:\n{t.df.to_string(index=False)}" for i, t in enumerate(tables)]
    return table_texts

# --- Ectract OCR images in pdf using PyTesseract
def extract_images_text_from_pdf(pdf_file):
    pdf_file.seek(0)
    images = convert_from_bytes(pdf_file.read())
    ocr_texts = []
    img_objects = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            ocr_texts.append(f"OCR page {i+1}:\n{text.strip()}")
        img_objects.append((i, img))
    return ocr_texts, img_objects

# --- Chunking & embedding ---

# --- Punkt is a pre-trained tokenizer model included in NLTK. Using for sentence splitting and word tokenization ---
import nltk
nltk.download("punkt")

# --- Chucking ---
def chunk_text(text, chunk_size=200, overlap=50):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = len(sent)
        else:
            current_chunk.append(sent)
            current_len += len(sent)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---Embedding ---
def get_embedding(text):
    response = azure_client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding

# --- Ingesting in Qdrant ---
def add_text_to_qdrant(text, metadata=None):
    chunks = chunk_text(text)
    points = []
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        payload = {"text": chunk, "chunk_index": i}
        if metadata:
            payload.update(metadata)
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

# --- PDF ingestion ---
def ingest_pdf_multimodal(pdf_file, metadata=None):
    total_chunks = 0
    # Text
    total_chunks += add_text_to_qdrant(extract_text_from_pdf(pdf_file), metadata)
    # Tables
    for t in extract_tables_from_pdf(pdf_file):
        total_chunks += add_text_to_qdrant(t, metadata)
    # Images OCR
    for t in extract_images_text_from_pdf(pdf_file)[0]:
        total_chunks += add_text_to_qdrant(t, metadata)
    return total_chunks

# --- Qdrant query ---
def query_qdrant_multimodal(query_text, top_k=10):
    vec = get_embedding(query_text)
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=top_k)
    return results

# --- Search in RAG pipeline ---
def rag_query_tool(query: str) -> str:
    """
    Search ingested PDFs (Qdrant).
    Returns answer text if found, else empty string.
    """
    results = query_qdrant_multimodal(query)
    if results and any(r.payload.get("text","").strip() for r in results):
        return ai_answer_with_context(query, results)
    return ""

#  --- Use GPT 4o mini to search web ---
def fetch_web_text(query: str) -> str:
    """
    Use GPT knowledge only (no RAG).
    """
    response = azure_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", 
                   "content": (
                    "You are a helpful assistant. "
                    "When the user query cannot be answered from ingested PDFs, "
                    "do NOT apologize or say the context is missing. "
                    "Instead, clearly state: 'Could not find in PDF, here is the answer:' "
                    "and then provide the answer concisely and accurately."
                )},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=512
    )
    return response.choices[0].message.content

#  --- Use GPT 4o mini to summarize web url ---
def summarize_url(url: str) -> str:
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/140.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive"
            }
    try:
        res = requests.get(url, headers=headers, timeout=30, stream=True)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        page_text = "\n".join(p.get_text() for p in paragraphs if p.get_text().strip())
        if not page_text:
            return "Unable to extract text from this URL."

        prompt = f"Summarize the following webpage content clearly and concisely:\n\n{page_text}"
        response = azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Summarize webpages concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to fetch URL: {str(e)}"


def answer_logic(query: str) -> str:
    # Detect URL
    url_match = re.search(r"https?://\S+", query)
    if url_match:
        url = url_match.group(0)
        return summarize_url(url)

    # Check PDFs
    pdf_answer = rag_query_tool(query)
    if pdf_answer.strip():
        return f"{pdf_answer}"
    
    # Search web
    else:
        gpt_answer = fetch_web_text(query)
        return f"{gpt_answer}"

# --- Azure GPT answer with context ---
def ai_answer_with_context(question, context_chunks, model="gpt-4o-mini", max_tokens=512, temperature=0.3):
    context_text = "\n".join([p.payload.get("text","") for p in context_chunks])
    prompt = f"""
    You are ChatGPT with access to retrieved CONTEXT from PDFs or web.
    Use CONTEXT to answer the QUESTION below intelligently.
    If context is insufficient, say so politely.

    CONTEXT:
    {context_text}

    QUESTION: {question}
    """
    response = azure_client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

# --- Image generation using prompt ---
def generate_image_from_prompt(prompt: str):
    if not torch.cuda.is_available():
        st.error("GPU not available for Stable Diffusion!")
        return None
    if "sd_pipe" not in st.session_state:
        with st.spinner("Loading Stable Diffusion model... (one-time setup)"):
            st.session_state.sd_pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                device_map="cuda"
            )
    image = st.session_state.sd_pipe(prompt).images[0]
    return image

# --- Streamlit UI ---
st.set_page_config(page_title="BD Assignment", layout="wide")
st.title("Gen AI Assignment")

if "slides" not in st.session_state:
    st.session_state.slides = []

menu = st.sidebar.radio("Navigation", ["Upload & Ask", "Create Post", "Preview & Download"])

# --- Upload & Ask ---
if menu == "Upload & Ask":
    st.header("Upload PDF-")
    if "pdf_file" not in st.session_state:
        st.session_state.pdf_file = None
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0 

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        if st.session_state.pdf_file != uploaded_pdf:
            st.session_state.pdf_file = uploaded_pdf
            # reset processed chunks so new PDF is indexed
            st.session_state.num_chunks = 0

    # process the PDF only if haven’t done it yet
    if st.session_state.pdf_file and st.session_state.num_chunks == 0:
        st.session_state.num_chunks = ingest_pdf_multimodal(
            st.session_state.pdf_file,
            {"source": "pdf", "filename": getattr(st.session_state.pdf_file, "name", "")}
        )
        st.success(f"Indexed {st.session_state.num_chunks} chunks (text, tables, OCR)")

    st.subheader("Ask a Question")
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.session_state.user_question = st.text_input(
        "Type your question here:",
        value=st.session_state.user_question
        )
    
    if "answer" not in st.session_state:
        st.session_state.answer = ""

    if st.button("Ask") and st.session_state.user_question.strip():
        with st.spinner("Thinking..."):
            st.session_state.answer = answer_logic(st.session_state.user_question)

    # Display answer
    if st.session_state.answer:
        st.subheader("Answer")
        st.write(st.session_state.answer)


elif menu == "Create Post":
    st.header("Create Post Slides")
    num_slides = st.number_input("Number of slides", 1, 10, 1)

    while len(st.session_state.slides) < num_slides:
        st.session_state.slides.append({"image": None, "caption": "", "prompt": ""})

    for i in range(num_slides):
        slide = st.session_state.slides[i]
        st.subheader(f"Slide {i+1}")

        # --- Image input mode ---
        img_mode = st.radio(
            f"Select image input mode for slide {i+1}",
            ["Upload Image", "Generate via Prompt"],
            key=f"img_mode_{i}"
        )

        if img_mode == "Upload Image":
            uploaded_img = st.file_uploader(
                f"Upload image for slide {i+1}",
                type=["jpg", "png"],
                key=f"upload_img_{i}"
            )
            if uploaded_img:
                slide["image"] = Image.open(uploaded_img)

        elif img_mode == "Generate via Prompt":
            slide["prompt"] = st.text_input(
                f"Image prompt for slide {i+1}",
                value=slide.get("prompt", ""),
                key=f"img_prompt_{i}"
            )
            if st.button(f"Generate Image {i+1}"):
                slide["image"] = generate_image_from_prompt(slide["prompt"])
                st.success(f"Generated image for slide {i+1}")

        # --- Caption input mode ---
        cap_mode = st.radio(
            f"Select caption input mode for slide {i+1}",
            ["Manual Typing", "AI Generate", "From RAG"],
            key=f"cap_mode_{i}"
        )

        if cap_mode == "Manual Typing":
            slide["caption"] = st.text_area(
                f"Caption for slide {i+1}",
                value=slide.get("caption", ""),
                key=f"cap_manual_{i}"
            )

        elif cap_mode == "AI Generate":
            cap_prompt = st.text_area(
                f"Enter caption idea for AI (slide {i+1})",
                value=slide.get("caption", ""),
                key=f"cap_ai_prompt_{i}"
            )
            if st.button(f"Generate AI Caption {i+1}"):
                slide["caption"] = ai_answer_with_context(cap_prompt, [])
                st.success(f"Generated caption for slide {i+1}")

        elif cap_mode == "From RAG":
            if "answer" in st.session_state and st.session_state.answer:
                slide["caption"] = st.session_state.answer
                st.info("Using answer from previous page as caption.")
            else:
                st.warning("No RAG answer available from previous page. Please ask a question there first.")

        st.session_state.slides[i] = slide

    st.info("Changes are saved automatically. Go to Preview & Download page to see your slides.")


# --- Preview & Download ---
elif menu == "Preview & Download":
    st.header("Preview & Download")

    if st.session_state.slides:
        if "current_slide" not in st.session_state:
            st.session_state.current_slide = 0

        total_slides = len(st.session_state.slides)
        current = st.session_state.current_slide
        slide = st.session_state.slides[current]

        # --- Centered Preview ---
        st.subheader(f"Slide {current+1} of {total_slides}")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if slide["image"]:
                img_buf = io.BytesIO()
                slide["image"].resize((1080, 566)).save(img_buf, format="PNG")
                img_buf.seek(0)
                st.image(img_buf, width=400)

            if slide.get("caption"):
                st.markdown(
                    f"<div style='text-align:center; font-size:18px; line-height:1.4;'>{slide['caption']}</div>",
                    unsafe_allow_html=True
                )

        # --- Navigation buttons ---
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            if st.button("Previous", disabled=current == 0):
                st.session_state.current_slide -= 1
                st.rerun()
        with col3:
            if st.button("Next", disabled=current == total_slides - 1):
                st.session_state.current_slide += 1
                st.rerun()

        st.markdown("---")

        # --- Slide Rearrangement ---
        st.subheader("Rearrange Slides")
        new_order = st.multiselect(
            "Drag to reorder slides:",
            options=list(range(total_slides)),
            default=list(range(total_slides)),
            format_func=lambda x: f"Slide {x+1}"
        )

        if st.button("Apply New Order"):
            st.session_state.slides = [st.session_state.slides[i] for i in new_order]
            st.success("Slides reordered successfully!")
            st.rerun()

        # --- Download Options ---
        st.subheader("Download Options")

        # Centered caption style for PDF
        styles = getSampleStyleSheet()
        caption_style = ParagraphStyle(
            'CenteredCaption',
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=12,
            spaceAfter=20
        )

        # Single Slide as PNG
        slide_idx = st.selectbox(
            "Select slide to download as image",
            options=list(range(len(st.session_state.slides))),
            format_func=lambda x: f"Slide {x+1}"
        )
        if st.button("Download Selected Slide as PNG"):
            sel_slide = st.session_state.slides[slide_idx]
            if sel_slide["image"]:
                img_buf = io.BytesIO()
                sel_slide["image"].resize((1080, 566)).save(img_buf, format="PNG")
                img_buf.seek(0)
                st.download_button(
                    label=f"📸 Download Slide {slide_idx+1} (PNG)",
                    data=img_buf,
                    file_name=f"slide_{slide_idx+1}.png",
                    mime="image/png"
                )
            else:
                st.warning("Selected slide has no image!")

        # All Slides as PDF
        if st.button("Download All Slides as PDF"):
            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
            elements = []

            for slide in st.session_state.slides:
                if slide["image"]:
                    img_buf = io.BytesIO()
                    slide["image"].resize((1080, 566)).save(img_buf, format="PNG")
                    img_buf.seek(0)
                    elements.append(RLImage(img_buf, width=4*inch, height=4*inch))
                    elements.append(Spacer(1, 0.3*inch))

                if slide.get("caption", ""):
                    para = Paragraph(slide["caption"], caption_style)
                    elements.append(para)
                    elements.append(Spacer(1, 1*inch))

            doc.build(elements)
            pdf_buf.seek(0)

            st.download_button(
                label="Download PDF",
                data=pdf_buf,
                file_name="slides.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("No slides created yet!")
