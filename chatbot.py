import streamlit as st
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDFs
import docx
import pandas as pd
import pytesseract
from PIL import Image
import cv2
import tempfile  # Handle temporary files

# 🔹 Set Your Gemini API Key
GEMINI_API_KEY = "AIzaSyDDmMm2gDaK6Syy4ZmbEzbsDbHMzhS5dgk"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index
DIMENSION = 384
index = faiss.IndexFlatL2(DIMENSION)

st.title("📄 RAG Chatbot with Gemini & FAISS")

# ✅ File Upload Section
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT, CSV, JPG, MP4)", 
    type=["pdf", "docx", "txt", "csv", "xlsx", "jpg", "jpeg", "png", "mp4"], 
    accept_multiple_files=True
)

# ✅ Function to extract text from different files **without saving**
def process_file(file):
    ext = file.name.split(".")[-1].lower()

    try:
        if ext == "pdf":
            return process_pdf(file)
        elif ext == "docx":
            return process_docx(file)
        elif ext == "txt":
            return process_text(file)
        elif ext == "csv":
            return process_csv(file)
        elif ext == "xlsx":
            return process_excel(file)
        elif ext in ("jpg", "jpeg", "png"):
            return process_image(file)
        elif ext == "mp4":
            return process_video(file)
        else:
            return f" Unsupported file type: {ext}"
    except Exception as e:
        return f" Error processing {file.name}: {str(e)}"

# ✅ File processing functions
def process_pdf(file):
    text = ""
    with fitz.open("pdf", file.read()) as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()

def process_docx(file):
    doc = docx.Document(file)
    return "\n".join(para.text for para in doc.paragraphs).strip()

def process_text(file):
    return file.getvalue().decode("utf-8").strip()

def process_csv(file):
    df = pd.read_csv(file)
    return df.to_string().strip()

def process_excel(file):
    df = pd.read_excel(file)
    return df.to_string().strip()

def process_image(file):
    try:
        return pytesseract.image_to_string(Image.open(file)).strip()
    except Exception as e:
        return f"⚠️ Error extracting text from image: {str(e)}"

#  Optimized Video Processing
def process_video(file):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(file.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        text = ""
        frame_count = 0

        if not cap.isOpened():
            return " Error: Could not open the video file."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 50:  # Stop after 50 frames
                break

            if frame_count % 10 == 0:  # Process every 10th frame
                # Convert frame to grayscale for better OCR
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply OCR (text extraction)
                extracted_text = pytesseract.image_to_string(gray_frame)
                text += extracted_text + "\n"

            frame_count += 1

        cap.release()
        return text.strip() if text.strip() else " No readable text found in the video."

    except Exception as e:
        return f" Error processing video: {str(e)}"

# ✅ Store embeddings in FAISS only if there are valid texts
all_text = ""
chunks = []

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Processing {file.name}... ")
        content = process_file(file)

        if content.strip():
            st.text_area(f"📜 Extracted Text from {file.name}", content[:1000])  # Preview
            all_text += content + "\n"

    # Ensure text is processed before adding to FAISS
    if all_text.strip():
        def get_embedding(text):
            return embedding_model.encode(text, convert_to_numpy=True)

        chunks = all_text.split(". ")  # Simple sentence-based chunking
        embeddings = np.array([get_embedding(chunk) for chunk in chunks if chunk.strip()])

        if embeddings.size > 0:
            index.add(embeddings)
            st.success("✅ Files processed & stored in FAISS!")
        else:
            st.warning("⚠️ No valid text found to store in FAISS.")

# ✅ Q&A Section
user_query = st.text_input("🔍 Ask a question:")
if user_query.strip():
    query_vector = embedding_model.encode(user_query).reshape(1, -1)

    if index.ntotal > 0:  # Ensure FAISS has data
        distances, indices = index.search(query_vector, 3)
        retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        context = "\n".join(retrieved_chunks)

        try:
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            response = model.generate_content(f"Context: {context}\n\nQuestion: {user_query}")
            answer = response.text.strip()
        except Exception as e:
            answer = f"⚠️ Error querying Gemini: {str(e)}"

        st.subheader("💡 Answer:")
        st.write(answer)
    else:
        st.warning("⚠️ No processed data available. Please upload documents first.")
