import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Generation, LLMResult, Document
from langchain.llms.base import LLM
from typing import Optional, List
from htmlTemplates import css, bot_template, user_template
import google.generativeai as genai
import pytesseract
from PIL import Image
import pdf2image
import os

st.set_page_config(page_title="PDF Chatbot")

# Clear Chat Button
col1, col2 = st.columns([8, 2])
with col2:
    if st.button("\ud83d\udd04 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Configure Gemini
genai.configure(api_key="AIzaSyAOUIV9e-hP4yrRPT5N9UgL0fI_f3LuaOs")
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

class GeminiLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = format_prompt(prompt)
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self) -> str:
        return "custom-gemini-llm"

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = [Generation(text=self._call(prompt)) for prompt in prompts]
        return LLMResult(generations=[generations])

# ---------- STEP 1: Extract PDF text ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if not page_text:
                images = pdf2image.convert_from_bytes(pdf.read())
                for image in images:
                    text += pytesseract.image_to_string(image)
            else:
                text += page_text
    return text

# ---------- STEP 2: Chunk text ----------
def get_text_chunks_with_metadata(pdf_docs):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=100, length_function=len
    )
    documents = []
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                chunks = splitter.split_text(page_text)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata={"page_number": page_num + 1}))
    return documents

# ---------- STEP 3: Vector Store ----------
def get_vectorstore(documents):
    embeddings = HuggingFaceInstructEmbeddings(model_name="thenlper/gte-small")
    return FAISS.from_documents(documents, embedding=embeddings)

# ---------- STEP 4: Conversation Chain ----------
def get_conversation_chain(vectorstore):
    llm = GeminiLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory, return_source_documents=True)

# ---------- STEP 5: prompt format ----------
def format_prompt(user_question):
    return f"""
You are an intelligent assistant helping users understand documents they uploaded.

Instructions:
- Be accurate and concise.
- Cite source content when appropriate.
- Answer in markdown format.

Question:
{user_question}
"""

# ---------- STEP 6: Handle user query ----------
def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({"question": user_question})
        chat_history = response['chat_history']
        rag_answer = chat_history[-1].content
        source_docs = response.get("source_documents", [])

        fallback_triggers = ["i don't know", "cannot find", "no relevant", "not contain", "unrelated", "i'm not sure"]
        is_fallback_needed = any(trigger in rag_answer.lower() for trigger in fallback_triggers) or len(rag_answer.strip()) < 10

        if is_fallback_needed:
            st.info("Using Gemini general knowledge...")
            gemini_response = gemini_model.generate_content(user_question)
            answer = gemini_response.text.strip()
            page_note = "\n\n_This answer is based on general knowledge._"
        else:
            answer = rag_answer
            page_note = ""
            if source_docs:
                page_nums = sorted(set(doc.metadata.get("page_number") for doc in source_docs if doc.metadata.get("page_number")))
                if page_nums:
                    page_note = f"\n\n_Referenced from page(s): {', '.join(map(str, page_nums))}_"

        final_response = answer + page_note

        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": final_response})

        for msg in st.session_state.chat_history:
            template = user_template if msg["role"] == "user" else bot_template
            st.write(template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"\u26a0\ufe0f Error answering question: {e}")

# ---------- STEP 7: Summarize PDF ----------
def summarize_pdf(text):
    summary_prompt = "Summarize this document:\n" + text[:4000]
    try:
        response = gemini_model.generate_content(summary_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# ---------- STEP 8: Main Streamlit UI ----------
def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs \ud83d\udcc4")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        try:
            handle_userinput(user_question)
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    documents = get_text_chunks_with_metadata(pdf_docs)
                    vectorstore = get_vectorstore(documents)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.subheader("Document Summary")
                    summary = summarize_pdf(raw_text)
                    st.success(summary)
                except Exception as e:
                    st.error(f"Failed to process documents: {str(e)}")

if __name__ == '__main__':
    main()

