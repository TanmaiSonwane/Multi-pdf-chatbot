import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Generation, LLMResult
from langchain.llms.base import LLM
from htmlTemplates import css, bot_template, user_template
from typing import Optional, List
import google.generativeai as genai
import os
st.set_page_config(page_title="Chat with PDFs", page_icon="📄", layout="wide")
# ---------- STEP 1: Extract PDF text ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# ---------- STEP 2: Chunk text ----------
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1500, chunk_overlap=100, length_function=len
    )
    return text_splitter.split_text(text)

# ---------- STEP 3: Vector Store ----------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="thenlper/gte-small")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# ---------- STEP 4: Custom Gemini LLM wrapper ----------
genai.configure(api_key="AIzaSyAOUIV9e-hP4yrRPT5N9UgL0fI_f3LuaOs")
gemini_model = gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

class GeminiLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = f"Answer the following question based on the documents:\n{prompt}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self) -> str:
        return "custom-gemini-llm"

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = [Generation(text=self._call(prompt)) for prompt in prompts]
        return LLMResult(generations=[generations])

# ---------- STEP 5: Conversation Chain ----------
def get_conversation_chain(vectorstore):
    llm = GeminiLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# ---------- STEP 6: Handle user query ----------
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ---------- STEP 7: Streamlit UI ----------
def main():
    load_dotenv()

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📄")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
