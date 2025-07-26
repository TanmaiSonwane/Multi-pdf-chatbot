import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Generation, LLMResult
from huggingface_hub import InferenceClient
from typing import Optional, List
from langchain.llms.base import LLM
from htmlTemplates import css, bot_template, user_template
import os

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

# ---------- STEP 4: Custom LLM wrapper over HF Inference API ----------
hf_client = InferenceClient(
    model="google/flan-t5-small",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
class CloudLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = f"Answer the following question based on the documents:\n{prompt}"
        response = hf_client.text_generation(prompt=prompt, max_new_tokens=256, temperature=0.7)
        return response.strip()

    @property
    def _llm_type(self) -> str:
        return "custom-huggingface-llm"

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = [Generation(text=self._call(prompt)) for prompt in prompts]
        return LLMResult(generations=[generations])

# ---------- STEP 5: Conversation Chain ----------
def get_conversation_chain(vectorstore):
    llm = CloudLLM()
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

    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
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
