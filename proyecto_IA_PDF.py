# proyecto_IA_PDF
# Construir un modelo GPT basado en un documento PDF

# Ejecutar desde el programa terminal o símbolo de sistema en la carpeta donde se guarde este archivo:
# streamlit run proyecto_IA_PDF.py
# La primera vez que se ejecuta pregunta el email, simplemente dejarlo en blanco.

# Cargamos las librerías necesarias para construir la interfaz web (streamlit) y el resto de componentes utilizados
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Diseño de la interfaz web
st.set_page_config('Proyecto IA PDF')
st.header("Construir un modelo GPT basado en un documento PDF")
OPENAI_API_KEY = st.text_input('Usamos el motor de ChatGPT OpenAI, introduce una API Key válida (con saldo)', type='password')
pdf_obj = st.file_uploader("Carga el documento PDF", type="pdf", on_change=st.cache_resource.clear)

# Se trata la caché y se procede a crear embeddings al cargar el documento PDF
@st.cache_resource 
def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    texto = ""
    for page in pdf_reader.pages:
        texto += page.extract_text()
    # Dividimos el texto en chunks de 800 caracteres solapados 100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function=len)        
    chunks = text_splitter.split_text(texto)
    # Usamos el modelo MiniLM para hacer los embeddings y no ocupar mucho espacio en disco: 471MB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Creamos la base de conocimiento con los chunks y embeddings
    base_conocimiento = FAISS.from_texts(chunks, embeddings)
    return base_conocimiento

if pdf_obj:
    base_conocimiento = create_embeddings(pdf_obj)
    pregunta = st.text_input("Haz una pregunta sobre el documento PDF:")
    if pregunta:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # Buscamos en la base de conocimiento (chunks+embeddings) 3 fragmentos que puedan contener la respuesta.
        # Se podrían trasladar más fragmentos a ChatGPT pero encareceríamos el coste de las preguntas.
        # Así, aproximadamente, cada pregunta cuesta 0,001€ 
        docs = base_conocimiento.similarity_search(pregunta, 3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question='Según el texto, '+pregunta)
        st.write(respuesta)

