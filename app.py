#.from_documents().....saving the uploaded file and loading as document.....only single file can be uploaded
import streamlit as st 
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI #Not used in this app
from langchain.chains.question_answering import load_qa_chain #Not used in this app
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA #not used,,,,,,i think RetrievalQA is just for OpenAI.....bcz not working with gemini

load_dotenv()#Loads environmental variables....

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def get_pdf_document(path_of_file):
    loader = PyPDFLoader(path_of_file)
    document = loader.load_and_split()
    return document

def get_document_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000 , chunk_overlap=1000) #Increasing the chunk_size and chunk_overlap,,inceases the RAG/Gen AI mode or systems performance or accuracy
    document_chunks = text_splitter.split_documents(document)
    return document_chunks


def get_vector_store(document_chunks):
    #we can use any embedding model,,,not neccessory that if we are using gemini model means we have to use googles embedding model only,,,,,,,,you can use openai's embedding , huggingface embedding , ...etc some are paid and some are freee.....but google offering free embedding models like(text-embedding-004 , embedding-001 ...which are free)...word2vec is also googles embedding model,,but you have to do tokenization and aggregation by manually,,so avoid it,,,,,,,,use these latest embedding models where you have to give texta and it will give toy vector/embedding with 768 dim (w.r.t googles embedding mode) vector representation for text
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks , embeddings)
    # return vector_store.index.ntotal -> no of vectors formed
    vector_store.save_local("faiss_index") #Persistiting changes in folder name "faiss_index"..


def make_rag_prompt(query, relevant_passage):
  
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below.
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
  strike a friendly and converstional tone.
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

  ANSWER:
  """).format(query=query, relevant_passage=relevant_passage)

  return prompt


def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    my_vector_store = FAISS.load_local("faiss_index",embeddings , allow_dangerous_deserialization = True)

    # docs = my_vector_store.similarity_search(user_question) OR
    retriever = my_vector_store.as_retriever(search_kwargs={"k":2})
    docs = retriever.invoke(user_question)
    top_2_most_similar_documents_concatinated = " ".join([docs[0].page_content ,docs[1].page_content]) #concatinating top 2 most similar documents.....



    prompt = make_rag_prompt(user_question, top_2_most_similar_documents_concatinated)

    
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    
    st.write(answer.text)
   

def save_file(uploaded_file):
    # Define the folder to save uploaded files
    UPLOAD_FOLDER = 'uploaded_files'

    # Create the folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Save the uploaded file to the specified folder
    saved_path=os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(saved_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return saved_path



def main():
    st.set_page_config("CHAT WITH MULTIPLE PDF")
    st.header("Chat with Multiple PDF's using Gemini")

    #When user enetr or add some file,,then below 3 lines run,,and call to user_input()
    user_question = st.text_input("Ask the question from pdf files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Upload Your PDF files and Click on the Submit & Proceed") #returns bytes object.............
        if uploaded_file is not None:
            saved_path = save_file(uploaded_file)
        if st.button("submit & Process"):
            with st.spinner("Processing..."):
                document = get_pdf_document(saved_path)
                document_chunks = get_document_chunks(document)
                get_vector_store(document_chunks)
                st.success("Done...")


if __name__ == "__main__":
    main()