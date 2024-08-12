#.from_documents().....saving the uploaded file and loading as document.....multiple's can be uploaded and stored
#track the user count,,and create separate folder for each user,(user_i) and inside that store files uploaded by user and you can delete after let say 30 miute
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
from langchain.document_loaders import DirectoryLoader,TextLoader
import PyPDF2
import pandas as pd
import csv

load_dotenv()#Loads environmental variables....

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

uid = None

def get_file_texts(saved_user_folder_path):

    def pdf_loader(file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text

    def csv_loader(file_path):
        st.write(pd.read_csv(file_path))
        with open(file_path, mode='r', newline='') as file:
         # Create a CSV reader object
            reader = csv.reader(file)
            text=[]
            # Iterate over each row in the CSV file
            for row in reader:
                # Join the row's elements into a single string with commas separating the values
                line = ', '.join(row)
                # Print the line
                text.append(line)
            text=" ".join(text)

        return text
                

    def text_loader(file_path):
        return TextLoader(file_path)
    

    def custom_loader(file_path):
        if file_path.endswith('.txt'):
            text_loader = DirectoryLoader(saved_user_folder_path, loader_cls=text_loader)
            return text_loader(file_path)
        elif file_path.endswith('.pdf'):
            return pdf_loader(file_path)
        elif file_path.endswith('.csv'):
            return csv_loader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    list_of_text = []
    for filename in os.listdir(saved_user_folder_path):
        file_path = os.path.join(saved_user_folder_path, filename)
        list_of_text.append(custom_loader(file_path))
    list_of_text = " ".join(list_of_text) #if more than 1 file is there,,then concatinate them,,bcz text_splitter.split_text(text),,requires,,just text like 'my name is ...' ,,not list of text like ['my name is..', 'goin to ride...']

    return list_of_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000 , chunk_overlap=1000) #Increasing the chunk_size and chunk_overlap,,inceases the RAG/Gen AI mode or systems performance or accuracy
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def get_vector_store(text_chunks):
    #we can use any embedding model,,,not neccessory that if we are using gemini model means we have to use googles embedding model only,,,,,,,,you can use openai's embedding , huggingface embedding , ...etc some are paid and some are freee.....but google offering free embedding models like(text-embedding-004 , embedding-001 ...which are free)...word2vec is also googles embedding model,,but you have to do tokenization and aggregation by manually,,so avoid it,,,,,,,,use these latest embedding models where you have to give texta and it will give toy vector/embedding with 768 dim (w.r.t googles embedding mode) vector representation for text
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks , embeddings)
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
    top_2_most_similar_documents_concatinated=[]
    for i in range(len(docs)):
        top_2_most_similar_documents_concatinated.append(docs[i].page_content)
    
    top_2_most_similar_documents_concatinated = " ".join(top_2_most_similar_documents_concatinated)
    
    # top_2_most_similar_documents_concatinated = " ".join([docs[0].page_content ,docs[1].page_content]) #concatinating top 2 most similar documents.....



    prompt = make_rag_prompt(user_question, top_2_most_similar_documents_concatinated)

    
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    
    st.write(answer.text)
   

def save_files(uploaded_files):
    # Define the folder to save uploaded files
    UPLOAD_FOLDER = 'uploaded_files'
    USER_FOLDER = "User_{i}"
    
    

    #count user and set User_i.........user_count file initially contains '1'
    with open('user_count','r+') as f: #read and write mode,,when file is closed then only it is saved..so dont perform 2 write operation otherwise,,recent write overwrite previous one
        global uid
        count = int(f.read())
        uid=count
        count+=1
        f.seek(0)   #to truncate the beginning value or first value and add new data using f.write(),,as below 
        f.write(str(count))
       
    #Create folder for user_i
    saved_path = os.path.join(UPLOAD_FOLDER,USER_FOLDER.format(i=uid))

    # Create the folder if it doesn't exist
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)


    for uploaded_file in uploaded_files:
        # Save the uploaded file to the specified folder
        file_saving_path=os.path.join(saved_path , uploaded_file.name)
        with open(file_saving_path, 'wb') as f:
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
        uploaded_files = st.file_uploader("Upload Your PDF files and Click on the Submit & Proceed",accept_multiple_files=True) #returns bytes object.............
        if uploaded_files is not None:
            saved_user_folder_path = save_files(uploaded_files)
        st.write(saved_user_folder_path)
        if st.button("submit & Process"):
            with st.spinner("Processing..."):
                text = get_file_texts(saved_user_folder_path)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success("Done...")


if __name__ == "__main__":
    main()