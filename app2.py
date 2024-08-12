#.from_texts().....not saving the file, when user upload any file it is converted to bytes object,,so converting that bytes object to text and perform operations on that text..(text and document has thin line of diff,,see chroma db u come to know).....only single file can be uploaded
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
import io
import pandas as pd
import PyPDF2

load_dotenv()#Loads environmental variables....

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def get_file_text(uploaded_file_which_is_in_bytes_format):

    file_type = uploaded_file_which_is_in_bytes_format.type
    file_name = uploaded_file_which_is_in_bytes_format.name
    file_bytes = uploaded_file_which_is_in_bytes_format.read()

    # Check the file type and process accordingly
    if 'text/csv' in file_type:
        # For CSV files
        file_text = file_bytes.decode('utf-8')
        df = pd.read_csv(io.StringIO(file_text))
        return df

    elif 'text/plain' in file_type:
        # For TXT files
        file_text = file_bytes.decode('utf-8')
        return file_text

    elif 'application/pdf' in file_type:
        # For PDF files
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in file_type:
        # For XLSX files
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df

    else:
        st.error(f"Unsupported file type: {file_type}")
    
    #Only txt and pdf file returning text

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
    #VVVVVVVVVVIIIIIIIIIIIIIIMMMMMMMMMMMPPPPPPPPPPPP
    top_2_most_similar_documents_concatinated = " ".join([docs[0].page_content ,docs[1].page_content]) #concatinating top 2 most similar documents.....
    #VVVVVVVVVVVVVVVVVIIIIIIIIIIIIIIIIMMMMMMMMMMMPPPPPPPPPP
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
        uploaded_file = st.file_uploader("Upload Your PDF files and Click on the Submit & Proceed",type=['csv', 'txt', 'pdf', 'xlsx']) #returns bytes object.............
        
        if st.button("submit & Process"):
            with st.spinner("Processing..."):
                text = get_file_text(uploaded_file)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.success("Done...")


if __name__ == "__main__":
    main()






#1) In app.py i have use FAISS.form_documents() -> saved the file in current working directory,,,and loaded the document using document = loader.load_and_split(),
    # thus we get document,,inside page_content 1 page data is there,,if u have 3 pages then 3 document you get in list,,,then perform chunking on each document to 
    # get several documents using split_documents,,(internally handles page_content),,,then .from_documents(list_of_chunks,embedding_func) to create embedding for 
    # page_content of each document and store that ebedding along woth respective page_content into vector db(if u mentioned remote db api then embeddings for each 
    # chunk is stored there else in local side ,,inside current working directory db is created and store the vector/embedding in sqllite as binary object which cant 
    # be readable,,just we can perform vector algebra operation on it like similarity search..etcand returning the page_content or text of the stored document w.r.t
    #  most similar vector to query vector)

#2) In this file app2.py i will  use FAISS.from_texts() -> not storing files,which are uploaded by user through streamlit app,,,,just converting uploaded file to bytes or binary object 
    # and pass it to get_file_text ,,which will convert those bytes obj in to readable text format like this 'hi my name is shree jagatap , i am data scientist, i work in pune and stay in 
    # belagavi',,,,,,,,,and then this text is converted to chunks like this 
    # ['hi my name is', 'name is shree jagata' , 'shree jagatap i am data scientist', 'data scientist i work in', 'i work in pune and', 'pune and stay in belagavi'],,,,
    # then this is passed to FAISS.from_texts(text_chunks,embedding_function),,thus each text and there respective vectors/embeddings are stored in vector db 
    # like this [('hi my name is',[0.76,-0.3,...]) , ('hi my name is',[-0.21,0.1,...]) , ().....etc],,,,,not exactly like this,,,,,
    # i have just shown you that both,chunk and it's repspective vector/embedding is stored in vector db,,,if you mentioned remote vector db api then ,,there it will be stored,else locally 
    # ie; inside current working dir , what we name mention by that name one folder is created and there in sqllite db vectors are stored in binary obj and chunks are stored in meta_data,,
    # and relationship is established to recognize which vector belongs to which chunk....when user quesry something,then that query is converted to vector and using this stored vector db 
    # performing similarity_search,,,,then this db will return chunk w.r.t most similar vector compare to query vector..if k=1 then only 1 chunk is retured which is most similar compare to 
    # other stored vectors , if k=2 then 2chunks were returned.....
    # these chunks may contain desired answer w.r.t query/user_question,,,,but assume [('hi my name is...............around 10k words',[0.76,-0.3,...]) , ('hi my name is..............around 10k words',[-0.21,0.1,...]) , ().....etc]
    #it is the case then retrieved chunk has around 10k words,,and finding specific answer and in specific format w.r.t query/user_question,ie;we want just 50 word response...in beautiful manner
    #which satsfy user,,what he asked,,,,,,,,,,,,,,,so to get "specific answer" OR "refine" the answer w.r.t asked query or user_question,,we have to pass this chunk got from vector db and query to any LLM
    #then it will give you specific or refined answer in beautifull format,,,thus user will satisfy,,,,,,,,,,,,,and if chunk doesnt contain answer,,bcz query/user question is out of 
    # uploaded document scope then LLM will return "sorry,i cant found answer in given text".................
    #and this connection b/n  "vector db (our use case data) (local or remote)" with any LLM(gen AI models),,,is called RAG(RETRIEVAL AUGMENTED GENERATION)............................

#RETRIEVAL -> retriving the most similar chunk from vector db w.r.t query vector.......not only similarity_search operation,,,,we can retrieve chunk using all kind of vector algebric operation to get most desired answer
#AUGMENTED -> to Refine the retrieved chunk or chunks or to extract specific info w.r.t query,,we use LLM ,,,ie; we are augmenting or modifying the chunk w.r.t query chunk,,by understanding context/meaning of query,,to get specific answer to our query presented in beautiful manner for easy understanding 
#GENERATION -> and we are generating new data based on given data and query,,, which is a generative AI task,,so known as generation task

#see my vector db folder in generative-ai learning,,you come to know what is document and what is text