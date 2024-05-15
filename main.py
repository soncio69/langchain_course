## https://www.diariodiunanalista.it/posts/chatbot-python-langchain-rag/

import pandas as pd

def load_dataset(dataset_name:str="dataset.csv"):
    """
    Funzione helper per caricare il dataset

    Args:
        dataset_name (str, optional): Nome del file salvato dalla fase di estrazione. Defaults to "dataset.csv".

    Returns:
        pd.DataFrame: DataFrame Pandas dei dati raccolti da LangChain
    """
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    return df

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_chunks(dataset:pd.DataFrame, chunk_size:int, chunk_overlap:int):
    """
    Crea chunk informazionali dal dataset 

    Args:
        dataset (pd.DataFrame): Dataset Pandas
        chunk_size (int): Quanti chunk informazionali?
        chunk_overlap (int): Quanti chunk condivisi?

    Returns:
        list: lista di chunk
    """
    text_chunks = DataFrameLoader(
        dataset, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )
    # aggiungiamo i metadati ai chunk stessi per facilitare il lavoro di recupero
    for doc in text_chunks:
        title = doc.metadata["title"]
        description = doc.metadata["description"]
        content = doc.page_content
        url = doc.metadata["url"]
        final_content = f"TITLE: {title}\DESCRIPTION: {description}\BODY: {content}\nURL: {url}"
        doc.page_content = final_content

    return text_chunks


from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Funzione per creare o caricare il database vettoriale dalla memoria locale

    Returns:
        FAISS: Vector store
    """
    embeddings = OpenAIEmbeddings() # possiamo cambiarla a piacimento!
    # embeddings = HuggingFaceInstructEmbeddings() # ad esempio rimuovendo il commento qui e commentando la riga di sopra

    if not os.path.exists("./db"):
        print("CREATING DB")
        vectorstore = FAISS.from_documents(
            chunks, embeddings
        )
        vectorstore.save_local("./db")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)

    return vectorstore


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def get_conversation_chain(vector_store: FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Oggetto LangChain che permette domanda-risposta tra umano e LLM

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """
    llm = ChatOpenAI(model="gpt-4") # possiamo cambiare modello a piacimento
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain