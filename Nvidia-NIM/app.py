import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

# Loading Nvidia NIM Api Key
os.environ['NVIDIA_NIM_API_KEY'] = os.getenv('NVIDIA_NIM_API_KEY')

class InitializeSessions:
    @staticmethod
    def InitializeVectorStore(url):
        try:
            if 'vectorstore' not in st.session_state:
                # Loading Youtube Video
                st.session_state.loader = YoutubeLoader.from_youtube_url(youtube_url=url)
                st.session_state.docs = st.session_state.loader.load()

                # Chunking the docs
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

                # Creating vector store
                st.session_state.embedding = OllamaEmbeddings(model='mxbai-embed-large:335m')
                st.session_state.vectorstore = Chroma.from_documents(documents=st.session_state.documents, embedding=st.session_state.embedding)
        
        except Exception as e:
            st.error(e)
            
    @staticmethod
    def InitializeChain():
        try:
            if 'memory' not in st.session_state:
                # Creating memory
                st.session_state.memory = []

            if 'retrievalchain' not in st.session_state:
                # Creating LLM from Nvidia NIM
                st.session_state.llm = ChatNVIDIA(
                    model="meta/llama-3.3-70b-instruct",
                    api_key=os.getenv('NVIDIA_NIM_API_KEY'),
                    temperature=1,
                    top_p=0.7,
                    max_tokens=1024,
                )

                # Creating a prompt template
                st.session_state.template = """
                    I am providing you the context. Reply to the user according to the given context:

                    Context: {context}

                    Question: {input}
                """

                st.session_state.prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful RAG Assistant. I will provide you a context and you must respond according to it."),
                        MessagesPlaceholder(variable_name='memory'),
                        ("user", f'{st.session_state.template}')
                    ]
                )

                st.session_state.retriever = st.session_state.vectorstore.as_retriever()

                st.session_state.document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.prompt)

                st.session_state.retrievalchain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)

        except Exception as e:
            st.error(e)


st.title('RAG Application Using Nvidia NIM')
st.write("Enter a YouTube Video URL to chat with...")

url = st.text_input('Enter Here')

if st.button('Start Chatting...'):
    # Initializing all sessions
    InitializeSessions.InitializeVectorStore(url)
    InitializeSessions.InitializeChain()

# Chat Interface
st.write("### Chat with YouTube Video")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Taking User Query
user_query = st.chat_input('Ask something related to the YouTube video...')

if user_query:
    # Append user message to memory and chat history
    st.session_state.memory.append(HumanMessage(content=user_query))
    st.session_state.messages.append(HumanMessage(content=user_query))

    # Generate Response
    response = st.session_state.retrievalchain.invoke({'input': user_query, 'memory': st.session_state.memory})

    # Append AI response to memory and chat history
    st.session_state.memory.append(AIMessage(content=response['output']))
    st.session_state.messages.append(AIMessage(content=response['output']))

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response)
