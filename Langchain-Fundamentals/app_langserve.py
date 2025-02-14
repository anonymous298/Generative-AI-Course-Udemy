import uvicorn

from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langserve import add_routes

app = FastAPI(
    title='Langserve API',
    version='0.1',
    description='An API for generating responses using Langserve'
)

llm = ChatOllama(model='llama3.2:3b')

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are an helpful assistant'),
        ('user', 'Question: {question}')
    ]
)

chain = prompt | llm | StrOutputParser()

add_routes(
    app,
    chain,
    path='/chain'
)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)