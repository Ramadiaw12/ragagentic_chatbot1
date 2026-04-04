from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain.tool import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_care.tools import creat_retriever_tool
from langchain_community.vectorstores import Chroma

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

load_dotenv(override=True)


llm = ChatOpenAI(model="gpt-4o", temperature=0)

texts = [
    "Je suis une futur ai engineer , now I work in CIH bank"


]

embedding_model = OpenAIEmbeddings(model = "test-embedding-ada-002") 
vectorstore = Chroma.from_texts(texts, embedding_model,collection_name="Agentic_AI")

retrieval = vectorstore.as_retriever(kwargs={"k":5})
retrieval_tool = create_retriever_tool(retriever=retrieval, name="kb_search", description="Search information about me")

@tool
def send