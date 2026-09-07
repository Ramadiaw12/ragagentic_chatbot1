from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv.ipython import load_dotenv
from IPython.display import Mardown

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage


# Load environnement variable from .env file
load_dotenv(override=True)

# Create a chatgroq instance avec specific model 
llm = ChatGroq(
    model= "openai/gpt-oss-120b",
    temperature=0

)

# Create an agent with a chatgroq instance
agent = create_agent(
    model=llm,
    system_prompt= "you are a helpful assistant"
)

# Invoke the agent from the user messages
resp= agent.invoke(input={"messages":[{"role":"user", "context":"myname is Rahma"}]})
# print the last messages content from the agent response
print(resp["messages"][-1].content)

# Definir la selection de model dynamic, if we have a multiple model different que nous voulons tester or choisir  le model for local or production environnement 
# The model selection is based on the runtime context
@wrap_model_call
def dynamic_select(request=ModelRequest, handler)-> ModelResponse:
    env=request.runtime.context.get("env", "test")
    if env=="env":
        model=llm
    else:
        model=llm

    return handler(request(model=model))

# Create an agent1 with chatgrok and dynamic model selection middleware for testing my functionnality
agent1=create_agent(
    model=llm,
    tool=[],
    debug=True,
    middleware=[dynamic_select]
)
