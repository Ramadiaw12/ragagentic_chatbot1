from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv.ipython import load_dotenv
from IPython.display import Markdown

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call, wrap_tool_call, dynamic_prompt, ModelRquest
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
# from langgraph.checkpoint.prostgres import PostgresSaver
from langchain.tools import tool
from ddgs import DDGS
from langchain.messages import ToolMessage
from typing import TypedDict
from pydantic import BaseModel
from langchain.agents.structured_output import ToolStrategy



# Load environnement variable from .env file
load_dotenv(override=True)

# Create a chatgroq instance avec specific model 
advanced_llm = ChatGroq(
    model= "openai/gpt-oss-120b",
    temperature=0

)
basic_llm= ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)
# Create an agent with a chatgroq instance
agent = create_agent(
    model=advanced_llm,
    system_prompt= "you are a helpful assistant"
)

# Invoke the agent from the user messages
resp= agent.invoke(input={"messages":[{"role":"user", "context":"myname is Rahma"}]})
# print the last messages content from the agent response
print(resp["messages"][-1].content)

# Definir la selection de model dynamic, if we have a multiple model different que nous voulons tester or choisir  le model for local or production environnement 
# The model selection is based on the runtime context
@wrap_model_call
def dynamic_select(request:ModelRequest, handler)-> ModelResponse:
    env=request.runtime.context.get("env", "test")
    if env=="env":
        model=basic_llm
    else:
        model=advanced_llm

    return handler(request(model=model))

# Create an agent1 with chatgrok and dynamic model selection middleware for testing my functionnality
agent1=create_agent(
    model=advanced_llm,
    tool=[],
    debug=True,
    middleware=[dynamic_select]
)
# Create an object context for the agent1 to test the model selected by my middleware function
resp=agent1.invoke(input=
    {
    "messages":[
    {
        "role":"user", 
        "context":"C'est quoi un agent loop"
     }]
    },

    context={"env":"test"}

    )
# Print the messges for  the agent1 response
print(resp["response"][-1].content)

# creata an agent with a checkpointer to save the conversation in memory
memory=InMemorySaver()
agent3=create_agent(
    model=advanced_llm,
    sytem_prompt="you are a helpful assistant",
    checkpointer=memory
)

# Invoke agent3 with a specific thread_id for save the conversation in memory
config={"configurable":{"thread_id":1}}
resp3=agent3.invoke(
    input={"messages": {"HumanMessage": "mon nom est rahma"}},
    config=config
    )

print(resp3["response"][-1].content)

resp4=agent3.invoke(
    input={"messages": {"HumanMessage": "C'est quoi mon nom"}},
    config=config
    )

print(resp4["response"][-1].content)


# CREATION DES TOOLS

# Create a tool to get get information about the city what I live
@tool
def get_info(city : str):
    """
    Get information about a city
    """
    print("Info tool invoke")
    return {
        "city": city,
        "temperature": "25 C",
        "humidity": "50%"
    }

# Create a tool for to get the information about an employer salary
@tool
def get_employes_infor(employer_name:str):
    """
    Obtenir des information à propos des employés de l'usine de production 
    """
    print("get_employes_infor Tool invoke")
    return {
        "name_employes": employer_name,
        "Salary":58000,
        "seniority":8
    }

# Rattachement des tools à un agent
agent4=create_agent(
    model="openai/gpt-oss-120b",
    tool=[get_info, get_employes_infor],
    checkpointer=memory,
    system_prompt="Repond à la question de l'utilisation en utilisant les tools providers"

)


# Create a thread_id 
config={"configurable":{"thread_id":1}}
resp4=agent4.invoke(input={"messages":[HumanMessage("Quel est la météo de Paris")]}, config=config)

print(resp4["messages"][-1].content)

# ADD NEW TOOLS FOR THE WEB RESEARCH


# Créer un tool pour que l'agent fasse des recherches dan sle web
@tool 
def web_research(query:str, num_results:int=8) -> str:
    """
    Search in the web using DuckDuckGo
    Arg:
        query: search query string
        num_result: number of the result to return(default=8)
    Returns:
        formated search results with title, description and urls

    """
    try:
        print(f"Search tool is called  for {query}")
        ddgs_search=DDGS
        results=ddgs_search.text(query=query, max_results=num_results, backend="google")
        if not results:
            print(f"No result find for the {query}")
        formated_results=[f"search for {query}: \n"]
        for i, result in enumerate(results, 1):
            title=result.get("title", "No title")
            body=result.get("body", "No description available")
            href=result.get("href", "")
            formated_results.append (f"{i}. **{title}**, {body}, {href}")
        return formated_results
    except Exception as e:
        print(str(e))


# 
agent5 = create_agent(
    model=advanced_llm,
    tools=[web_research, get_employes_infor, get_info],
    debug=True
)

resp5=agent5.invoke(input={"messages":[HumanMessage("Donnes moi les derniers news sur l'intélligence artificielle")]})
print(display(Markdown(resp["messages"][-1].content)))


# CREATE A MIDDLEWARES WHO MANAGE A TOOLS ERRORS
@wrap_tool_call
def tools_errors(request, handler):
    """Handle tool execution erros with custom messages"""
    try:
        return handler(request)
    except Exception as e:
        print("ERROR")
        # retrun a custom error message to the model
    return ToolMessage(
        content=f"Tool error: Please check your input and try again. ({str(e)})",
        tool_call_id=request.tool_call["id"]
    )

# Agent qui  
agent=create_agent(
    model=advanced_llm,
    middleware=[tools_errors],
    tool=[web_research, get_employes_infor],
    debug=True
)


# CREATE A SYSTEM_PROMPT DYNAMIQUE
class Context(TypedDict):
    user_role: str

@dynamic_prompt
def use_role_prompt(request: ModelRequest) -> str:
    """Generate a system prompt based a user role"""
    user_role=request.runtime.context.get("user_role", "user")
    base_prompt="You are a helpful assistant."
    if user_role =="expert":
        return f"{base_prompt} Provide detailed technical responses"
    elif user_role == "beginner":
        return f"{base_prompt} Explained the context with simply and avoid jargon."
    
    return base_prompt

agent=create_agent(
    model=advanced_llm,
    middleware=[dynamic_prompt],
    tools=[],
    context_schema=Context,
    debug=True
)
# Resultat 1
result=agent.invoke(input={"messages":[
    {"role": "user", "content": "Explain Agent Loop"}
]},
context={"user_role": "expert"}
)
print(display(Markdown(result["messages"][-1].content)))

# Result 2
result=agent.invoke(input={"messages":[
    {"role": "user", "content": "Explain Agent Loop"}
]},
context={"user_role": "beginner"}
)
print(display(Markdown(result["messages"][-1].content)))

# Recupétaion d'un resultat structuré dans le output
class PersonalInfo(BaseModel):
    name: str
    email: str
    website: str

agent=create_agent(
    model=advanced_llm,
    tools=[],
    response_format=ToolStrategy(PersonalInfo)
)