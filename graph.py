from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText

# Charge le .env
load_dotenv(override=True)

# Vérifie que la clé est bien chargée
api_key = os.getenv("OPENAI_API_KEY")

# from langchain.chat_models import ChatOpenAI

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
# from dotenv.ipython import load_dotenv
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import Chroma
# from dotenv import load_dotenv
# import os

# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# load_dotenv(load_dotenv(override=True))



# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

texts = [
    "Je me nomme Rahma DIAWANE"
    "J'ai 24ans "
    "Je travaille en freelance en tant que développeuse d'intelligence artificielle, où je crée des solutions innovantes pour mes clients en utilisant les dernières technologies d'IA"
    "J'ai des compétences en N8N"
    "Je suis une futur ai engineer , now I work in CIH bank",
    "Je étudiant entrepreneur",
    "Je suis passionné par le développement de l'intelligence artificielle et j'aime apprendre de nouvelles technologies",
    "Je travaille à CIH bank en tant que stagiaire dans le département de l'innovation, où je contribue à des projets liés à l'intelligence artificielle et à la transformation digitale"
    "Je gagne 5000 dirhams par mois et j'ai une seniorité de 1 an dans le domaine de l'intelligence artificielle"

]

# embedding_model = OpenAIEmbeddings(model = "test-embedding-ada-002") 
vectorstore = Chroma.from_texts(texts, embedding_model,collection_name="CV_collection")

retrieval = vectorstore.as_retriever(kwargs={"k":5})
retrieval_tool = create_retriever_tool(
    retriever=retrieval, name="kb_search", description="Search information about me"
    )
@tool
def send_mail(mail:str, subject:str, content:str):
    """Send email to the givel email with the provided subject and content"""
    print("=="*50)
    print("send_mail tool invoked")
    print("=="*50)

    # CONFIG
    sender_email = "ton_email@gmail.com"
    app_password = "ton_app_password"  # ⚠️ celui généré

    # Création du message
    msg = MIMEText(content)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = mail

    # Envoi via Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("Email envoyé avec succès")
    except Exception as e:
        print("Erreur:", e)

    return f"This mail has been sent : destination {mail}, subject : {subject}, content : {content}"
@tool
def get_employee_info (name: str):
    """Get more  informationa about this employee (name, salary, seniority)"""
    print("=="*50)
    print("get_emplee_info tool invoked")
    print("=="*50)

    return {"name": name, "salary": 15000, "seniority": 2}


# Création de l'agent
graph = create_agent(
    model=llm,
    tools=[get_employee_info, retrieval_tool, send_mail],
    system_prompt="answer the user question using prived tools",

)
resp = graph.invoke(
    input={"messages":[HumanMessage("What is my salary and seniority?")]},


)
print(resp['messages'][-1].content)