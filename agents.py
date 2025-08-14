from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Load FAISS
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str

# Tools
def retrieve_from_pdf(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

def web_search(query: str) -> str:
    results = tavily.search(query=query, max_results=3)
    return "\n\n".join([r["content"] for r in results["results"]])

pdf_tool = Tool(name="pdf_retriever", func=retrieve_from_pdf, description="Retrieve from PDF")
web_tool = Tool(name="web_search", func=web_search, description="Search the web")

# Router Node
router_prompt = ChatPromptTemplate.from_template(
    "Classify the query: {query}\n"
    "Options: 'pdf' (if answerable from documents), 'web' (if needs external info or requested), 'clarify' (if ambiguous).\n"
    "Output only the option."
)
def router(state: AgentState):
    query = state["messages"][-1].content
    classification = llm.invoke(router_prompt.format(query=query)).content.strip().lower()
    return {"messages": state["messages"], "next": classification}

# Clarification Node
clarify_prompt = ChatPromptTemplate.from_template(
    "The query '{query}' is ambiguous. Suggest clarifications based on context: {context}"
)
def clarify(state: AgentState):
    query = state["messages"][-1].content
    context = retrieve_from_pdf(query)
    response = llm.invoke(clarify_prompt.format(query=query, context=context)).content.strip().lower()
    return {"messages": state["messages"] + [AIMessage(content=response)]}

# RAG Node
rag_prompt = ChatPromptTemplate.from_template(
    "Answer based on context: {context}\nQuestion: {query}\nChat history: {history}"
)
def rag_agent(state: AgentState):
    query = state["messages"][-1].content
    context = pdf_tool.run(query)
    history = "\n".join([m.content for m in state["messages"][:-1]])
    response = llm.invoke(rag_prompt.format(context=context, query=query, history=history)).content
    return {"messages": state["messages"] + [AIMessage(content=response)]}

# Web Node
web_prompt = ChatPromptTemplate.from_template(
    "Summarize web results: {results}\nFor question: {query}"
)
def web_agent(state: AgentState):
    query = state["messages"][-1].content
    results = web_tool.run(query)
    response = llm.invoke(web_prompt.format(results=results, query=query)).content
    return {"messages": state["messages"] + [AIMessage(content=response)]}

# Build Graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("router", router)
workflow.add_node("clarify", clarify)
workflow.add_node("rag", rag_agent)
workflow.add_node("web", web_agent)
workflow.add_edge("clarify", END)
workflow.add_edge("rag", END)
workflow.add_edge("web", END)

# Condition edges from router
workflow.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {"clarify": "clarify", "pdf": "rag", "web": "web"},
)

workflow.set_entry_point("router")
app = workflow.compile()

# Function to run graph
def run_agent(query: str, session_id: str, messages: list[BaseMessage] = []):
    inputs = {"messages": messages + [HumanMessage(content=query)], "session_id": session_id}
    outputs = app.invoke(inputs)
    return outputs["messages"][-1].content, outputs["messages"]
