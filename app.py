from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from IPython.display import Image, display
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Annotated, Sequence, List, Dict, Any
from typing_extensions import TypedDict
import json, os, re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal


# Load keys
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")



# LLM + embeddings
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
embeddings = OpenAIEmbeddings()
# -------------------------
# State Definition
# -------------------------
class OceanRAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    question: str
    query_type: Literal["specific", "summary", "irrelevant"]
    relevant: bool
    region: str
    year: str
    values: List[str]
    month: str
    retrieved_context: str
    answer: str

class ClassifyQuery(BaseModel):
    query_type : Literal["specific", "summary", "irrelevant"]


class ExtractFilters(BaseModel):
    month: str = Field(description="Full month out of the user query, no short forms. Leave blank if no value provided")
    year: str = Field(description="Year in the query.  Leave blank if no value provided")
    values: List[str] = Field(description="The values asked by the user, can be salinity, temperature or other geochemical data. Leave blank if no value provided")
    region: str = Field(description="Extract the region the user is talking about. Leave blank if you cannot determine")
    

# -------------------------
# Oceanographic RAG System
# -------------------------
class OceanographicRAGSystem:
    def __init__(self, vectorstore_dir="./vectorstores"):
        self.vectorstore_dir = vectorstore_dir

    # Node 1: Extract filters
    def extract_filters(self, state: OceanRAGState):
        extraction_llm = llm.with_structured_output(ExtractFilters)
        response = extraction_llm.invoke(input=state["question"])
        return {**state, "month": response.month, "year": response.year, "values": response.values}


    # Node 2: Classify query
    def classify_query(self, state: OceanRAGState):
        prompt = PromptTemplate(
            template="""
            Classify this query as either "specific" (exact values requested, e.g., a date/depth/location)
            or "summary" (general trends, averages, anomalies) or irrelevant

            Query: {question}
            """,
            input_variables=["question"]
        )
        chain = prompt | llm.with_structured_output(ClassifyQuery)
        query_type = chain.invoke({"question": state["question"]})
        return {**state, "query_type": query_type.query_type}

    # Node 3a: SQL Tool (placeholder for exact values)
    def sql_tool(self, state: OceanRAGState):
        year = state.get("year")
        if not year:
            return {**state, "retrieved_context": "⚠️ No year found to query SQL data."}
        # In real system: connect to yearly CSV/SQL file for this year
        dummy_context = f"SQL Result: In {year}, salinity at 200m in January 3 was 34.6 PSU."
        return {**state, "retrieved_context": dummy_context}

    # Node 3b: Vector Retrieval (dynamic by year)
    def vector_retrieve(self, state: OceanRAGState):
        year = state["year"]
        if not year:
            return {**state, "retrieved_context": "⚠️ No year found for vectorstore."}
        
        vs = FAISS.load_local(folder_path=f"./vectorstores/{year}-faiss", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = vs.similarity_search(state["question"], k=3)
        context = "\n---\n".join([d.page_content for d in docs])
        return {**state, "retrieved_context": context}



    # Node 4: Frame Answer
    def frame_answer(self, state: OceanRAGState):
        question = state["question"]
        context = state.get("retrieved_context", "")
        prompt = PromptTemplate(
            template="""
            You are an expert oceanographer. Use the given context to answer the user's query.

            Context:
            {context}

            Question: {question}

            Answer scientifically and include values when available.
            """,
            input_variables=["context","question"]
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return {**state, "answer": answer}

    # Node 5: Natural Answer for irrelevant queries
    def natural_answer(self, state: OceanRAGState):
        chain = llm | StrOutputParser()
        answer = chain.invoke(state["question"])
        return {**state, "answer": answer}

    # -------------------------
    # Build Graph
    # -------------------------
    def create_rag_graph(self):
        workflow = StateGraph(OceanRAGState)

        # Nodes
        workflow.add_node("extract", self.extract_filters)
        workflow.add_node("classify", self.classify_query)
        workflow.add_node("sql_tool", self.sql_tool)
        workflow.add_node("vector_retrieve", self.vector_retrieve)
        workflow.add_node("frame_answer", self.frame_answer)
        workflow.add_node("natural_answer", self.natural_answer)

        # Edges
        workflow.add_edge(START, "extract")
        workflow.add_edge("extract", "classify")
        workflow.add_conditional_edges(
            "classify",
            lambda s: s["query_type"],
            {"specific": "sql_tool", "summary": "vector_retrieve", "irrelevant": "natural_answer"}
        )
        workflow.add_edge("sql_tool", "frame_answer")
        workflow.add_edge("vector_retrieve", "frame_answer")
        workflow.add_edge("frame_answer", END)
        workflow.add_edge("natural_answer", END)

        wf = workflow.compile()
        
        display(Image(wf.get_graph().draw_mermaid_png()))

        return workflow.compile()

    # -------------------------
    # Query Runner
    # -------------------------
    def query(self, question: str):
        app = self.create_rag_graph()
        init_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "query_type": "",
            "relevant": True,
            "year": None,
            "retrieved_context": "",
            "answer": ""
        }
        result = app.invoke(init_state)
        return result["answer"]
 

# -------------------------
# Test Run
# -------------------------
if __name__ == "__main__":
    rag_system = OceanographicRAGSystem(vectorstore_dir="./vectorstores")

    print(rag_system.query(input("Enter your query: ")))
