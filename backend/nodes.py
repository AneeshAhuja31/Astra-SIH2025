import os
from backend.schemas import GraphState,check_sql_and_graph,graphData
from backend.prompts import check_sql_and_graph_prompt,create_sql_query,answer_non_sql_queestion,answer_sql_non_graph_queestion,answer_graph_question,format_graph_coordinates
from datetime import date 
import psycopg2
load_dotenv()

USER = os.getenv("SUPABASE_USER")
PASSWORD = os.getenv("SUPABASE_PASSWORD")
HOST = os.getenv("SUPABASE_HOST")
PORT = os.getenv("SUPABASE_PORT")
DBNAME = os.getenv("SUPABASE_DBNAME")

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_aPI_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

def check_sql_and_graph_node(state:GraphState):
    print("--CHECK SQL--")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GEMINI_API_KEY).with_structured_output(check_sql_and_graph)
    response = llm.invoke([
        check_sql_and_graph_prompt.format(),
        {"role": "user", "content": state["user_prompt"]}
    ])

    state["check_sql"] = response.check_sql
    state["check_graph"] = response.check_graph

    return state


def create_sql_query(state:GraphState):
    print("--CREATE SQL QUERY--")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GEMINI_API_KEY)
    response = llm.invoke([
        create_sql_query.format(),
        {"role": "user", "content": state["user_prompt"]}
    ])
    state['sql_query'] = response.content
    return state

def sql_tool(state:GraphState):
    print("--FETCHING SQL DATA---")
    connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
    print("Connection successful!")
    cursor = connection.cursor()
    cursor.execute(state['sql_query'])
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    state['fetched_rows'] = [dict(zip(column_names, row)) for row in rows]
    print(f"Fetched {len(state['fetched_rows'])} rows.")
    
    cursor.close()
    connection.close()

    return state

def check_graph(state:GraphState):
    print("--CHECKING GRAPH--")
    if state['check_graph']:
        return "use_graph"
    return "dont_use_graph"

def format_result_for_graph(state:GraphState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY).with_structured_output(graphData)
    response = llm.invoke([
        format_graph_coordinates.format(),
        {"role": "system", "content": state["fetched_rows"]},
    ])
    state["graph_data"] = {
        "coordinates": response.coords,
        "x_title":response.x_title,
        "y_title":response.y_title
    }
    return state

def create_final_answer(state: GraphState):
    print("--GENERATE FINAL ANSWER--")
    
    if not state['check_sql']:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
        response = llm.invoke([
            answer_non_sql_queestion.format(),
            {"role": "user", "content": state["user_prompt"]}
        ])
        state['generated_answer'] = response.content
        return state

    elif not state['check_graph']:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
        response = llm.invoke([
            answer_sql_non_graph_queestion.format(),
            {"role": "user", "content": state["user_prompt"]},
            {"role": "system", "content": f"SQL Query: {state['sql_query']}"},
            {"role": "system", "content": f"Fetched Rows: {state['fetched_rows']}"}
        ])
        state['generated_answer'] = response.content
        return state

    else:
        graph_metadata = format_result_for_graph(state)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
        response = llm.invoke([
            answer_graph_question.format(),
            {"role": "user", "content": state["user_prompt"]},
            {"role": "system", "content": f"SQL Query: {state['sql_query']}"},
            {"role": "system", "content": f"Fetched Rows: {state['fetched_rows']}"},
            {"role": "system", "content": f"Graph Metadata: {graph_metadata}"}
        ])
        state['generated_answer'] = response.content
        return state