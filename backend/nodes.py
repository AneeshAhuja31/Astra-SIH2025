import os
import json
import re

from schemas_ import GraphState,check_sql_and_graph,graphData
from prompts import check_sql_and_graph_prompt,create_sql_query,answer_non_sql_queestion,answer_sql_non_graph_queestion,answer_graph_question,format_graph_coordinates
from datetime import date 
import psycopg2
from dotenv import load_dotenv
load_dotenv()

USER = os.getenv("SUPABASE_USER")
PASSWORD = os.getenv("SUPABASE_PASSWORD")
HOST = os.getenv("SUPABASE_HOST")
PORT = os.getenv("SUPABASE_PORT")
DBNAME = os.getenv("SUPABASE_DBNAME")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_aPI_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def extract_json_from_text(text):
    """Extract JSON from text that might contain other content"""
    try:
        # First try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find JSON content within the text
        # Look for JSON-like patterns
        json_pattern = r'\{[^{}]*"check_sql"\s*:\s*(true|false)[^{}]*"check_graph"\s*:\s*(true|false)[^{}]*\}'
        match = re.search(json_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                # Clean up the matched JSON
                json_str = match.group(0)
                # Replace true/false with proper case
                json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
                json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract boolean values manually
        sql_match = re.search(r'"check_sql"\s*:\s*(true|false)', text, re.IGNORECASE)
        graph_match = re.search(r'"check_graph"\s*:\s*(true|false)', text, re.IGNORECASE)
        
        if sql_match and graph_match:
            return {
                "check_sql": sql_match.group(1).lower() == 'true',
                "check_graph": graph_match.group(1).lower() == 'true'
            }
        
        # If all else fails, return defaults
        return {"check_sql": False, "check_graph": False}

def check_sql_and_graph_node(state: GraphState):
    print("--CHECK SQL--")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    response = llm.invoke([
        check_sql_and_graph_prompt.format(),
        {"role": "user", "content": state["user_prompt"]}
    ])

    # Parse the JSON response from the LLM
    try:
        # Extract the content from the AIMessage
        response_content = response.content.strip()
        print(f"LLM Response: {response_content}")
        
        # Parse the JSON response with improved error handling
        parsed_response = extract_json_from_text(response_content)
        
        # Update state with the parsed values
        state["check_sql"] = bool(parsed_response.get("check_sql", False))
        state["check_graph"] = bool(parsed_response.get("check_graph", False))
        
        print(f"Parsed - SQL required: {state['check_sql']}, Graph required: {state['check_graph']}")
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response_content}")
        # Default fallback values
        state["check_sql"] = False
        state["check_graph"] = False

    return state


def create_sql_query_node(state: GraphState):
    print("--CREATE SQL QUERY--")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    response = llm.invoke([
        create_sql_query.format(),
        {"role": "user", "content": state["user_prompt"]}
    ])
    
    # Clean up the SQL query - remove any markdown formatting or extra text
    sql_query = response.content.strip()
    
    # Remove markdown code block formatting if present
    if sql_query.startswith("```sql") or sql_query.startswith("```"):
        lines = sql_query.split('\n')
        # Remove the first line (```sql or ```) and last line (```) if they exist
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql_query = '\n'.join(lines).strip()
    
    state['sql_query'] = sql_query
    print(f"Generated SQL Query: {state['sql_query']}")
    return state


def sql_tool(state: GraphState):
    print("--FETCHING SQL DATA---")
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        print("Connection successful!")
        cursor = connection.cursor()
        
        print(f"Executing query: {state['sql_query']}")
        cursor.execute(state['sql_query'])
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        state['fetched_rows'] = [dict(zip(column_names, row)) for row in rows]
        print(f"Fetched {len(state['fetched_rows'])} rows.")
        
        # Print sample data for debugging
        if state['fetched_rows']:
            print(f"Sample row: {state['fetched_rows'][0]}")
        
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Database error: {e}")
        state['fetched_rows'] = []

    return state


def check_graph(state: GraphState):
    print("--CHECKING GRAPH--")
    return state['check_graph']
        

def format_result_for_graph(state: GraphState):
    print("--FORMATTING GRAPH DATA--")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    response = llm.invoke([
        format_graph_coordinates.format(),
        {"role": "system", "content": str(state["fetched_rows"])},
    ])
    
    try:
        # Parse the JSON response for graph data
        response_content = response.content.strip()
        print(f"Graph formatting response: {response_content}")
        
        parsed_response = extract_json_from_text(response_content)
        state["graph_data"] = {
            "coordinates": parsed_response.get("coords", []),
            "x_title": parsed_response.get("x_title", ""),
            "y_title": parsed_response.get("y_title", "")
        }
        print(f"Graph data formatted: {len(state['graph_data']['coordinates'])} points")
    except Exception as e:
        print(f"Error parsing graph response: {e}")
        state["graph_data"] = {"coordinates": [], "x_title": "", "y_title": ""}
    
    return state


def create_final_answer(state: GraphState):
    print("--GENERATE FINAL ANSWER--")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    if not state['check_sql']:
        print("Generating non-SQL answer...")
        response = llm.invoke([
            answer_non_sql_queestion.format(),
            {"role": "user", "content": state["user_prompt"]}
        ])
        state['generated_answer'] = response.content
        
    elif not state['check_graph']:
        print("Generating SQL non-graph answer...")
        response = llm.invoke([
            answer_sql_non_graph_queestion.format(),
            {"role": "user", "content": state["user_prompt"]},
            {"role": "system", "content": f"SQL Query: {state['sql_query']}"},
            {"role": "system", "content": f"Fetched Rows: {state['fetched_rows']}"}
        ])
        state['generated_answer'] = response.content
        
    else:
        print("Generating graph answer...")
        # Format graph data first if needed
        if not state['graph_data'] or not state['graph_data']['coordinates']:
            format_result_for_graph(state)
            
        response = llm.invoke([
            answer_graph_question.format(),
            {"role": "user", "content": state["user_prompt"]},
            {"role": "system", "content": f"SQL Query: {state['sql_query']}"},
            {"role": "system", "content": f"Fetched Rows: {state['fetched_rows']}"},
            {"role": "system", "content": f"Graph Metadata: {state['graph_data']}"}
        ])
        state['generated_answer'] = response.content
    
    print(f"Final answer generated: {state['generated_answer'][:100]}...")
    return state