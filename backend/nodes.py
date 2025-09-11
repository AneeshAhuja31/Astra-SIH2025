import os
import json
import re
from langchain_core.prompts import PromptTemplate
from schemas_ import GraphState,check_sql_and_graph,graphData,ExtractFilters,ClassifyQuery
from prompts import check_sql_and_graph_prompt,create_sql_query,answer_non_sql_queestion,answer_sql_non_graph_queestion,answer_graph_question,classify_prompt,natural_answer_prompt,summarize_vectorstore_prompt
from datetime import date 
import psycopg2
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

USER = os.getenv("SUPABASE_USER")
PASSWORD = os.getenv("SUPABASE_PASSWORD")
HOST = os.getenv("SUPABASE_HOST")
PORT = os.getenv("SUPABASE_PORT")
DBNAME = os.getenv("SUPABASE_DBNAME")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
embeddings = OpenAIEmbeddings()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def extract_json_from_text(text):
    """Extract JSON from text that might contain other content"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r'\{[^{}]*"check_sql"\s*:\s*(true|false)[^{}]*"check_graph"\s*:\s*(true|false)[^{}]*\}'
        match = re.search(json_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
                json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        sql_match = re.search(r'"check_sql"\s*:\s*(true|false)', text, re.IGNORECASE)
        graph_match = re.search(r'"check_graph"\s*:\s*(true|false)', text, re.IGNORECASE)
        
        if sql_match and graph_match:
            return {
                "check_sql": sql_match.group(1).lower() == 'true',
                "check_graph": graph_match.group(1).lower() == 'true'
            }
        
        return {"check_sql": False, "check_graph": False}
    
def extract_filters(state: GraphState):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    extraction_llm = llm.with_structured_output(ExtractFilters)
    response = extraction_llm.invoke(input=state["user_prompt"])
    metadata = { "month": response.month, "year": response.year, "region":response.region,"values": response.values}
    state["metadata"] = metadata
    return state

def classify_query(state: GraphState):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY).with_structured_output(ClassifyQuery)
    
    # Extract metadata and ensure they are strings
    metadata = state.get("metadata", {})
    month = str(metadata.get("month", ""))
    year = str(metadata.get("year", ""))
    values = str(metadata.get("values", []))
    region = str(metadata.get("region", ""))
    
    # Create a single system message with all metadata
    system_content = f"""
    Context from user query:
    - Month: {month}
    - Year: {year}
    - Values: {values}
    - Region: {region}
    """
    
    response = llm.invoke([
        classify_prompt.format(),
        {"role": "system", "content": system_content},
        {"role": "user", "content": state["user_prompt"]}
    ])
    
    state["query_type"] = response.query_type
    return state

def vector_retrieve(state: GraphState):
    """
    Retrieve relevant documents from vector store based on user prompt.
    Uses year from metadata to select appropriate vector store, with validation.
    """
    # Extract metadata safely
    metadata = state.get("metadata", {})
    
    # Extract year from metadata and validate
    year = None
    if metadata and isinstance(metadata, dict):
        year = metadata.get("year")
    
    # Convert year to integer if it's a string
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    
    # Validate year range and set default
    if not year or not isinstance(year, int) or year < 2000 or year > 2017:
        print(f"Invalid year: {year}. Using default year 2017.")
        year = 2017
    
    print(f"Using vector store for year: {year}")
    
    try:
        # Load the appropriate vector store
        vectorstore_path = f"./vectorstores/{year}-faiss"
        vs = FAISS.load_local(
            folder_path=vectorstore_path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Perform similarity search
        user_prompt = state.get("user_prompt", "")
        if not user_prompt:
            print("Warning: Empty user prompt")
            state["retrieved_context"] = ""
            return state
            
        docs = vs.similarity_search(user_prompt, k=3)
        
        # Extract and combine document content
        if docs:
            context = "\n---\n".join([doc.page_content for doc in docs])
            print(f"Retrieved {len(docs)} documents from vector store")
        else:
            context = ""
            print("No documents retrieved from vector store")
        
        state["retrieved_context"] = context
        print("*****************************************************")
        print(context)
        print("*****************************************************")

    except FileNotFoundError:
        print(f"Vector store not found for year {year} at path: ./vectorstores/{year}-faiss")
        state["retrieved_context"] = ""
    except Exception as e:
        print(f"Error loading vector store or performing search: {str(e)}")
        state["retrieved_context"] = ""
    
    return state

def natural_answer(state: GraphState):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    response = llm.invoke([
        natural_answer_prompt.format(),
        {"role": "user", "content": state["user_prompt"]}
    ])
    state["generated_answer"] = response.content.strip()
    return state

def check_sql_and_graph_node(state: GraphState):
    print("--CHECK SQL--")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    response = llm.invoke([
        check_sql_and_graph_prompt.format(),
        {"role": "user", "content": state["user_prompt"]}
    ])

    try:
        response_content = response.content.strip()
        print(f"LLM Response: {response_content}")
        
        parsed_response = extract_json_from_text(response_content)
        
        state["check_sql"] = bool(parsed_response.get("check_sql", False))
        state["check_graph"] = bool(parsed_response.get("check_graph", False))
        
        print(f"Parsed - SQL required: {state['check_sql']}, Graph required: {state['check_graph']}")
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response_content}")
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
    
    sql_query = response.content.strip()
    
    if sql_query.startswith("```sql") or sql_query.startswith("```"):
        lines = sql_query.split('\n')
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
        
        if state['fetched_rows']:
            print(state['fetched_rows'])
        
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Database error: {e}")
        state['fetched_rows'] = []

    return state
        

# def format_result_for_graph(state: GraphState):
#     print("--FORMATTING GRAPH DATA--")
#     llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
#     response = llm.invoke([
#         format_graph_coordinates.format(),
#         {"role": "system", "content": str(state["fetched_rows"])},
#     ])
    
#     try:
#         response_content = response.content.strip()
#         print(f"Graph formatting response: {response_content}")
        
#         parsed_response = extract_json_from_text(response_content)
#         state["graph_data"] = {
#             "coordinates": parsed_response.get("coords", []),
#             "x_title": parsed_response.get("x_title", ""),
#             "y_title": parsed_response.get("y_title", "")
#         }
#         print(f"Graph data formatted: {len(state['graph_data']['coordinates'])} points")
#     except Exception as e:
#         print(f"Error parsing graph response: {e}")
#         state["graph_data"] = {"coordinates": [], "x_title": "", "y_title": ""}
    
#     return state

def format_result_for_graph(state: GraphState):
    """
    Format SQL results into graph coordinates using pure Python logic.
    Returns max 10 evenly distributed points with appropriate axis titles.
    """
    print("--FORMATTING GRAPH DATA--")
    
    fetched_rows = state.get("fetched_rows", [])
    if not fetched_rows:
        state["graph_data"] = {"coordinates": [], "x_title": "", "y_title": ""}
        return state
    
    # Get available columns from the first row
    available_columns = list(fetched_rows[0].keys())
    
    # Define column mappings for axis titles with units
    column_mappings = {
        'depth': ('Depth', 'm'),
        'temperature': ('Temperature', '°C'),
        'salinity': ('Salinity', 'PSU'),
        'density': ('Density', 'kg/m³'),
        'latitude': ('Latitude', '°'),
        'longitude': ('Longitude', '°'),
        'date': ('Date', ''),
        'observation_count': ('Observation Count', ''),
        'count': ('Count', ''),
        'avg': ('Average', ''),
        'sum': ('Sum', ''),
        'max': ('Maximum', ''),
        'min': ('Minimum', '')
    }
    
    def get_axis_title(column_name):
        """Generate axis title with units for a column"""
        column_lower = column_name.lower()
        for key, (title, unit) in column_mappings.items():
            if key in column_lower:
                return f"{title} ({unit})" if unit else title
        # Fallback: capitalize and clean the column name
        return column_name.replace('_', ' ').title()
    
    def is_numeric_column(column_name, rows):
        """Check if a column contains numeric data"""
        try:
            for row in rows[:5]:  # Check first 5 rows
                value = row.get(column_name)
                if value is not None:
                    float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def convert_to_numeric(value):
        """Convert value to numeric, handling different types"""
        if value is None:
            return None
        try:
            # Handle Decimal objects and other numeric types
            if hasattr(value, '__float__'):
                return float(value)
            return float(value)
        except (ValueError, TypeError):
            return None
    
    # Find numeric columns
    numeric_columns = [col for col in available_columns if is_numeric_column(col, fetched_rows)]
    
    if len(numeric_columns) < 2:
        print(f"Not enough numeric columns found. Available: {numeric_columns}")
        state["graph_data"] = {"coordinates": [], "x_title": "", "y_title": ""}
        return state
    
    # Select X and Y columns based on common oceanographic patterns
    x_col, y_col = None, None
    
    # Prioritize common oceanographic relationships
    priority_pairs = [
        ('depth', 'temperature'),
        ('depth', 'salinity'),
        ('depth', 'density'),
        ('temperature', 'salinity'),
        ('latitude', 'temperature'),
        ('longitude', 'temperature'),
    ]
    
    # Try to find priority pairs
    for x_pref, y_pref in priority_pairs:
        x_candidates = [col for col in numeric_columns if x_pref in col.lower()]
        y_candidates = [col for col in numeric_columns if y_pref in col.lower()]
        if x_candidates and y_candidates:
            x_col, y_col = x_candidates[0], y_candidates[0]
            break
    
    # Fallback: use first two numeric columns
    if not x_col or not y_col:
        x_col, y_col = numeric_columns[0], numeric_columns[1]
    
    print(f"Selected columns - X: {x_col}, Y: {y_col}")
    
    # Extract and clean data points
    data_points = []
    for row in fetched_rows:
        x_val = convert_to_numeric(row.get(x_col))
        y_val = convert_to_numeric(row.get(y_col))
        if x_val is not None and y_val is not None:
            data_points.append({"x": x_val, "y": y_val})
    
    if not data_points:
        print("No valid numeric data points found")
        state["graph_data"] = {"coordinates": [], "x_title": "", "y_title": ""}
        return state
    
    # If we have more than 10 points, select 10 evenly distributed ones
    if len(data_points) > 10:
        # Sort by x-value for even distribution
        data_points.sort(key=lambda p: p["x"])
        
        # Select evenly spaced indices
        indices = []
        step = (len(data_points) - 1) / 9  # 9 steps for 10 points
        for i in range(10):
            indices.append(int(round(i * step)))
        
        # Remove duplicates and ensure we have unique indices
        indices = sorted(list(set(indices)))
        
        # If we still have too many after deduplication, take first 10
        if len(indices) > 10:
            indices = indices[:10]
        
        selected_points = [data_points[i] for i in indices]
        print(f"Reduced from {len(data_points)} to {len(selected_points)} points")
    else:
        selected_points = data_points
        print(f"Using all {len(selected_points)} points")
    
    # Generate axis titles
    x_title = get_axis_title(x_col)
    y_title = get_axis_title(y_col)
    
    # Update state
    state["graph_data"] = {
        "coordinates": selected_points,
        "x_title": x_title,
        "y_title": y_title
    }
    
    print(f"Graph data formatted: {len(selected_points)} points")
    print(f"X-axis: {x_title}, Y-axis: {y_title}")
    
    return state

def create_final_answer(state: GraphState):
    print("--GENERATE FINAL ANSWER--")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    if state["query_type"] == "summary":
        print("Generating answer from vectorstore")
        system_content = f"Retrieved context: {state['retrieved_context']}"
        response = llm.invoke([
            summarize_vectorstore_prompt.format(),
            {"role": "system", "content": system_content},
            {"role": "user", "content": state["user_prompt"]}
        ])
        state['generated_answer'] = response.content
        
    elif not state['check_sql']:
        print("Generating non-SQL answer...")
        response = llm.invoke([
            answer_non_sql_queestion.format(),
            {"role": "user", "content": state["user_prompt"]}
        ])
        state['generated_answer'] = response.content
        
    elif not state['check_graph']:
        print("Generating SQL non-graph answer...")
        system_content = f"""
        SQL Query: {state['sql_query']}
        Fetched Rows: {state['fetched_rows'][:10]}
        """
        response = llm.invoke([
            answer_sql_non_graph_queestion.format(),
            {"role": "system", "content": system_content},
            {"role": "user", "content": state["user_prompt"]}
        ])
        state['generated_answer'] = response.content
        
    else:
        print("Generating graph answer...")
        if not state['graph_data'] or not state['graph_data']['coordinates']:
            format_result_for_graph(state)
        
        system_content = f"""
        SQL Query: {state['sql_query']}
        Fetched Rows: {state['fetched_rows'][:10]}
        Graph Metadata: {state['graph_data']}
        """
        response = llm.invoke([
            answer_graph_question.format(),
            {"role": "system", "content": system_content},
            {"role": "user", "content": state["user_prompt"]}
        ])
        state['generated_answer'] = response.content
    
    print(f"Final answer generated: {state['generated_answer'][:100]}...")
    return state