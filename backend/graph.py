# from langgraph.graph import StateGraph, START, END
# from schemas_ import GraphState
# from nodes import (
#     check_sql_and_graph_node,
#     create_sql_query_node,
#     sql_tool,
#     format_result_for_graph,
#     create_final_answer
# )
# import psycopg2
# from dotenv import load_dotenv
# import os

# load_dotenv()

# def create_oceanographic_workflow():
#     """
#     Creates and compiles the complete LangGraph workflow for oceanographic queries.
#     """
#     workflow = StateGraph(GraphState)
#     workflow.add_node("check_sql_and_graph", check_sql_and_graph_node)
#     workflow.add_node("create_sql_query", create_sql_query_node)
#     workflow.add_node("sql_tool", sql_tool)
#     workflow.add_node("format_result_for_graph", format_result_for_graph)
#     workflow.add_node("create_final_answer", create_final_answer)
    
#     def route_after_sql_check(state: GraphState) -> str:
#         """Route after checking if SQL is needed"""
#         print(f"Routing: check_sql={state['check_sql']}")
#         if state["check_sql"]:
#             return "create_sql_query"
#         else:
#             return "create_final_answer"
    
#     def route_after_graph_check(state: GraphState) -> str:
#         """Route after checking if graph is needed"""
#         print(f"Routing: check_graph={state['check_graph']}")
#         if state["check_graph"]:
#             return "format_result_for_graph"
#         else:
#             return "create_final_answer"
    
    
#     workflow.add_edge(START, "check_sql_and_graph")
    
#     workflow.add_conditional_edges(
#         "check_sql_and_graph",
#         route_after_sql_check,
#         {
#             "create_sql_query": "create_sql_query",
#             "create_final_answer": "create_final_answer"
#         }
#     )
    
#     workflow.add_edge("create_sql_query", "sql_tool")
    
#     workflow.add_conditional_edges(
#         "sql_tool",
#         route_after_graph_check,
#         {
#             "format_result_for_graph": "format_result_for_graph",
#             "create_final_answer": "create_final_answer"
#         }
#     )
    
#     workflow.add_edge("format_result_for_graph", "create_final_answer")
    
#     workflow.add_edge("create_final_answer", END)
    
#     app = workflow.compile()
#     return app


from langgraph.graph import StateGraph, END
from schemas_ import GraphState
from nodes import (
    extract_filters,
    classify_query,
    vector_retrieve,
    natural_answer,
    check_sql_and_graph_node,
    create_sql_query_node,
    sql_tool,
    format_result_for_graph,
    create_final_answer
)

def create_oceanographic_workflow():
    """
    Creates and returns the oceanographic data processing workflow graph.
    """
    
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    workflow.add_node("extract", extract_filters)
    workflow.add_node("classify", classify_query)
    workflow.add_node("vectorstore_retrieve", vector_retrieve)
    workflow.add_node("natural_answer", natural_answer)
    workflow.add_node("check_sql_and_graph", check_sql_and_graph_node)
    workflow.add_node("create_sql_query", create_sql_query_node)
    workflow.add_node("sql_tool", sql_tool)
    workflow.add_node("format_result_for_graph", format_result_for_graph)
    workflow.add_node("final_answer", create_final_answer)
    
    # Set entry point
    workflow.set_entry_point("extract")
    
    # Add edges - following the flow from your diagram
    workflow.add_edge("extract", "classify")
    
    # Conditional edges from classify node
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "summary": "vectorstore_retrieve",
            "irrelevant": "natural_answer", 
            "specific": "check_sql_and_graph"
        }
    )
    
    # From vectorstore_retrieve and natural_answer to final_answer
    workflow.add_edge("vectorstore_retrieve", "final_answer")
    workflow.add_edge("natural_answer", "final_answer")
    
    # From check_sql_and_graph, always go to create_sql_query
    workflow.add_edge("check_sql_and_graph", "create_sql_query")
    
    # From create_sql_query to sql_tool
    workflow.add_edge("create_sql_query", "sql_tool")
    
    # Conditional edges from sql_tool based on graph requirement
    workflow.add_conditional_edges(
        "sql_tool",
        route_after_sql_tool,
        {
            "graph": "format_result_for_graph",
            "no_graph": "final_answer"
        }
    )
    
    # From format_result_for_graph to final_answer
    workflow.add_edge("format_result_for_graph", "final_answer")
    
    # End at final_answer
    workflow.add_edge("final_answer", END)
    
    # Compile and return the graph
    app =  workflow.compile()
    # print(app.get_graph().draw_ascii())
    # app.get_graph().draw_mermaid_png(output_file_path="./sexy.png")
    return app


def route_after_classify(state: GraphState) -> str:
    """
    Route based on query classification.
    """
    query_type = state.get("query_type", "irrelevant")
    
    if query_type == "summary":
        return "summary"
    elif query_type == "specific":
        return "specific"
    else:
        return "irrelevant"

def route_after_sql_tool(state: GraphState) -> str:
    """
    Route based on whether graph is required after SQL execution.
    """
    check_graph = state.get("check_graph", False)
    
    if check_graph:
        return "graph"
    else:
        return "no_graph"

def run_oceanographic_query(user_prompt: str) -> dict:
    """
    Main function to run a query through the oceanographic workflow.
    
    Args:
        user_prompt (str): The user's query
        
    Returns:
        dict: The final state containing the answer and any graph data
    """
    
    # Create the workflow
    app = create_oceanographic_workflow()
    
    # Initialize state
    initial_state = {
        "user_prompt": user_prompt,
        "check_sql": False,
        "check_graph": False,
        "sql_query": "",
        "fetched_rows": {},
        "graph_data": {},
        "generated_answer": "",
        "metadata": {},
        "retrieved_context": "",
        "query_type": ""
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    return final_state

# Example usage function
def example_usage():
    """
    Example of how to use the oceanographic workflow.
    """
    
    # Example queries
    queries = [
        "What is the average salinity in the Bay of Bengal in 2013?",
        "Plot temperature vs depth for the Arabian Sea",
        "Summarize the temperature trends in the Indian Ocean",
        "What are Argo floats?"
    ]
    
    print("=== Oceanographic Query Processing Examples ===\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 50)
        
        try:
            result = run_oceanographic_query(query)
            
            print(f"Query Type: {result.get('query_type', 'Unknown')}")
            print(f"SQL Required: {result.get('check_sql', False)}")
            print(f"Graph Required: {result.get('check_graph', False)}")
            
            if result.get('sql_query'):
                print(f"SQL Query: {result['sql_query']}")
            
            if result.get('graph_data') and result['graph_data'].get('coordinates'):
                print(f"Graph Data: {len(result['graph_data']['coordinates'])} points")
                print(f"X-Axis: {result['graph_data'].get('x_title', 'N/A')}")
                print(f"Y-Axis: {result['graph_data'].get('y_title', 'N/A')}")
            
            print(f"Answer: {result.get('generated_answer', 'No answer generated')}")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
        
        print("\n" + "="*60 + "\n")

# if __name__ == "__main__":
#     # Run examples when script is executed directly
#     example_usage()
create_oceanographic_workflow()
