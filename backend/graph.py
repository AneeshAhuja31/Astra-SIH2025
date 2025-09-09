from langgraph.graph import StateGraph, START, END
from schemas_ import GraphState
from nodes import (
    check_sql_and_graph_node,
    create_sql_query_node,
    sql_tool,
    check_graph,
    format_result_for_graph,
    create_final_answer
)
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def create_oceanographic_workflow():
    """
    Creates and compiles the complete LangGraph workflow for oceanographic queries.
    """
    workflow = StateGraph(GraphState)
    
    workflow.add_node("check_sql_and_graph", check_sql_and_graph_node)
    workflow.add_node("create_sql_query", create_sql_query_node)
    workflow.add_node("sql_tool", sql_tool)
    workflow.add_node("format_result_for_graph", format_result_for_graph)
    workflow.add_node("create_final_answer", create_final_answer)
    
    def route_after_sql_check(state: GraphState) -> str:
        """Route after checking if SQL is needed"""
        print(f"Routing: check_sql={state['check_sql']}")
        if state["check_sql"]:
            return "create_sql_query"
        else:
            return "create_final_answer"
    
    def route_after_graph_check(state: GraphState) -> str:
        """Route after checking if graph is needed"""
        print(f"Routing: check_graph={state['check_graph']}")
        if state["check_graph"]:
            return "format_result_for_graph"
        else:
            return "create_final_answer"
    
    
    workflow.add_edge(START, "check_sql_and_graph")
    
    workflow.add_conditional_edges(
        "check_sql_and_graph",
        route_after_sql_check,
        {
            "create_sql_query": "create_sql_query",
            "create_final_answer": "create_final_answer"
        }
    )
    
    workflow.add_edge("create_sql_query", "sql_tool")
    
    workflow.add_conditional_edges(
        "sql_tool",
        route_after_graph_check,
        {
            "format_result_for_graph": "format_result_for_graph",
            "create_final_answer": "create_final_answer"
        }
    )
    
    workflow.add_edge("format_result_for_graph", "create_final_answer")
    
    workflow.add_edge("create_final_answer", END)
    
    app = workflow.compile()
    return app
