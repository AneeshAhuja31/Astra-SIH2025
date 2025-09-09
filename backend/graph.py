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

def test_database_connection():
    """Test database connection and check if required tables exist"""
    print("Testing database connection...")
    try:
        USER = os.getenv("SUPABASE_USER")
        PASSWORD = os.getenv("SUPABASE_PASSWORD")
        HOST = os.getenv("SUPABASE_HOST")
        PORT = os.getenv("SUPABASE_PORT")
        DBNAME = os.getenv("SUPABASE_DBNAME")
        
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        print("✅ Database connection successful!")
        
        cursor = connection.cursor()
        
        # Check if argo_data_2001 table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'argo_data_2001'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if table_exists:
            print("✅ Table argo_data_2001 exists!")
            
            # Check table structure and sample data
            cursor.execute("SELECT COUNT(*) FROM argo_data_2001;")
            row_count = cursor.fetchone()[0]
            print(f"✅ Table contains {row_count} rows")
            
            if row_count > 0:
                cursor.execute("SELECT * FROM argo_data_2001 LIMIT 1;")
                sample_row = cursor.fetchone()
                column_names = [desc[0] for desc in cursor.description]
                print(f"✅ Sample data structure: {dict(zip(column_names, sample_row))}")
            else:
                print("⚠️  Table exists but is empty")
        else:
            print("❌ Table argo_data_2001 does not exist!")
            return False
        
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_highest_argos_region_2001():
    """
    Test case: Find the region with the highest number of Argo observations in 2001.
    
    Expected behavior:
    - check_sql: True (needs database query)
    - check_graph: False (just needs count, no visualization)
    - Should generate SQL query to count observations by region
    - Should return the region with maximum count
    """
    print("="*60)
    print("TEST CASE: Region with Highest Argo Count in 2001")
    print("="*60)
    
    # First test database connection
    if not test_database_connection():
        print("❌ Database connection test failed. Aborting workflow test.")
        return None, False
    
    # Create the workflow
    app = create_oceanographic_workflow()
    
    # Define the test query
    test_query = "Which region had the highest number of Argo observations in 2001?"
    
    # Initialize state
    initial_state = {
        "user_prompt": test_query,
        "check_sql": False,
        "check_graph": False,
        "sql_query": "",
        "fetched_rows": {},
        "graph_data": {},
        "generated_answer": ""
    }
    
    print(f"\nInput Query: {test_query}")
    print("\nExecuting workflow...")
    print("-" * 40)
    
    try:
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Display results
        print("\n" + "="*40)
        print("WORKFLOW EXECUTION RESULTS")
        print("="*40)
        
        print(f"✓ SQL Required: {final_state['check_sql']}")
        print(f"✓ Graph Required: {final_state['check_graph']}")
        
        if final_state['sql_query']:
            print(f"\n✓ Generated SQL Query:")
            print(f"   {final_state['sql_query']}")
        
        if final_state['fetched_rows']:
            print(f"\n✓ Data Retrieved: {len(final_state['fetched_rows'])} rows")
            if isinstance(final_state['fetched_rows'], list) and final_state['fetched_rows']:
                print("   Sample data:", final_state['fetched_rows'][:3])  # Show first 3 rows
            else:
                print("   Data:", final_state['fetched_rows'])
        
        print(f"\n✓ Final Answer:")
        print(f"   {final_state['generated_answer']}")
        
        # Validate expected behavior
        print(f"\n" + "="*40)
        print("VALIDATION")
        print("="*40)
        
        validations = [
            ("SQL should be required", final_state['check_sql'] == True),
            ("Graph should not be required", final_state['check_graph'] == False),
            ("SQL query should be generated", bool(final_state['sql_query'])),
            ("Data should be fetched", bool(final_state['fetched_rows'])),
            ("Final answer should be provided", bool(final_state['generated_answer']))
        ]
        
        for description, passed in validations:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {description}")
        
        all_passed = all(passed for _, passed in validations)
        
        print(f"\n{'🎉 TEST PASSED' if all_passed else '⚠️  TEST FAILED'}")
        
        return final_state, all_passed
        
    except Exception as e:
        print(f"\n❌ ERROR during workflow execution:")
        print(f"   {str(e)}")
        
        # Print detailed traceback for debugging
        import traceback
        print(f"\nDetailed traceback:")
        traceback.print_exc()
        
        return None, False

def test_multiple_queries():
    """Test multiple different query types"""
    test_queries = [
        {
            "query": "Which region had the highest number of Argo observations in 2001?",
            "expected_sql": True,
            "expected_graph": False,
            "description": "SQL aggregation query"
        },
        {
            "query": "What is an Argo float?",
            "expected_sql": False,
            "expected_graph": False,
            "description": "General knowledge query"
        },
        {
            "query": "Show me temperature vs depth for the Bay of Bengal in 2005",
            "expected_sql": True,
            "expected_graph": True,
            "description": "Graph visualization query"
        }
    ]
    
    print("="*60)
    print("MULTIPLE QUERY TESTS")
    print("="*60)
    
    app = create_oceanographic_workflow()
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- TEST {i}: {test_case['description']} ---")
        print(f"Query: {test_case['query']}")
        
        initial_state = {
            "user_prompt": test_case['query'],
            "check_sql": False,
            "check_graph": False,
            "sql_query": "",
            "fetched_rows": {},
            "graph_data": {},
            "generated_answer": ""
        }
        
        try:
            final_state = app.invoke(initial_state)
            
            # Check results
            sql_correct = final_state['check_sql'] == test_case['expected_sql']
            graph_correct = final_state['check_graph'] == test_case['expected_graph']
            
            print(f"Expected SQL: {test_case['expected_sql']}, Got: {final_state['check_sql']} {'✅' if sql_correct else '❌'}")
            print(f"Expected Graph: {test_case['expected_graph']}, Got: {final_state['check_graph']} {'✅' if graph_correct else '❌'}")
            
            if final_state['sql_query']:
                print(f"SQL Query: {final_state['sql_query'][:100]}...")
            
            print(f"Answer: {final_state['generated_answer'][:100]}...")
            
            results.append({
                'test': test_case['description'],
                'sql_correct': sql_correct,
                'graph_correct': graph_correct,
                'passed': sql_correct and graph_correct
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'test': test_case['description'],
                'sql_correct': False,
                'graph_correct': False,
                'passed': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("MULTI-TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(1 for r in results if r['passed'])
    total_tests = len(results)
    
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status}: {result['test']}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    return results

if __name__ == "__main__":
    # Run the single test case
    print("Running single test case...")
    result, success = test_highest_argos_region_2001()
    
    if success:
        print(f"\n{'='*60}")
        print("SINGLE TEST SUMMARY: SUCCESS ✅")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("SINGLE TEST SUMMARY: FAILED ❌")
        print(f"{'='*60}")
        if result is None:
            print("Workflow execution failed completely.")
        else:
            print("Please check the validation results above.")
    
    # Run multiple test cases
    print(f"\n{'='*80}")
    print("Running multiple test cases...")
    multi_results = test_multiple_queries()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    single_test_status = "PASSED" if success else "FAILED"
    multi_passed = sum(1 for r in multi_results if r['passed'])
    multi_total = len(multi_results)
    
    print(f"Single Test Case: {single_test_status}")
    print(f"Multiple Test Cases: {multi_passed}/{multi_total} PASSED")
    
    if success and multi_passed == multi_total:
        print("\n🎉 ALL TESTS PASSED! Your workflow is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the following:")
        print("1. Database connection and table existence")
        print("2. Environment variables (API keys, DB credentials)")
        print("3. LLM response parsing")
        print("4. SQL query generation")