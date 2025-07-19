import os
import json
import sys
from elasticsearch import Elasticsearch, exceptions

# --- Configuration ---
ES_HOST = os.environ.get('VISIONE_ES_HOST', 'localhost')
ES_PORT = int(os.environ.get('VISIONE_ES_PORT', 9200))
ES_INDEX = os.environ.get('VISIONE_ES_INDEX', 'videos')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'es_video_schema.json')

def check_elasticsearch():
    """
    Connects to Elasticsearch, checks for the index, and verifies the setup.
    """
    print("--- Elasticsearch Connection Test ---")
    print(f"Attempting to connect to: http://{ES_HOST}:{ES_PORT}")

    try:
        es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
        if not es.ping():
            raise exceptions.ConnectionError("Ping failed. Check host and port.")
        print("\033[92mSUCCESS:\033[0m Connection to Elasticsearch established.")
    except exceptions.ConnectionError as e:
        print(f"\033[91mFAILURE:\033[0m Could not connect to Elasticsearch. Error: {e}")
        print("\nPlease ensure Elasticsearch is running and accessible at the specified host and port.")
        sys.exit(1)

    # --- Check Index ---
    print(f"\nChecking for index: '{ES_INDEX}'")
    try:
        if not es.indices.exists(index=ES_INDEX):
            print(f"\033[93mWARNING:\033[0m Index '{ES_INDEX}' does not exist.")
            if input("Would you like to create it now? (y/n): ").lower() == 'y':
                try:
                    with open(SCHEMA_PATH, 'r') as f:
                        schema = json.load(f)
                    es.indices.create(index=ES_INDEX, body=schema)
                    print(f"\033[92mSUCCESS:\033[0m Index '{ES_INDEX}' created successfully.")
                except FileNotFoundError:
                    print(f"\033[91mFAILURE:\033[0m Could not find schema file at {SCHEMA_PATH}")
                except Exception as create_e:
                    print(f"\033[91mFAILURE:\033[0m Could not create index. Error: {create_e}")
        else:
            print(f"\033[92mSUCCESS:\033[0m Index '{ES_INDEX}' found.")

    except Exception as e:
        print(f"\033[91mFAILURE:\033[0m An error occurred while checking for the index: {e}")
        sys.exit(1)

    # --- Test Indexing ---
    print("\nAttempting to perform a test write/read/delete operation...")
    test_doc_id = 'visione-test-doc'
    test_doc_body = {'test_field': 'hello_visione'}
    try:
        es.index(index=ES_INDEX, id=test_doc_id, document=test_doc_body)
        print(" - Write: OK")
        retrieved = es.get(index=ES_INDEX, id=test_doc_id)
        assert retrieved['_source'] == test_doc_body
        print(" - Read: OK")
        es.delete(index=ES_INDEX, id=test_doc_id)
        print(" - Delete: OK")
        print("\033[92mSUCCESS:\033[0m Test document indexed and deleted successfully.")
    except Exception as e:
        print(f"\033[91mFAILURE:\033[0m An error occurred during the test operation: {e}")
        print("Please check Elasticsearch logs and user permissions.")
        sys.exit(1)

    print("\n--- Test Complete ---")
    print("Your Elasticsearch connection and index appear to be configured correctly.")

if __name__ == '__main__':
    check_elasticsearch()
