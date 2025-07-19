import os
import json
import sys
import yaml
import logging
from elasticsearch import Elasticsearch, exceptions

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'es_video_schema.json')

def check_elasticsearch(collection_path):
    """
    Connects to Elasticsearch, checks for the index, and verifies the setup.
    """
    print("*** Running check_es.py v2 with verbose logging ***")
    # --- Enable Verbose Logging ---
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    es_logger = logging.getLogger('elasticsearch')
    es_logger.addHandler(handler)
    es_logger.setLevel(logging.DEBUG)

    # --- Configuration ---
    config_path = os.path.join(collection_path, 'config.yaml')
    print(f"\n--- Elasticsearch Connection Test ---")
    print(f"Reading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            config = {}
    except FileNotFoundError:
        print(f"\033[91mFAILURE:\033[0m Configuration file not found at {config_path}")
        sys.exit(1)

    es_config = config.get('elasticsearch', {})
    ES_HOST = es_config.get('host', 'localhost')
    ES_PORT = es_config.get('port', 9200)
    ES_INDEX = es_config.get('index', 'videos')

    print(f"Attempting to connect to: http://{ES_HOST}:{ES_PORT}")

    try:
        es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
        if not es.ping():
            raise exceptions.ConnectionError("Ping failed. Check host and port.")
        print("\033[92mSUCCESS:\033[0m Connection to Elasticsearch established.")
    except exceptions.ConnectionError as e:
        print(f"\n\033[91mFAILURE:\033[0m Could not connect to Elasticsearch. Error: {e}")
        print("\nPlease ensure Elasticsearch is running and accessible at the specified host and port.")
        print("See detailed logs above for more information.")
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
    if len(sys.argv) != 2:
        print("Usage: python -m visione.utils.check_es <path_to_collection_directory>")
        sys.exit(1)
    collection_path = sys.argv[1]
    check_elasticsearch(collection_path)
