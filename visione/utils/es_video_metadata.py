import os
import hashlib
import json
from elasticsearch import Elasticsearch

ES_HOST = os.environ.get('VISIONE_ES_HOST', 'localhost')
ES_PORT = int(os.environ.get('VISIONE_ES_PORT', '9200'))
ES_INDEX = os.environ.get('VISIONE_ES_INDEX', 'videos')

es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])

def file_checksum(path, block_size=65536):
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()

def get_relative_original_path(original_path):
    # If absolute, strip to relative from original-videos root
    # Handles /data/original-videos/ and subfolders
    import os
    base = os.environ.get('VISIONE_ORIGINAL_VIDEO_DIR', '/data/original-videos')
    abs_base = os.path.abspath(base)
    abs_path = os.path.abspath(original_path)
    if abs_path.startswith(abs_base):
        rel_path = os.path.relpath(abs_path, abs_base)
        return rel_path.replace('\\', '/')
    return os.path.basename(original_path)

def index_video_metadata(original_path, preprocessed_path, thumbnail_path=None):
    rel_original_path = get_relative_original_path(original_path)
    doc = {
        'original_path': rel_original_path,
        'preprocessed_path': preprocessed_path,
        'thumbnail_path': thumbnail_path,
        'original_checksum': file_checksum(original_path) if os.path.exists(original_path) else None,
        'preprocessed_checksum': file_checksum(preprocessed_path) if os.path.exists(preprocessed_path) else None,
    }
    res = es.index(index=ES_INDEX, document=doc)
    print(f"Indexed video: {json.dumps(doc, indent=2)}")
    return res

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python es_video_metadata.py <original_path> <preprocessed_path> [thumbnail_path]")
        sys.exit(1)
    original = sys.argv[1]
    preprocessed = sys.argv[2]
    thumb = sys.argv[3] if len(sys.argv) > 3 else None
    index_video_metadata(original, preprocessed, thumb)
