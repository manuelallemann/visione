import importlib
import sys

CORE_LIBS = [
    'numpy',
    'pandas',
    'torch',
    'torchvision',
    'cv2',
    'faiss',
    'sklearn',
]

def log_versions():
    print('--- Core Library Versions ---')
    for lib in CORE_LIBS:
        try:
            if lib == 'cv2':
                mod = importlib.import_module('cv2')
                print(f"opencv-python: {mod.__version__}")
            elif lib == 'faiss':
                try:
                    mod = importlib.import_module('faiss')
                except ImportError:
                    mod = importlib.import_module('faiss_cpu')
                print(f"faiss-cpu: {mod.__version__}")
            elif lib == 'sklearn':
                mod = importlib.import_module('sklearn')
                print(f"scikit-learn: {mod.__version__}")
            else:
                mod = importlib.import_module(lib)
                print(f"{lib}: {mod.__version__}")
        except Exception as e:
            print(f"{lib}: not installed or error ({e})")
    print('-----------------------------')

if __name__ == "__main__":
    log_versions()
