from utils.vector_builder import build_faiss_index
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    domain = "backlight"
    kb_path = f"domains/{domain}/knowledge_base.txt"
    print(f"Rebuilding FAISS index for domain: {domain}")
    build_faiss_index(domain, kb_path)
    print("Index rebuilding complete.")
