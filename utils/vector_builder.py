# # utils/vector_builder.py

# import os
# import json
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# import logging

# logger = logging.getLogger(__name__)

# def build_faiss_index(domain: str, kb_file_path: str):
#     """
#     Build and save FAISS vector index for the specified domain.
#     """
#     try:
#         with open(kb_file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # The file contains a stream of JSON objects. We need to parse them one by one.
#         json_content = []
#         decoder = json.JSONDecoder()
#         content = content.strip()
#         pos = 0

#         while pos < len(content):
#             try:
#                 obj, pos = decoder.raw_decode(content, pos)
#                 json_content.append(obj)
#                 # Skip whitespace and newlines between objects
#                 while pos < len(content) and content[pos].isspace():
#                     pos += 1
#             except json.JSONDecodeError:
#                 # Handle cases where there might be trailing characters or malformed data
#                 break

#         # Flatten the list since the knowledge base can contain both lists of objects and single objects.
#         all_items = []
#         for item in json_content:
#             if isinstance(item, list):
#                 all_items.extend(item)
#             else:
#                 all_items.append(item)

#         documents = []
#         for item in all_items:
#             # Create a single coherent block of text for each item.
#             # This keeps the symptom, causes, and solutions together.
#             content_parts = []
#             component = item.get("component", "generic")

#             # Top-level symptom block
#             if "symptom" in item:
#                 content_parts.append(f"Component: {component}")
#                 content_parts.append(f"Symptom: {item['symptom']}")
#                 if item.get("possible_causes"):
#                     causes_str = '\n'.join([f"- {cause}" for cause in item["possible_causes"]])
#                     content_parts.append(f"Possible Causes:\n{causes_str}")
#                 if item.get("solutions"):
#                     solutions_str = '\n'.join([f"- {solution}" for solution in item["solutions"]])
#                     content_parts.append(f"Solutions:\n{solutions_str}")

#             # Handle nested 'issues' structure
#             elif "component" in item and 'issues' in item:
#                 for issue in item['issues']:
#                     if isinstance(issue, dict) and 'symptom' in issue:
#                         issue_parts = [f"Component: {component}"]
#                         issue_parts.append(f"Symptom: {issue['symptom']}")
#                         if issue.get("possible_causes"):
#                             causes_str = '\n'.join([f"- {cause}" for cause in issue["possible_causes"]])
#                             issue_parts.append(f"Possible Causes:\n{causes_str}")
#                         if issue.get("solutions"):
#                             solutions_str = '\n'.join([f"- {solution}" for solution in issue["solutions"]])
#                             issue_parts.append(f"Solutions:\n{solutions_str}")
                        
#                         # Create a separate document for each nested issue
#                         full_content = "\n\n".join(issue_parts)
#                         documents.append(Document(
#                             page_content=full_content,
#                             metadata={"component": component}
#                         ))

#                 continue # Skip creating a main document for the component itself

#             if content_parts:
#                 full_content = "\n\n".join(content_parts)
#                 documents.append(Document(
#                     page_content=full_content,
#                     metadata={"component": component}
#                 ))

#         print(f"the document is {documents}")
#         embedding = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )

#         vectorstore = FAISS.from_documents(documents, embedding) # Use 'documents' directly
#         index_dir = os.path.join("domains", domain, "faiss_index")
#         vectorstore.save_local(index_dir)
#         logger.info(f"FAISS index built and saved to {index_dir}")

#     except Exception as e:
#         logger.error(f"Failed to build FAISS index for domain '{domain}': {e}")
#         raise

# current mooxy server code ->

# utils/vector_builder.py

# import os
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# import logging
# import torch

# logger = logging.getLogger(__name__)

# def build_faiss_index(domain: str, kb_file_path: str):
#     """
#     Build and save FAISS vector index for the specified domain.
#     """
#     try:
#         loader = TextLoader(kb_file_path)
#         docs = loader.load()

#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             length_function=len,
#             separators = ["\n\n","###","*","."]
#         )
#         chunks = splitter.split_documents(docs)

#         embedding = HuggingFaceEmbeddings(
#             model_name="intfloat/multilingual-e5-large",
#             model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"},
#             encode_kwargs={'normalize_embeddings': True}
#         )

#         vectorstore = FAISS.from_documents(chunks, embedding)
#         index_dir = os.path.join("domains", domain, "faiss_index")
#         vectorstore.save_local(index_dir)
#         logger.info(f"FAISS index built and saved to {index_dir}")

#     except Exception as e:
#         logger.error(f"Failed to build FAISS index for domain '{domain}': {e}")
#         raise


# utils/vector_builder.py

import os
import re
import json
import torch
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def preprocess_kb_file(kb_file_path: str) -> str:
    """
    Preprocess the knowledge base file to ensure valid JSON format.
    - Removes trailing commas before closing braces/brackets
    - Wraps multiple JSON objects in an array if needed
    - Fixes common JSON formatting issues
    """
    with open(kb_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing commas before closing braces/brackets
    content = re.sub(r',\s*([}\]])(?!\s*[{\[])', r'\1', content)
    
    # Fix missing commas between objects
    content = re.sub(r'}\s*{', '},{', content)
    
    # Ensure proper JSON array format
    content = content.strip()
    if not content.startswith('[') and not content.startswith('{'):
        content = '[' + content + ']'
    elif not content.startswith('[') and content.startswith('{'):
        content = '[' + content + ']'
    
    return content

def parse_kb_content(content):
    """Parse knowledge base content with multiple fallback strategies."""
    # Try parsing as a single JSON array first
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    # Try to find and parse individual JSON objects
    objects = []
    decoder = json.JSONDecoder()
    pos = 0
    content = content.strip()

    print("coming inside the parse_kb_content function , with content {}".format(content))
    
    
    while pos < len(content):
        try:
            # Skip whitespace
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
                
            # Try to parse next JSON object
            obj, pos = decoder.raw_decode(content, pos)
            objects.append(obj)
        except json.JSONDecodeError as e:
            # If we can't parse at this position, try the next character
            pos += 1
    
    if objects:
        print(f"the objects arre {objects}")
        return objects
    
    # If still no luck, try to fix common JSON issues
    try:
        print("13343144***************")
        # Remove trailing commas
        content = re.sub(r',\s*([}\]])', r'\1', content)
        # Fix missing commas between objects
        content = re.sub(r'}\s*{', '},{', content)
        # Ensure proper array format
        if not content.startswith('[') and not content.startswith('{'):
            content = '[' + content + ']'
        elif content.startswith('{') and not content.startswith('[{'):
            content = '[' + content + ']'
        
        data = json.loads(content)
        print("----------*****************-----------------")
        print(f"the data after formatting is {data}")
        print("----------*****************-----------------")
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse knowledge base: {str(e)}")


def build_faiss_index(domain: str, kb_file_path: str):
    """
    Build and save FAISS vector index for the specified domain.
    Uses regex-based chunking to keep symptoms, causes, and solutions together.
    """

    try:
        # --- Step 1: Read file ---
        if not os.path.exists(kb_file_path):
            raise FileNotFoundError(f"Knowledge base file not found: {kb_file_path}")

        with open(kb_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            raise ValueError("Knowledge base file is empty.")

        # --- Step 2: Parse JSON content ---
        try:
            # First try to parse as a single JSON object
            kb_data = parse_kb_content(content)
            print("parsed the kb data in parse kb content method ")
        except json.JSONDecodeError:
            # If that fails, try to parse as newline-delimited JSON
            try:
                kb_data = [json.loads(line) for line in content.split('\n') if line.strip()]
                print("parsed the kb data in other method ")
            except json.JSONDecodeError:
                # If both fail, try to fix common JSON issues and parse
                content = re.sub(r',\s*([}\]])(?!\s*[{\[])', r'\1', content)  # Remove trailing commas
                content = re.sub(r'}\s*{', '},{', content)  # Add missing commas between objects
                if not content.startswith('[') and not content.startswith('{'):
                    content = '[' + content + ']'
                kb_data = json.loads(content)
                if not isinstance(kb_data, list):
                    kb_data = [kb_data]

        # --- Step 3: Process KB data into chunks ---
        chunks = []
        print(f"Processing {len(kb_data)} items...")
        print(f"the kb data that we got is {kb_data}")
        
        def create_chunk(component, data, parent=None):
            """Helper function to create a chunk from component data"""
            chunk_parts = []
            
            # Add component info if available
            if component and component != "generic":
                chunk_parts.append(f"Component: {component}")
            
            # Add symptom/description
            if 'symptom' in data:
                chunk_parts.append(f"Symptom: {data['symptom']}")
            
            # Add possible causes
            if 'possible_causes' in data and data['possible_causes']:
                causes = '\n'.join(f"- {cause}" for cause in data['possible_causes'])
                chunk_parts.append(f"Possible Causes:\n{causes}")
            
            # Add tests and diagnostics
            if 'tests_diagnostics' in data and data['tests_diagnostics']:
                tests = '\n'.join(f"- {test}" for test in data['tests_diagnostics'])
                chunk_parts.append(f"Tests & Diagnostics:\n{tests}")
            
            # Add solutions
            if 'solutions' in data and data['solutions']:
                solutions = '\n'.join(f"- {sol}" for sol in data['solutions'])
                chunk_parts.append(f"Solutions:\n{solutions}")
            
            # Add knowledge facts if available
            if 'knowledge_facts' in data and data['knowledge_facts']:
                facts = '\n'.join(f"• {fact}" for fact in data['knowledge_facts'])
                chunk_parts.append(f"Key Facts:\n{facts}")
            
            # Add analogies if available
            if 'analogy' in data:
                chunk_parts.append(f"Note: {data['analogy']}")
            
            # Add description if available
            if 'description' in data:
                chunk_parts.append(f"Description: {data['description']}")
                
            if 'faq' in data:
                for q,a in zip(data['faq'].get('questions',[]), data['faq'].get('answers',[])):
                    chunk_parts.append(f"FAQ Q: {q}\nFAQ A: {a}")
            
            # Add diagnostic_procedure if available
            if 'diagnostic_procedure' in data:
                chunk_parts.append(f"Diagnostic Procedure: {data['diagnostic_procedure']}")
            
            # Add functionality details if available
            if 'functionality_details' in data:
                # Handle functionality_details as dictionary
                if isinstance(data['functionality_details'], dict):
                    for key, value in data['functionality_details'].items():
                        if isinstance(value, str):
                            chunk_parts.append(f"{key.title()}: {value}")
                # Or as list
                elif isinstance(data['functionality_details'], list):
                    details = '\n'.join(f"- {detail}" for detail in data['functionality_details'])
                    chunk_parts.append(f"Functionality Details:\n{details}")
            
            # Add reference bands if available
            if 'reference_bands' in data and isinstance(data['reference_bands'], dict):
                band_info = []
                for band_type, bands in data['reference_bands'].items():
                    if isinstance(bands, dict):
                        band_list = [f"{k}: {v}" for k, v in bands.items()]
                        band_info.append(f"{band_type}: {', '.join(band_list)}")
                if band_info:
                    chunk_parts.append("Reference Bands:\n" + '\n'.join(f"- {info}" for info in band_info))
            
            # Add pins information if available
            if 'pins' in data and isinstance(data['pins'], dict):
                pin_info = []
                for pin, details in data['pins'].items():
                    if isinstance(details, dict):
                        pin_desc = [f"{pin}:"]
                        if 'function' in details:
                            pin_desc.append(f"  Function: {details['function']}")
                        if 'connections' in details:
                            conns = details['connections']
                            if isinstance(conns, list):
                                conns = '; '.join(conns)
                            pin_desc.append(f"  Connections: {conns}")
                        pin_info.append('\n'.join(pin_desc))
                    else:
                        pin_info.append(f"{pin}: {details}")
                if pin_info:
                    chunk_parts.append("Pin Configuration:\n" + '\n'.join(f"- {p}" for p in pin_info))
            
            return "\n\n".join(chunk_parts)

        def process_component(component_data, parent_component=None):
            """Process a component and its nested items"""
            if not isinstance(component_data, dict):
                return []
                
            component_chunks = []
            component = component_data.get('component', parent_component or 'generic')
            
            # Create chunk for the component itself
            component_chunks.append(create_chunk(component, component_data))
            
            # Process nested items
            for item_type in ['issues', 'issues_by_symptom']:
                if item_type in component_data and isinstance(component_data[item_type], list):
                    for item in component_data[item_type]:
                        if isinstance(item, dict):
                            component_chunks.append(create_chunk(component, item, component_data))
            
            return component_chunks

        # Main processing loop
        chunks = []
        for item in kb_data:
            if isinstance(item, list):
                # Handle list of components
                for subitem in item:
                    if isinstance(subitem, dict):
                        chunks.extend(process_component(subitem))
            elif isinstance(item, dict):
                chunks.extend(process_component(item))

        # Remove empty chunks and deduplicate
        chunks = [c for c in chunks if c.strip()]
        unique_chunks = []
        seen = set()
        for chunk in chunks:
            # Use first 200 chars as key for deduplication
            key = chunk[:200].strip()
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        logger.info("Prepared %d unique chunks for FAISS indexing.", len(unique_chunks))

        # --- Step 4: Create embeddings and build FAISS index ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        print(f"the text chunks are {unique_chunks}")
        # Build and save FAISS index
        vectorstore = FAISS.from_texts(unique_chunks, embeddings)
        index_dir = os.path.join("domains", domain, "faiss_index")
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_dir)

        logger.info("✅ FAISS index built and saved to %s", index_dir)

    except Exception as e:
        logger.error("❌ Failed to build FAISS index for domain '%s': %s", domain, e, exc_info=True)
        raise
