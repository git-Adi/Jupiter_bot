import os
import json
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

# Initializing the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

index_name = os.getenv('PINECONE_INDEX_NAME', 'jupiter')

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connecting to the index
index = pc.Index(index_name)

def process_final_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for i, item in enumerate(tqdm(data, desc="Processing final_content.json")):
        if 'summary' in item and 'title' in item['summary']:
            text = f"{item['summary']['title']}. {item['summary']['summary']} "
            text += ". ".join(item['summary'].get('key_points', []))
            
            # Generating embedding
            embedding = model.encode(text).tolist()
            
            # Creating record
            record = {
                'id': f"fc_{i}",
                'values': embedding,
                'metadata': {
                    'source': 'final_content.json',
                    'url': item.get('url', ''),
                    'title': item['summary']['title'],
                    'content': text,
                    'type': 'community_post'
                }
            }
            records.append(record)
    return records

def process_final_content1(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    doc_counter = 0
    
    for doc_idx, doc in enumerate(tqdm(data, desc="Processing final_content1.json")):
        url = doc.get('url', '')
        title = doc.get('title', '')
        
        
        for section in doc.get('sections', []):
            heading = section.get('heading', '')
            content = " ".join(section.get('content', []))
            
            if not content.strip():
                continue
                
            # Template
            text = f"{title}. {heading}. {content}"
            
            # Generating embedding
            embedding = model.encode(text).tolist()
            
            # Creating record
            record = {
                'id': f"fc1_{doc_idx}_{doc_counter}",
                'values': embedding,
                'metadata': {
                    'source': 'final_content1.json',
                    'url': url,
                    'title': f"{title} - {heading}" if heading else title,
                    'content': text,
                    'type': 'documentation'
                }
            }
            records.append(record)
            doc_counter += 1
    
    return records

def upsert_to_pinecone(records, batch_size=100):
    for i in tqdm(range(0, len(records), batch_size), desc="Upserting to Pinecone"):
        batch = records[i:i + batch_size]
        # Converting records to Pinecone upsert format
        vectors = []
        for r in batch:
            vector = {
                'id': r['id'],
                'values': r['values'],
                'metadata': r['metadata']
            }
            vectors.append(vector)
        index.upsert(vectors=vectors)

def main():
    if not os.getenv('PINECONE_API_KEY'):
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Processing both files
    final_content_records = process_final_content('final_content.json')
    final_content1_records = process_final_content1('final_content1.json')
    
    # Combining all records
    all_records = final_content_records + final_content1_records
    
    print(f"Total records to upsert: {len(all_records)}")
    
    # Upserting to Pinecone
    upsert_to_pinecone(all_records)
    print("Upsert completed successfully!")

if __name__ == "__main__":
    main()
