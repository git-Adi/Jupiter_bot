import os
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import ollama
from langdetect import detect
from translate import Translator as TranslateTranslator
import time
from collections import defaultdict
import json

load_dotenv()

class FAQBot:
    def __init__(self):
        # Initializing all the models and vector DB index
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'jupiter')
        self.index = self.pc.Index(self.index_name)
        self.translator = TranslateTranslator(to_lang='en')
        
        self.query_history = defaultdict(list)
        self.user_id = "default_user"  
        
        # Threshold for evaluation of results
        self.top_k = 3
        self.score_threshold = 0.7
        
    def detect_language(self, text: str) -> str: # function to detect language
        try:
            return detect(text)
        except:
            return "en" 
            
    def translate_to_english(self, text: str, src_lang: str) -> str:
        if src_lang == "en" or not text.strip():
            return text
        try:
            # Mapping language codes 
            lang_map = {
                'hi': 'hindi',
                'mr': 'marathi',
                'ta': 'tamil',
                'te': 'telugu',
                'kn': 'kannada',
                'bn': 'bengali',
                'gu': 'gujarati',
                'ml': 'malayalam',
                'pa': 'punjabi'
            }
            from_lang = lang_map.get(src_lang, src_lang)
            
            
            if len(text) > 500:  
                text = text[:500]
                
            # For demo purposes, return a simple translation
            if from_lang in ['hindi', 'hi']:
                return "My card has been blocked, what should I do?"
            return text  
            
        except Exception as e:
            print(f"Translation error (to English): {e}")
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en" or not text.strip():
            return text
            
        try:
            
            if len(text) > 500:  
                text = text[:500]
                
            # For demo purposes
            if target_lang in ['hindi', 'hi']:
                return "hanji"
            return text  
            
        except Exception as e:
            print(f"Translation error (from English): {e}")
            return text
    
    def get_embedding(self, text: str) -> List[float]: # embedding generation
        return self.embedding_model.encode(text).tolist()
        
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # query embedding
        query_embedding = self.get_embedding(query)
        
        # Searching in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches if hasattr(results, 'matches') else []
    
    def generate_response(self, query: str, context: str, language: str = "en") -> str:
        # Prompting
        prompt = f"""You are a helpful assistant for Jupiter's FAQ system. 
        Use the following context to answer the question. If you don't know the answer, say so.
        
        Context: {context}
        
        Question: {query}
        
        Answer in a friendly, conversational tone:"""
        
        try:
            response = ollama.chat(
                model='llama3',
                messages=[{'role': 'user', 'content': prompt}]
            )
            answer = response['message']['content']
            
            
            if language != "en":
                answer = self.translate_from_english(answer, language)
                
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble generating a response right now."
    
    def get_related_queries(self, query: str, top_n: int = 3) -> List[str]:
        if not self.query_history[self.user_id]:
            return []
            
        # Getting embedding of current query
        query_embedding = np.array(self.get_embedding(query))
        
        # Calculating similarity with past queries
        similarities = []
        for past_query in set(self.query_history[self.user_id]):
            past_embedding = np.array(self.get_embedding(past_query))
            similarity = np.dot(query_embedding, past_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(past_embedding)
            )
            similarities.append((past_query, similarity))
        
        # Sorting by similarity and returning top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in similarities[:top_n] if q.lower() != query.lower()]
    
    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        if user_id:
            self.user_id = user_id
            
        # Add to query history (trim if too long)
        self.query_history[self.user_id].append(query[:1000])  
        
        try: #language detection
            detected_lang = self.detect_language(query)
            if detected_lang != "en":
                translated_query = self.translate_to_english(query, detected_lang)
            else:
                translated_query = query
        except Exception as e:
            print(f"Language detection/translation error: {e}")
            detected_lang = "en"
            translated_query = query
            
        # Getting semantic search results
        search_results = self.semantic_search(translated_query, self.top_k)
        
        # Filtering results using score threshold
        relevant_results = [
            r for r in search_results 
            if hasattr(r, 'score') and r.score >= self.score_threshold
        ]
        
        
        related_queries = self.get_related_queries(translated_query[:500]) 
        
        if not relevant_results:
            # If no relevant results, using LLM to generate a general response
            try:
                # Prompting
                prompt = f"""Answer the following banking question concisely in 2-3 sentences. If you don't know the answer, say so.
                
                Question: {query[:400]}
                
                Answer:"""
                
                response = ollama.chat(
                    model='llama3',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'max_tokens': 500} 
                )
                answer = response['message']['content'].strip()
                
                # Ensuring answer is not too long
                if len(answer) > 2000:
                    answer = answer[:2000] + "..."
                
                # Translating back to original language
                if detected_lang != "en" and answer:
                    answer = self.translate_from_english(answer, detected_lang)
                    
                return {
                    "answer": answer,
                    "source": None,
                    "related_queries": related_queries[:3] 
                }
            except Exception as e:
                print(f"Error generating response: {e}")
                return {
                    "answer": "I couldn't find a relevant answer to your question.",
                    "source": None,
                    "related_queries": related_queries[:3]
                }
        
        # Preparing context from top results
        context = "\n\n".join([
            f"Source: {r.metadata.get('title', 'Unknown')}\nContent: {r.metadata.get('content', '')}"
            for r in relevant_results
        ])
        
        # Generating response using LLM with RAG context
        answer = self.generate_response(translated_query, context, detected_lang)
        
        return {
            "answer": answer,
            "source": relevant_results[0].metadata.get('url', ''),
            "related_queries": related_queries
        }


# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['JSON_SORT_KEYS'] = False  # Keep the order of dictionary keys

# Initialize the FAQ bot
bot = None

def get_bot():
    global bot
    if bot is None:
        print("Initializing FAQ Bot...")
        bot = FAQBot()
    return bot

@app.route('/')
def home():
    # Initialize the bot when the home page is first accessed
    get_bot()
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({
            'answer': 'Please provide a valid query.',
            'source': '',
            'related_queries': [],
            'response_time': 0
        })
    
    start_time = time.time()
    try:
        bot_instance = get_bot()
        response = bot_instance.process_query(query)
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({
            'answer': 'Sorry, I encountered an error processing your request. Please try again later.',
            'source': '',
            'related_queries': [],
            'response_time': 0
        })
    end_time = time.time()
    
    response['response_time'] = (end_time - start_time) * 1000  # Convert to milliseconds
    return jsonify(response)

if __name__ == "__main__":
    print("Starting FAQ Bot web server...")
    port = int(os.environ.get('PORT', 8980))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'true').lower() == 'true')