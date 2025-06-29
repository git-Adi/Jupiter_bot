import os
import time
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langdetect import detect
from translate import Translator as TranslateTranslator
from collections import defaultdict

load_dotenv()

class FAQBot:
    def __init__(self):
        # Initialize Pinecone with error handling
        try:
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            self.index_name = os.getenv('PINECONE_INDEX_NAME', 'jupiter')
            
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
                
            print(f"Initializing Pinecone with index: {self.index_name}")
            self.pc = Pinecone(api_key=pinecone_api_key)
            
            # Test the connection
            print("Testing Pinecone connection...")
            self.index = self.pc.Index(self.index_name)
            self.index.describe_index_stats()  # This will raise an exception if there's an issue
            print("Successfully connected to Pinecone")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {str(e)}")
            print("Pinecone will not be available. Some features may not work.")
            self.index = None
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize the language model
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_name = "microsoft/phi-2"
        
        print(f"Using device: {self.device}")
        
        # Configure quantization if CUDA is available
        bnb_config = None
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer and model with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set model to float16 for CUDA, float32 for MPS/CPU
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            # For MPS, we need to use a specific approach
            if self.device == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype
                )
                self.model = self.model.to(self.device)
            else:
                # For CUDA/CPU, use the standard approach
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto" if self.device == "cuda" else None,
                    quantization_config=bnb_config if self.device == "cuda" else None,
                    torch_dtype=torch_dtype
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            # Ensure tokenizer has a padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU with no quantization...")
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto"
            )
        
        # Initialize translator
        self.translator = TranslateTranslator(to_lang="en")
        
        # Configuration
        self.top_k = 5
        self.score_threshold = 0.6
        self.max_retries = 3
        self.retry_delay = 2

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        return self.embedding_model.encode(text).tolist()

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar vectors in Pinecone."""
        if not self.index:
            print("Pinecone index not available. Cannot perform semantic search.")
            return []
        try:
            query_embedding = self.get_embedding(query)
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            matches = []
            if hasattr(query_response, 'matches'):
                for i, match in enumerate(query_response.matches, 1):
                    match_dict = {
                        'id': match.id,
                        'score': match.score,
                        'metadata': match.metadata or {}
                    }
                    print(f"\n=== Search Result {i} ===")
                    print(f"ID: {match_dict['id']}")
                    print(f"Score: {match_dict['score']:.4f}")
                    if 'text' in match_dict['metadata']:
                        print(f"Text: {match_dict['metadata']['text'][:200]}...")
                    if 'source' in match_dict['metadata']:
                        print(f"Source: {match_dict['metadata']['source']}")
                    matches.append(match_dict)
            return matches
        except Exception as e:
            print(f"Error in semantic_search: {str(e)}")
            return []

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        try:
            return detect(text)
        except:
            return "en"

    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate text to English if it's not already in English."""
        if src_lang == "en" or not text.strip():
            return text
            
        # Mapping language codes to translator format
        lang_map = {
            'hi': 'hi',  # Hindi
            'mr': 'mr',  # Marathi
            'bn': 'bn',  # Bengali
            'te': 'te',  # Telugu
            'ta': 'ta',  # Tamil
            'gu': 'gu',  # Gujarati
            'kn': 'kn',  # Kannada
            'ml': 'ml',  # Malayalam
            'pa': 'pa'   # Punjabi
        }
        
        try:
            translator = TranslateTranslator(from_lang=lang_map.get(src_lang, 'auto'), to_lang='en')
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate text from English to target language."""
        if target_lang == "en" or not text.strip():
            return text
            
        lang_map = {
            'hi': 'hi',  # Hindi
            'mr': 'mr',  # Marathi
            'bn': 'bn',  # Bengali
            'te': 'te',  # Telugu
            'ta': 'ta',  # Tamil
            'gu': 'gu',  # Gujarati
            'kn': 'kn',  # Kannada
            'ml': 'ml',  # Malayalam
            'pa': 'pa'   # Punjabi
        }
        
        try:
            translator = TranslateTranslator(from_lang='en', to_lang=lang_map.get(target_lang, 'en'))
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def generate_response(self, query: str, context: str, language: str = "en") -> str:
        try:
            print("   Formatting prompt...")
            # Format the prompt for the Phi-2 model with instructions for detailed response
            prompt = f"""Context: {context}

Question: {query}

Please provide a detailed response of 3-5 sentences that directly answers the question using the context. Be informative and thorough in your explanation.

Answer:"""
            
            print("   Tokenizing input...")
            # Tokenize the input and ensure it's on the correct device
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,  # Increased max length for longer context
                return_attention_mask=True
            )
            
            # Move all tensors to the same device as the model
            print(f"   Moving tensors to {self.device}...")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with adjusted parameters for better quality
            print("   Generating text...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Increased for longer responses
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,  # Slightly higher for more diverse responses
                    top_k=50,    # Limit to top-k tokens
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    length_penalty=1.2,  # Encourage longer responses
                    repetition_penalty=1.1  # Reduce repetition
                )
            
            print("   Decoding response...")
            # Decode the full response
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up the response while preserving multiple sentences
            # First, split into sentences and keep only complete ones
            sentences = [s.strip() for s in response.split('. ') if s.strip()]
            if sentences:
                # Join sentences with periods and add a final period
                response = '. '.join(sentences) + ('' if response.endswith('.') else '.')
            
            # Translate back to original language if needed
            if language != "en":
                print("   Translating response...")
                response = self.translate_from_english(response, language)
                
            return response
            
        except Exception as e:
            import traceback
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())
            # Return a fallback response with context if available
            if context:
                return f"Here's what I found: {context[:500]}{'...' if len(context) > 500 else ''}"
            return "I'm sorry, I encountered an error while generating a response. Please try again later."

    def get_related_queries(self, query: str, top_k: int = 5) -> List[str]:
        """Get related queries based on semantic similarity."""
        try:
            print(f"Getting related queries for: {query}")
            # Get similar queries from the knowledge base
            results = self.semantic_search(query, top_k=top_k)
            related = []
            
            for result in results:
                # Handle both dictionary and object-style access
                metadata = result.get('metadata', {}) if isinstance(result, dict) else getattr(result, 'metadata', {})
                if isinstance(metadata, dict) and 'related_queries' in metadata:
                    if isinstance(metadata['related_queries'], list):
                        related.extend(metadata['related_queries'])
                    elif isinstance(metadata['related_queries'], str):
                        # If it's a string, try to parse it as JSON
                        try:
                            parsed = json.loads(metadata['related_queries'])
                            if isinstance(parsed, list):
                                related.extend(parsed)
                        except json.JSONDecodeError:
                            print(f"Could not parse related_queries: {metadata['related_queries']}")
            
            # Ensure we return a list of strings
            related = [str(q).strip() for q in related if q and str(q).strip()]
            # Remove duplicates and limit to 2
            unique_related = list(dict.fromkeys(related))[:2]
            print(f"Found related queries: {unique_related}")
            return unique_related
            
        except Exception as e:
            print(f"Error getting related queries: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            print(f"\n=== Starting query processing ===")
            print(f"Query: {query}")
            
            # Language detection
            print("\n1. Detecting language...")
            language = self.detect_language(query)
            print(f"   Detected language: {language}")
            
            # Translation if needed
            if language != "en":
                print(f"\n2. Translating to English...")
                query_en = self.translate_to_english(query, language)
                print(f"   Translated query: {query_en}")
            else:
                query_en = query
                
            # Semantic search
            print(f"\n3. Performing semantic search...")
            results = self.semantic_search(query_en, self.top_k)
            print(f"   Found {len(results)} matches")
            
            # Filter results
            print(f"\n4. Filtering results (threshold: {self.score_threshold})...")
            
            # Helper function to safely get score from either dict or object
            def get_score(match):
                if isinstance(match, dict):
                    return match.get('score', 0)
                return getattr(match, 'score', 0)
            
            # Filter results using the helper function
            filtered_results = [r for r in results if get_score(r) >= self.score_threshold]
            
            if not filtered_results and results:
                first_score = get_score(results[0])
                print(f"   No results above threshold, using top result (score: {first_score:.3f})")
                filtered_results = [results[0]]
            else:
                print(f"   Found {len(filtered_results)} results above threshold")
                
            # Prepare context
            context = ""
            source = None
            if filtered_results:
                best_match = max(filtered_results, key=get_score)
                
                # Safely get metadata from either dict or object
                if isinstance(best_match, dict):
                    metadata = best_match.get('metadata', {})
                    context = metadata.get('text', '')
                    source = metadata.get('source', 'Unknown source')
                    score = best_match.get('score', 0)
                else:
                    metadata = getattr(best_match, 'metadata', {}) or {}
                    context = getattr(metadata, 'text', '')
                    source = getattr(metadata, 'source', 'Unknown source')
                    score = getattr(best_match, 'score', 0)
                
                print(f"   Best match score: {score:.3f}")
                print(f"   Context length: {len(context)} characters")
                print(f"   Source: {source}")
                print(f"   Metadata type: {type(metadata).__name__}")
                if hasattr(metadata, '__dict__'):
                    print(f"   Metadata attributes: {vars(metadata).keys()}")
                elif isinstance(metadata, dict):
                    print(f"   Metadata keys: {list(metadata.keys())}")
            else:
                print("   No valid context found")
            
            # Generate response
            print(f"\n5. Generating response...")
            answer = self.generate_response(query_en, context, language)
            print(f"   Generated response: {answer[:150]}...")
            
            # Get related queries (limit to 2)
            print(f"\n6. Finding related queries...")
            related_queries = []
            if context:
                related_queries = self.get_related_queries(query_en)[:2]  # Limit to 2 related queries
                print(f"   Found {len(related_queries)} related queries: {related_queries}")
            else:
                print("   No context available for related queries")
            
            # Extract source URL from the best match
            source_url = None
            if filtered_results:
                best_match = max(filtered_results, key=get_score)
                if isinstance(best_match, dict):
                    source_url = best_match.get('metadata', {}).get('source')
                else:
                    source_url = getattr(getattr(best_match, 'metadata', {}), 'source', None)
            
            # Prepare search results for debug info
            search_results_debug = []
            for i, result in enumerate(filtered_results, 1):
                if isinstance(result, dict):
                    search_results_debug.append({
                        'rank': i,
                        'id': result.get('id', 'N/A'),
                        'score': result.get('score', 0),
                        'text': (result.get('metadata', {}).get('text', '')[:200] + '...') if result.get('metadata', {}).get('text') else 'N/A',
                        'source': result.get('metadata', {}).get('source', 'N/A')
                    })
            
            # Prepare the response in the desired format
            response = {
                "answer": answer,
                "source": source_url if source_url else source,  # Prefer source_url if available
                "related_queries": related_queries[:2],  # Ensure only 2 related queries
                "debug": {
                    "search_results": search_results_debug,
                    "top_search_result": {
                        "id": best_match.get('id', 'N/A') if filtered_results else None,
                        "score": best_match.get('score', 0) if filtered_results else None,
                        "context": context[:500] + ('...' if len(context) > 500 else '') if context else "",
                        "source": source,
                        "source_url": source_url if source_url else None
                    } if filtered_results else None,
                    "query_language": language,
                    "query_translated": query_en if language != "en" else "N/A (already in English)",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            print("\n=== Query processing complete ===\n")
            print("=== DEBUG: Top Search Result ===")
            print(f"Score: {response['debug']['top_search_result']['score'] if response['debug']['top_search_result'] else 'N/A'}")
            print(f"Context: {response['debug']['top_search_result']['context'] if response['debug']['top_search_result'] else 'N/A'}")
            print(f"Source: {response['debug']['top_search_result']['source'] if response['debug']['top_search_result'] else 'N/A'}")
            print("================================\n")
            
            return response
            
        except Exception as e:
            print(f"Error in process_query: {e}")
            return {
                "answer": "I'm having trouble processing your request. Please try again later.",
                "source": None,
                "related_queries": []
            }

# Initialize the Flask app
app = Flask(__name__)
bot = FAQBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    try:
        if not request.is_json:
            return jsonify({
                'answer': 'Invalid request format. Please send JSON data.',
                'source': '',
                'related_queries': [],
                'response_time': 0
            }), 400
            
        data = request.get_json()
        if 'query' not in data or not data['query'].strip():
            return jsonify({
                'answer': 'Please provide a valid query.',
                'source': '',
                'related_queries': [],
                'response_time': 0
            }), 400
            
        print(f"Processing query: {data['query']}")
        response = bot.process_query(data['query'])
        print(f"Generated response: {response['answer'][:100]}...")
        
        # Calculate response time in milliseconds
        response_time = (time.time() - start_time) * 1000
        
        # Ensure response has all required fields
        if not isinstance(response, dict):
            response = {
                'answer': str(response) if response else 'No response generated',
                'source': '',
                'related_queries': []
            }
        
        # Add response time to the response
        response['response_time'] = response_time
        
        # Ensure related_queries exists and is a list
        if 'related_queries' not in response or not isinstance(response['related_queries'], list):
            response['related_queries'] = []
            
        # Ensure source exists
        if 'source' not in response:
            response['source'] = ''
            
        # Ensure answer exists
        if 'answer' not in response or not response['answer']:
            response['answer'] = 'I apologize, but I couldn\'t generate a response for that query.'
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in /query endpoint: {str(e)}")
        return jsonify({
            'answer': 'Sorry, I encountered an error processing your request. Please try again later.',
            'source': '',
            'related_queries': [],
            'response_time': (time.time() - start_time) * 1000
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8980)