import os
import re
import time
import json
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import traceback
from translate import Translator as TranslateTranslator
from langdetect import detect
from collections import defaultdict

load_dotenv()

class FAQBot:
    def __init__(self):
        # Initialize query history
        self.query_history = []
        self.max_history = 10  # Keep last 10 queries in history
        
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
        
        # Model configuration
        self.model_name = "microsoft/phi-2"
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer and model with error handling
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print("Loading model (this may take a moment, model is ~2.3GB)...")
            
            # Load model with bfloat16 precision for better stability
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa"  # Use SDPA attention for better performance
            )
            
            # Set up the generation pipeline with more conservative parameters
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                max_new_tokens=200,  # More conservative token limit
                do_sample=True,
                temperature=0.3,     # Lower temperature for more focused outputs
                top_p=0.9,           # Use nucleus sampling
                top_k=40,            # Limit to top 40 tokens
                repetition_penalty=1.15,  # Slightly higher to prevent repetition
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=4,   # Slightly larger n-gram penalty
                clean_up_tokenization_spaces=True
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'
            
            # Get the actual device being used by the model
            device = next(self.model.parameters()).device
            print(f"Successfully loaded {self.model_name} on {device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
            raise RuntimeError(
                "Failed to load the model. Please ensure:\n"
                "1. You have accepted the model's terms of use at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct\n"
                "2. Your HUGGING_FACE_HUB_TOKEN is correctly set in your .env file\n"
                "3. You have a stable internet connection\n"
                "4. You have enough disk space (model is ~6GB)"
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
        if not text.strip() or src_lang == 'en':
            return text
            
        try:
            print(f"   Translating from {src_lang} to en...")
            translator = TranslateTranslator(from_lang=src_lang, to_lang="en")
            return translator.translate(text)
            
        except Exception as e:
            print(f"Translation error ({src_lang}->en): {str(e)}")
            return text
            
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate text from English to target language."""
        if not text.strip() or target_lang == 'en':
            return text
            
        try:
            print(f"   Translating from en to {target_lang}...")
            translator = TranslateTranslator(from_lang="en", to_lang=target_lang)
            return translator.translate(text)
            
        except Exception as e:
            print(f"Translation error (en->{target_lang}): {str(e)}")
            return text

    def generate_response(self, query: str, context: str, language: str = "en", search_results: List[Dict] = None) -> str:
        try:
            print("   Formatting prompt with semantic search results as context...")
            
            # Process search results to create a structured context
            context_parts = []
            
            if search_results:
                # Add relevant information from top search results with their scores
                for i, result in enumerate(search_results[:3], 1):  # Use top 3 most relevant results
                    if isinstance(result, dict):
                        metadata = result.get('metadata', {})
                        text = metadata.get('text', '').strip()
                        title = metadata.get('title', '')
                        score = result.get('score', 0)
                        
                        # Only include results with sufficient relevance
                        if text and score > self.score_threshold:
                            # Clean and format the text
                            text = ' '.join(text.split())  # Normalize whitespace
                            context_parts.append(
                                f"--- RESULT {i} (Relevance: {score:.2f}) ---\n"
                                f"Title: {title}\n"
                                f"Content: {text}\n"
                            )
            
            # Add the original context if it's not already included
            if context and context.strip() not in '\n'.join(context_parts):
                context_parts.insert(0, f"--- QUERY CONTEXT ---\n{context}\n")
            
            # Combine all context parts
            full_context = '\n'.join(context_parts).strip()
            
            if not full_context:
                full_context = "No specific context available for this query."
            
            # Format the prompt for Llama 3
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for Jupiter. Provide accurate and concise answers based on the given context."},
                {"role": "user", "content": f"""Use the following context to answer the question. If the answer isn't in the context, say so.
                
Context:
{full_context}

Question: {query}

Answer concisely (2-4 sentences). If multiple perspectives exist, mention them briefly."""}
            ]
            
            # Format prompt for Phi-2
            prompt = f"""### Instruction:
            Answer the following question based on the provided context.
            
            ### Context:
            {context}
            
            ### Question:
            {query}
            
            ### Response:
            """
            
            print("   Generating response with Phi-2...")
            
            # Generate response using the pipeline with stable parameters
            response = self.generator(
                prompt,
                max_new_tokens=200,  # Match the pipeline settings
                temperature=0.3,     # Match the pipeline settings
                top_p=0.9,           # Match the pipeline settings
                top_k=40,            # Match the pipeline settings
                do_sample=True,
                num_return_sequences=1,
                repetition_penalty=1.15,  # Match the pipeline settings
                no_repeat_ngram_size=4,   # Match the pipeline settings
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                clean_up_tokenization_spaces=True
            )
            
            # Extract the response text
            response_text = response[0]['generated_text']
            
            # Remove the input prompt from the response
            if prompt in response_text:
                response_text = response_text.replace(prompt, "").strip()
            
            # Clean up the response
            response_text = response_text.split('</s>')[0].strip()  # Remove any trailing tokens
            response_text = re.sub(r'\s+', ' ', response_text)  # Normalize whitespace
            
            # Ensure the response ends with proper punctuation
            if response_text and not response_text.endswith(('.', '!', '?')):
                response_text += '.'
            
            # Translate back to original language if needed
            if language != "en":
                print("   Translating response...")
                response_text = self.translate_from_english(response_text, language)
            
            return response_text
            
        except Exception as e:
            import traceback
            print(f"Error in generate_response: {str(e)}")
            print(traceback.format_exc())
            # Return a fallback response
            if context:
                return f"Here's what I found: {context[:500]}{'...' if len(context) > 500 else ''}"
            return "I'm sorry, I encountered an error while generating a response. Please try again later."

    def generate_related_queries(self, query: str, context: str, num_queries: int = 2) -> List[str]:
        """Generate related queries using the LLM based on the original query and context."""
        try:
            print(f"Generating {num_queries} related queries for: {query}")
            
            # Create a prompt for generating related queries
            prompt = f"""Given the following question and context, generate {num_queries} related questions that someone might also want to ask. 
            Make sure the questions are relevant and specific to the context.
            
            Original Question: {query}
            
            Context: {context}
            
            Generate exactly {num_queries} related questions, one per line:"""
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate related queries
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,  # Enough for several questions
                    temperature=0.8,     # Slightly higher for more diverse questions
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse the response to extract questions
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            questions = [q for q in questions if q and q[0].isdigit() or q[0] == '-']  # Filter out non-question lines
            
            # Clean up the questions
            questions = [re.sub(r'^\d+[\.\)\-]?\s*', '', q).strip() for q in questions]
            questions = [q for q in questions if q and q[-1] == '?']  # Keep only questions
            
            # Limit to the requested number of questions
            questions = questions[:num_queries]
            
            # If we didn't get enough questions, generate more
            if len(questions) < num_queries:
                remaining = num_queries - len(questions)
                additional = self.generate_related_queries(query, context, remaining)
                questions.extend(additional)
                questions = questions[:num_queries]  # Ensure we don't exceed the limit
                
            print(f"Generated related queries: {questions}")
            return questions
            
        except Exception as e:
            print(f"Error in generate_related_queries: {str(e)}")
            return []
            
    def get_related_queries(self, query: str, context: str = "", num_queries: int = 2) -> List[str]:
        """Get previous queries as related queries."""
        try:
            print("Getting previous queries as related queries...")
            
            # Get previous queries (excluding current one)
            previous_queries = [q for q in self.query_history if q.lower() != query.lower()]
            
            # Return up to num_queries previous queries (most recent first)
            related = previous_queries[-(num_queries):]
            
            # If we don't have enough previous queries, just return what we have
            print(f"   Returning previous queries as related: {related}")
            return related
            
        except Exception as e:
            print(f"Error in get_related_queries: {str(e)}")
            return []
    
    def get_score(self, match):
        """Helper method to get score from a match (dict or object)."""
        if isinstance(match, dict):
            return match.get('score', 0)
        return getattr(match, 'score', 0)
    
    def process_query(self, query: str, language: str = None) -> Dict[str, Any]:
        """Process a user query and return a response with answer, source, and related queries."""
        start_time = time.time()
        try:
            print(f"\n=== Starting query processing ===")
            print(f"Query: {query}")
            
            # Add current query to history (before processing)
            self.query_history.append(query)
            # Keep only the most recent queries
            self.query_history = self.query_history[-self.max_history:]
            
            # Language detection
            print("\n1. Detecting language...")
            if language is None:
                language = self.detect_language(query)
            print(f"   Detected language: {language}")
            
            # Translation if needed
            if language != "en":
                print("\n2. Translating to English...")
                query_en = self.translate_to_english(query, language)
                print(f"   Translated query: {query_en}")
            else:
                query_en = query
                
            # Semantic search
            print("\n3. Performing semantic search...")
            results = self.semantic_search(query_en, top_k=self.top_k)
            print(f"   Found {len(results)} matches")
            
            # Filter results
            print(f"\n4. Filtering results (threshold: {self.score_threshold})...")
            filtered_results = [r for r in results if self.get_score(r) >= self.score_threshold]
            
            if not filtered_results and results:
                first_score = self.get_score(results[0])
                print(f"   No results above threshold, using top result (score: {first_score:.3f})")
                filtered_results = [results[0]]
            else:
                print(f"   Found {len(filtered_results)} results above threshold")
            
            # Prepare context from all relevant search results
            context_parts = []
            sources = []
            scores = []
            best_match = None
            
            if filtered_results:
                print(f"   Processing {len(filtered_results)} filtered results...")
                # Sort results by score in descending order
                filtered_results.sort(key=self.get_score, reverse=True)
                best_match = filtered_results[0]  # Keep track of best match for source
                
                # Use all filtered results for context
                for i, result in enumerate(filtered_results, 1):
                    try:
                        if isinstance(result, dict):
                            metadata = result.get('metadata', {})
                            text = metadata.get('text', '')
                            if not text:
                                text = metadata.get('content', '')  # Try alternative field names
                            text = str(text).strip()  # Ensure text is a string and strip whitespace
                            title = str(metadata.get('title', '')).strip()
                            score = float(result.get('score', 0))
                            
                            # Debug log each result
                            print(f"   - Result {i}: score={score:.3f}, has_text={bool(text)}, has_title={bool(title)}")
                            
                            # Include result even if text is empty, but prefer results with text
                            context_text = f"Title: {title}\nContent: {text}" if title else text
                            if context_text.strip():
                                context_parts.append(
                                    f"--- Result {i} (Relevance: {score:.2f}) ---\n"
                                    f"{context_text}\n"
                                )
                                sources.append(metadata.get('source'))
                                scores.append(score)
                                print(f"   - Added result {i} to context")
                            else:
                                print(f"   - Skipping empty result {i}")
                    except Exception as e:
                        print(f"   - Error processing result {i}: {str(e)}")
                        continue
                
                # Combine all context parts
                context = '\n'.join(context_parts) if context_parts else None
                
                # Get the best source (from highest scoring result with a source)
                source = next((s for s in sources if s), None)
                
                print(f"   Using context from {len(context_parts)} relevant results (best score: {max(scores) if scores else 0:.3f})")
                if not context_parts:
                    print("   Warning: No valid context was extracted from the search results")
            else:
                print("   No filtered results available, will attempt to generate response without specific context")
                context = f"User asked: {query_en}"  # Fallback context
                source = None
                
            # Fallback if no context was extracted but we have results
            if not context and filtered_results:
                print("   No valid context extracted, using raw results as fallback")
                context = "\n".join([
                    f"--- Result {i} ---\n{json.dumps(r, indent=2, default=str)}" 
                    for i, r in enumerate(filtered_results[:3], 1)
                ])
            
            # Generate response with search results for better context
            print("\n5. Generating response...")
            answer = self.generate_response(
                query=query_en, 
                context=context, 
                language=language,
                search_results=filtered_results  # Pass all filtered results for context
            )
            print(f"   Generated response: {answer[:150]}...")
            
            # Get related queries using the full context from search results
            print("\n6. Getting related queries...")
            related_queries = self.get_related_queries(query_en, context, num_queries=2)
            print(f"   Found {len(related_queries)} related queries")
            
            # Prepare the response with additional metadata
            response = {
                "answer": answer,
                "source": source,
                "related_queries": related_queries,
                "response_time": (time.time() - start_time) * 1000,  # in milliseconds
                "search_results_count": len(filtered_results),
                "best_match_score": self.get_score(best_match) if best_match else None
            }
            
            print("\n=== Query processing complete ===\n")
            return response
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "answer": "I'm having trouble processing your request. Please try again later.",
                "source": None,
                "related_queries": [],
                "response_time": (time.time() - start_time) * 1000
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