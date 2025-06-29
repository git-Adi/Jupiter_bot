import os
import json
from http import HTTPStatus
from flask import Flask, request, jsonify
from faq_bot import FAQBot
import sys
from pathlib import Path

# Add the project directory to the path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize the FAQ bot
bot = FAQBot()

def handler(event, context):
    from flask import Request
    from werkzeug.wrappers import Response
    from werkzeug.serving import run_simple
    
    # Handle API Gateway event
    if event.get('httpMethod'):
        # API Gateway event
        path = event['path']
        method = event['httpMethod']
        headers = event.get('headers', {})
        body = event.get('body', '{}')
        query_params = event.get('queryStringParameters', {}) or {}
        
        # Convert to WSGI environ
        environ = {
            'REQUEST_METHOD': method,
            'PATH_INFO': path,
            'QUERY_STRING': '&'.join([f"{k}={v}" for k, v in query_params.items()]),
            'SERVER_NAME': 'localhost',
            'SERVER_PORT': '80',
            'SERVER_PROTOCOL': 'HTTP/1.1',
            'wsgi.url_scheme': 'http',
            'wsgi.input': None,
            'wsgi.errors': sys.stderr,
            'wsgi.version': (1, 0),
            'wsgi.multithread': False,
            'wsgi.multiprocess': False,
            'wsgi.run_once': False,
        }
        
        # Add headers
        for key, value in headers.items():
            key = key.upper().replace('-', '_')
            if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                key = f'HTTP_{key}'
            environ[key] = value
        
        # Handle request body
        if body:
            if isinstance(body, dict):
                body = json.dumps(body)
            environ['wsgi.input'] = type('', (), {
                'read': lambda: body.encode('utf-8')
            })()
            environ['CONTENT_LENGTH'] = str(len(body))
        
        # Create response
        response = {}
        
        def start_response(status, response_headers, exc_info=None):
            nonlocal response
            response['statusCode'] = int(status.split(' ')[0])
            response['headers'] = dict(response_headers)
            response['body'] = ''
            return lambda data: None
        
        # Handle the request
        from io import StringIO
        from werkzeug.wrappers import Response
        
        with app.request_context(environ):
            try:
                if path == '/api/ask':
                    return handle_ask()
                elif path == '/':
                    return handle_index()
                else:
                    return jsonify({
                        'error': 'Not Found',
                        'message': 'The requested resource was not found.'
                    }), HTTPStatus.NOT_FOUND
            except Exception as e:
                print(f"Error handling request: {e}")
                return jsonify({
                    'error': 'Internal Server Error',
                    'message': 'An error occurred while processing your request.'
                }), HTTPStatus.INTERNAL_SERVER_ERROR
    else:
        # Direct function call (for local testing)
        return handle_http_event(event)

@app.route('/')
def handle_index():
    return jsonify({
        'status': 'ok',
        'message': 'FAQ Bot API is running',
        'endpoints': {
            'POST /api/ask': 'Ask a question',
            'GET /': 'Health check'
        }
    })

@app.route('/api/ask', methods=['POST'])
def handle_ask():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Query parameter is required.'
            }), HTTPStatus.BAD_REQUEST
        
        # Process the query
        result = bot.process_query(query)
        
        return jsonify({
            'answer': result['answer'],
            'source': result.get('source'),
            'related_queries': result.get('related_queries', [])
        })
        
    except Exception as e:
        print(f"Error in /api/ask: {e}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to process your question.'
        }), HTTPStatus.INTERNAL_SERVER_ERROR

# For local testing
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
