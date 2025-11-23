"""
Web-based chat interface for TNG Digital RAG System
Simple Flask application with a modern chat UI
"""

from flask import Flask, render_template, request, jsonify
from rag_system import ask_tngd_bot

app = Flask(__name__)


@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Please enter a message'
            }), 400
        
        # Get response from RAG system
        result = ask_tngd_bot(user_message)
        
        # Format response
        response = {
            'success': True,
            'answer': result.get('final_answer', 'I apologize, but I could not generate a response.'),
            'sources': result.get('retrieved_chunks', []),
            'blocked': result.get('blocked', False)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("TNG Digital RAG Web Chat Interface")
    print("=" * 70)
    print("\nStarting web server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

