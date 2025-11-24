"""
Web-based chat interface for TNG Digital RAG System
Simple Flask application with a modern chat UI
"""

from flask import Flask, render_template, request, jsonify, session
from rag_system import ask_tngd_bot
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Required for session management


@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with conversation history"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Please enter a message'
            }), 400
        
        # Initialize conversation history in session if not present
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        
        # Get conversation history from session
        conversation_history = session.get('conversation_history', [])
        
        # Get response from RAG system with conversation history
        result = ask_tngd_bot(user_message, conversation_history=conversation_history)
        
        # Update conversation history in session
        # Add user message
        conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        # Add assistant response
        conversation_history.append({
            'role': 'assistant',
            'content': result.get('final_answer', 'I apologize, but I could not generate a response.')
        })
        
        # Keep only last 10 exchanges (20 messages) to prevent session bloat
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        # Save updated history back to session (reassign to trigger Flask's change detection)
        session['conversation_history'] = conversation_history
        session.modified = True  # Explicitly mark session as modified
        
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

