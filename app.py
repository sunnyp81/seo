from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app)

print("Loading sentence-transformers model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "all-MiniLM-L6-v2"})

@app.route('/embed', methods=['POST'])
def generate_embeddings():
    try:
        data = request.get_json()
        
        if not data or 'keywords' not in data:
            return jsonify({"error": "Missing 'keywords' in request body"}), 400
        
        keywords = data['keywords']
        
        if not isinstance(keywords, list):
            return jsonify({"error": "'keywords' must be a list"}), 400
        
        if len(keywords) == 0:
            return jsonify({"error": "keywords list is empty"}), 400
        
        if len(keywords) > 50000:
            return jsonify({"error": "Maximum 50,000 keywords allowed"}), 400
        
        print(f"Generating embeddings for {len(keywords)} keywords...")
        
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
        
        print(f"Embeddings generated successfully!")
        
        return jsonify({
            "embeddings": all_embeddings,
            "count": len(all_embeddings),
            "dimension": len(all_embeddings[0])
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Keyword Clustering API",
        "version": "1.0",
        "model": "all-MiniLM-L6-v2"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
