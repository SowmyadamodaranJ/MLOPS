import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify
from flask_cors import CORS
from src.api.routes import api_bp
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_app():
    """
    Application factory for the Predictive Maintenance REST API.
    """
    app = Flask(__name__)
    
    # Configure CORS for the React Frontend (Phase 6)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register API blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "predictive-maintenance-api",
            "version": "1.0.0"
        })
        
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not Found", "message": str(error)}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500

    logger.info("Flask API successfully initialized.")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
