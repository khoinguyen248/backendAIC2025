from flask import Flask, jsonify
from .config import Config
from .extensions import mongo, mongo2
from flask_cors import CORS
from .routes.user_routes import user_bp
from .routes.search_routes import search_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # --------------------
    # Extensions
    # --------------------
    # Cho phép tất cả domain gọi API (React, Postman, v.v.)
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

    # Init Mongo 1
    app.config["MONGO_URI"] = Config.MONGO_URI
    mongo.init_app(app)

    # Init Mongo 2
    mongo2.init_app(app, uri=Config.MONGO_URI2)

    # --------------------
    # Health check
    # --------------------
    @app.get("/health/app")
    def health_app():
        return jsonify({"ok": True, "message": "App is running"}), 200

    @app.get("/health/db1")
    def health_db1():
        try:
            mongo.db.command("ping")
            return jsonify({"ok": True, "db": "MONGO_URI"}), 200
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/health/db2")
    def health_db2():
        try:
            mongo2.db.command("ping")
            return jsonify({"ok": True, "db": "MONGO_URI2"}), 200
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    # --------------------
    # Blueprints
    # --------------------
    app.register_blueprint(user_bp, url_prefix="/user")
    app.register_blueprint(search_bp, url_prefix="/search")

    return app
