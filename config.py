import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")
MODEL_CANDIDATE_PATHS = [
	os.path.join(BASE_DIR, "models", "best_model.keras"),
	os.path.join(BASE_DIR, "models", "cnn_model.h5"),
	os.path.join(BASE_DIR, "model", "pharmaguard_model.h5"),
]

# Upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Database
DATABASE = os.path.join(BASE_DIR, "database", "pharmaguard.db")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Max upload size: 16 MB
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
