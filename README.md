# PHARMAGUARD - AI-Powered Medicine Authentication System

PharmaGuard is a robust, web-based application designed to combat the spread of counterfeit pharmaceuticals. It utilizes a 3-layer verification pipeline combining Convolutional Neural Networks (CNN), EasyOCR, and Multi-Pass Fuzzy Matching to verify the authenticity of medicine packaging in seconds.

## 🚀 Key Features

*   **3-Layer Verification Architecture (A + T + C = V):**
    *   **A(I) Authenticity Layer:** A transfer-learning CNN (MobileNetV2/ResNet50V2-based) analyzes visual characteristics of medicine packaging.
    *   **T(I) Text & Expiry Layer:** EasyOCR reads the text, applying advanced preprocessing (adaptive thresholding, downscaling) and robust expiry date extraction.
    *   **C(I) Cross-Reference Layer:** RapidFuzz performs multi-pass fuzzy matching against 10,000+ real medicines and manufacturer databases.
    *   **V (Verdict):** A sophisticated **7-Pass Verdict Logic** aggregates the results to return statuses like `100% GENUINE`, `GENUINE (UNVERIFIED MANUFACTURER)`, `SUSPICIOUS`, or `FAKE`.
*   **Interactive Dashboard:** Provides a detailed history of all scans and visualizes detection statistics (weekly, monthly, yearly trends) using interactive charts.
*   **Dataset Conversion Pipeline:** Includes a seamless `coco_to_classification.py` utility to instantly convert COCO-format medical datasets into structured training folders.
*   **PDF Reporting:** Instantly generate and download comprehensive PDF reports for each detection scan.

## 🛠️ Technology Stack

*   **Backend:** Python, Flask
*   **Machine Learning / AI:** TensorFlow, Keras, OpenCV, NumPy
*   **OCR:** EasyOCR (Deep Learning-based text extraction)
*   **Matching Algorithms:** RapidFuzz (Multi-pass Token Sort & Set Ratio matching), Pandas
*   **Database:** SQLite3
*   **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript, Bootstrap 5, Chart.js

## 📋 Prerequisites

Before running the application, ensure you have the following installed:
1.  **Python 3.8+**
2.  **Virtual Environment (Recommended)**

*(Note: Tesseract is no longer required as PharmaGuard v2 has migrated entirely to EasyOCR).*

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/PHARMAGUARD.git
   cd PHARMAGUARD
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the Database:**
   The SQLite database (`database.db`) will initialize automatically upon the first run of the application via `init_db()` in `app.py`.

## 💻 Running the Application

1. Make sure your virtual environment is activated.
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to: **http://127.0.0.1:5000**

## 🧠 Training & Dataset Management

If you want to train your own model on a COCO dataset:

1. **Convert the Dataset:**
   Place your COCO dataset inside the `dataset/` directory and run:
   ```bash
   python coco_to_classification.py
   ```
   *This automatically sorts annotated medicine strips into `dataset_cls/train/real` and `dataset_cls/train/fake`.*

2. **Train the Model:**
   ```bash
   python train_model.py
   ```
   *This performs a 2-phase fine-tuning process on a MobileNetV2 architecture and automatically balances class weights.*

## 📂 Project Structure

```text
PHARMAGUARD/
│
├── app.py                 # Main Flask application and routing
├── coco_to_classification.py # Dataset converter (COCO to flat folders)
├── train_model.py         # 2-Phase CNN Training script
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
│
├── dataset/               # Raw COCO data & drug databases (CSV/XLSX)
├── dataset_cls/           # Auto-generated classification training data
├── models/                # Saved models (.keras / .h5)
│   └── predict.py         # Core 3-Layer 7-Pass Verification Logic
│
├── database/              # SQLite DB and initialization scripts
├── utils/                 # Auth and Database Helper functions
├── static/                # CSS, JS, and Images
└── templates/             # HTML Jinja2 templates
```

## 📄 License
This project is licensed under the MIT License.
