# SignBridge — Real-Time Sign Language to Text & Speech Converter

> Production-grade CNN + LSTM system for real-time sign language recognition.
> Final year project built as an industry-grade application.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Node.js 18+
node --version

# MySQL 8.0+
mysql --version

# CUDA (for RTX 4050 GPU training)
nvidia-smi
```

### 1. Clone & Setup Python Environment

```bash
git clone <repo-url>
cd sign-language-converter

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt
pip install mediapipe opencv-python tqdm scikit-learn seaborn matplotlib
```

### 2. Setup Database

```bash
mysql -u root -p < scripts/init_db.sql
cp .env.example .env
# Edit .env with your DB credentials
```

### 3. Collect Dataset

```bash
# Collect custom webcam data
python scripts/collect_data.py

# (Optional) Convert ASL Kaggle dataset
# Download: https://www.kaggle.com/grassknoted/asl-alphabet
# Extract to: dataset/raw/asl_alphabet_train/
python scripts/convert_static_to_sequence.py

# Augment data
python scripts/augment_data.py

# Preprocess and create splits
python scripts/preprocess_dataset.py
```

### 4. Train Model

```bash
# Stage 1: Pretrain (both stages)
python model/train.py

# Or individually:
python model/train.py --stage 1   # pretrain only
python model/train.py --stage 2   # finetune only

# Evaluate
python model/evaluate.py

# (Optional) Convert to TFLite
python model/convert_tflite.py --quantization float16
```

### 5. Start Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
```

### 6. Start Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# App: http://localhost:5173
```

---

## 🏗️ Architecture

```
Webcam (30 frames)
    ↓
React Frontend → POST /predict
    ↓
FastAPI Backend
    ↓
MediaPipe Hand Detection → ROI Extraction
    ↓
TimeDistributed MobileNetV2 (CNN)
    ↓
Temporal Frame Attention
    ↓
Bidirectional LSTM × 2
    ↓
Dense + Dropout → Softmax
    ↓
Prediction + Confidence → MySQL Log
    ↓
Web Speech API (TTS)
```

---

## 📊 GPU Optimization (RTX 4050)

- **Mixed precision** (float16) — ~2× speedup
- **Batch size 8** for pretraining, 4 for fine-tuning
- **VRAM limit** set to 5120 MB (leaves 1GB headroom)
- **TimeDistributed CNN** — shared weights across frames
- **MobileNetV2 backbone** — lightweight but accurate
- **Gradient checkpointing** enabled via Keras

---

## 🔄 Continuous Learning

1. Users correct wrong predictions in the UI
2. Frames saved to `new_data/<label>/<uuid>/`
3. DB entry created in `feedback_data`
4. Run retraining when enough samples accumulate:

```bash
python scripts/retrain.py --min-samples 10 --epochs 15
```

---

## 🌐 Deployment

### Frontend → Vercel
```bash
cd frontend
npm run build
# Deploy dist/ to Vercel
```

### Backend → Render
```bash
# Set environment variables in Render dashboard
# Start command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

---

## 📁 Project Structure

```
├── dataset/           # Raw, processed, augmented data
├── model/             # Architecture, training, evaluation
├── backend/           # FastAPI server
├── frontend/          # React + Tailwind UI
├── scripts/           # Data collection, preprocessing, retraining
├── new_data/          # User feedback (for continuous learning)
└── .env.example       # Environment variables template
```