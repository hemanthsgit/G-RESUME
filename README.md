# G-RESUME

G-Resume
AI-powered career assistant for interview preparation and resume optimization.​

Overview
G-Resume analyzes video interviews using computer vision and audio processing, optimizes resumes, and matches job descriptions through ML-powered recommendations.​

Features
Video Interview Analysis — MediaPipe for facial/gesture analysis, Librosa for voice metrics​

Resume Optimization — AI-driven suggestions for ATS compatibility​

JD Matching — Skill gap analysis and job fit scoring​

Secure API — JWT authentication for user data protection​

Tech Stack
Frontend: React
Backend: Flask (Python)
ML/AI: scikit-learn, TensorFlow, PyTorch
Data: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Testing: Postman​

Installation
bash
# Clone repository
git clone https://github.com/RahuljiV2004/G-RESUME.git
cd G-RESUME

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
cd frontend
npm install
Usage
bash
# Start Flask backend
python app.py

# Start React frontend (separate terminal)
cd frontend
npm start
Access at http://localhost:3000.​

API Testing
Import Postman collection from /api/docs for endpoint testing
