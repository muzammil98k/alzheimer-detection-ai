# 🧠 Alzheimer’s Disease Detection using 3D Deep Learning

A deep learning system for detecting Alzheimer’s Disease from **3D MRI brain scans** using a hybrid **3D CNN + Transformer architecture**.

The project includes:

- PyTorch training pipeline
- Flask inference API
- Modern React-based web interface

---

# 📌 Project Overview

Alzheimer’s disease is a progressive neurological disorder that causes brain atrophy and cognitive decline. Early detection using MRI scans can help clinicians diagnose the disease earlier.

This project builds an **AI-powered system that analyzes 3D MRI brain scans and predicts Alzheimer’s Disease**.

The system includes:

- MRI preprocessing pipeline
- 3D deep learning architecture
- Cross-validation training
- Web interface for inference
- Flask backend API

---

# 🚀 Features

✔ 3D MRI preprocessing  
✔ 3D CNN + Transformer architecture  
✔ Cross-validation training  
✔ Flask API for model inference  
✔ React web interface  
✔ Confidence score visualization  
✔ Modern AI diagnostic interface  

---

# 🧠 Model Architecture

The model combines **3D Convolutional Neural Networks** with **Transformer attention**.

Pipeline:

```
MRI Volume
↓
Preprocessing
↓
3D CNN Feature Extraction
↓
Transformer Attention Layer
↓
Fully Connected Layer
↓
Alzheimer / Non-Alzheimer Classification
```

---

# 📊 Model Performance

Evaluation was performed using **5-Fold Cross Validation**.

| Fold | Accuracy |
|-----|--------|
| Fold 1 | 0.75 |
| Fold 2 | 0.56 |
| Fold 3 | 0.73 |
| Fold 4 | 0.80 |
| Fold 5 | 0.53 |

Average Accuracy

```
67.5%
```

---

# 📁 Project Structure

```
alzheimer-detection-ai
│
├── alzheimer-ui
│   ├── src
│   ├── public
│   └── package.json
│
├── data
│   └── sample_mri.pt
│
├── models
│   └── best_model.pth
│
├── utils
│   ├── preprocess_all.py
│   └── dataset.py
│
├── static
├── templates
│
├── app.py
├── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/muzammil98k/alzheimer-detection-ai.git
cd alzheimer-detection-ai
```

---

# 📦 Install Backend Dependencies

```
pip install -r requirements.txt
```

---

# ▶ Run Backend (Flask API)

```
python app.py
```

Server will start at

```
http://127.0.0.1:5000
```

---

# 💻 Run Frontend

```
cd alzheimer-ui
npm install
npm run dev
```

Open browser

```
http://localhost:5173
```

---

# 🖥️ Web Interface

The web interface allows users to:

1. Upload MRI brain scan
2. Run AI analysis
3. View prediction results
4. See confidence score

Example output:

```
Prediction: Alzheimer Detected
Confidence: 72%
```

---

# 🧰 Tech Stack

Backend

- Python
- PyTorch
- Flask
- Numpy
- Scikit-Learn
- Nibabel

Frontend

- React
- Vite
- Three.js
- Modern UI animations

---

# 📚 Dataset

MRI brain scans from the **OASIS Alzheimer Dataset**.

Dataset used for:

- MRI preprocessing
- Model training
- Validation

Dataset reference:

Open Access Series of Imaging Studies (OASIS)

---

# 🔬 Future Improvements

- Larger MRI datasets
- Multi-modal learning (MRI + clinical data)
- Explainable AI using Grad-CAM
- 3D brain visualization
- Cloud deployment

---

# 👨‍💻 Author

Mohammed Muzammil Uddin Ahmed

AI / Machine Learning Student  
Lords Institute of Engineering and Technology

GitHub  
https://github.com/muzammil98k

LinkedIn  
https://www.linkedin.com/in/muzammil-ahmed-576345311

---

⭐ If you find this project useful, consider giving it a star!
