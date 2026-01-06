# 🎬 CineSense AI

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

**CineSense AI** is an advanced Natural Language Processing (NLP) application that predicts movie genres based on their plot summaries. Powered by **Machine Learning (TF-IDF + Logistic Regression)** and wrapped in a modern, responsive **FastAPI** backend with a sleek **HTML/Tailwind CSS** frontend.

---

## 🚀 Features

- **🎭 Real-time Genre Prediction**: Instantly predicts genres from any text input.
- **📊 Confidence Scores**: Displays probability percentages for the top predicted genres.
- **🎨 Modern UI**: Beautiful glassmorphism design with smooth animations and interactive elements.
- **⚡ High Performance**: Optimized TF-IDF vectorization for sub-second inference.
- **📱 Fully Responsive**: Works seamlessly on desktop and mobile devices.

---

## 🛠️ Tech Stack

### **Backend**
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Server**: Uvicorn
- **ML Libraries**: Scikit-learn, Pandas, NumPy, Joblib

### **Frontend**
- **Structure**: HTML5
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) + Bootstrap 5
- **Interactivity**: Vanilla JavaScript (Fetch API)

### **Machine Learning**
- **Algorithm**: Logistic Regression (One-vs-Rest)
- **Feature Extraction**: TF-IDF (Unigrams + Bigrams, 20k features)
- **Dataset**: 54,000+ movie plot summaries (multilabel)

---

## 📂 Project Structure

```bash
CineSense-AI/
├── backend/
│   ├── app.py                # 🚀 Main FastAPI application
│   └── model/                # 🧠 Trained ML models & vectorizers
│       ├── movie_genre_model.pkl
│       └── tfidf_vectorizer.pkl
├── frontend/
│   ├── index.html            # 🎨 Main user interface
│   └── assets/               # 🖼️ Static assets (icons, images)
├── movie_genre_model.ipynb   # 📓 Jupyter Notebook for model training
├── requirements.txt          # 📦 Python Dependencies
└── README.md                 # 📖 Project Documentation
```

---

## ⚡ Installation & Setup

Follow these steps to run the project locally.

### **1. Clone the Repository**
```bash
git clone https://github.com/adi3733/ArNitaInfotech_AI-ML_MOVIE_GENRE_CLASSIFICATION
cd cinesense-ai
```

### **2. Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### **3. Run the Backend Server**
Navigate to the backend directory and start the FastAPI server:
```bash
cd backend
python app.py
```
*The server will start at `http://127.0.0.1:8000`*

### **4. Launch the Frontend**
Simply open the `frontend/index.html` file in your preferred web browser.

---

## 📸 Usage

1. **Enter a Plot**: Type or paste a movie plot summary into the text area.
2. **Click Predict**: Hit the "🚀 Predict Genre" button.
3. **View Results**: See the top predicted genre with a confidence gauge and other likely genres.

---

## 🧠 Model Details

The model was trained on a dataset of over **54,000 movies**. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors and a **Logistic Regression** classifier to predict probabilities for **27 different genres**, including:
- Action, Comedy, Drama, Horror, Sci-Fi, Thriller, Romance, Documentary, and more.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <b>Aditya Ghayal</b>
</p>
