# EMI Predict AI Pro

## 📌 Project Overview
**EMI Predict AI Pro** is an advanced AI-powered application designed to help users assess their loan eligibility and calculate safe Equated Monthly Installments (EMI). By combining rule-based financial logic with Machine Learning predictions, the app provides accurate, personalized financial insights.

## 🚀 Features

### 1. 🏠 Home
- Landing page with a modern, responsive design.
- Overview of the application's capabilities.

### 2. 🧮 EMI Calculator
- **Hybrid Logic**: Uses both credit score rules and ML models.
- **Eligibility Check**: Instantly tells you if you are eligible based on Credit Score.
- **Safe EMI Prediction**: If eligible, predicts a safe monthly EMI amount tailored to your financial profile.

### 3. 📊 Model Performance
- Visualizes the performance of the underlying Machine Learning models.
- TRACKS metrics using **MLflow**.
- Interactive charts powered by **Plotly**.

### 4. 📈 Data Insights
- Explore the dataset used for training.
- Gain insights into global financial trends and user demographics.

### 5. 🤖 AI Assistant
- An integrated AI assistant to answer your financial queries and guide you through the app.

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Visualization**: [Plotly](https://plotly.com/)
- **Experiment Tracking**: [MLflow](https://mlflow.org/)

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd EMI_Predict_AI_Pro
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
```
EMI_Predict_AI_Pro/
├── .gitignore          # Files to ignore in git
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── app.py              # Main entry point for Streamlit
├── pages/              # Streamlit pages (Home, EMI Calculator, etc.)
├── utils/              # Helper functions, config, and styles
├── data/               # Dataset files (ignored in git)
├── models/             # Trained ML models
└── mlruns/             # MLflow tracking data (ignored in git)
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License.
