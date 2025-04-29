
# 🛡️ Supervised Learning for Real-Time Phishing Website Detection

This project demonstrates the use of supervised machine learning algorithms to detect phishing websites in real-time. It involves data preprocessing, exploratory data analysis, model building, evaluation, and visualization using Python and Jupyter Notebook.

---

## 📌 Table of Contents
- [🔍 Project Overview](#-project-overview)
- [📁 Dataset](#-dataset)
- [🛠️ Tools & Libraries Used](#️-tools--libraries-used)
- [📊 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🧠 Machine Learning Models](#-machine-learning-models)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [✅ Results](#-results)
- [📎 How to Run](#-how-to-run)
- [📚 Conclusion](#-conclusion)
- [📄 License](#-license)

---

## 🔍 Project Overview
Phishing websites are designed to deceive users into sharing sensitive information such as login credentials, credit card numbers, and personal data. This project uses **supervised learning techniques** to identify phishing websites based on their features.

The models are trained on a labeled dataset of legitimate and phishing websites, enabling real-time prediction and classification.

---

## 📁 Dataset
The dataset used (`phishing_data.csv`) includes various website features and a target column `status` indicating whether the website is **phishing (1)** or **legitimate (0)**.

---

## 🛠️ Tools & Libraries Used
- Python 3.x
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

---

## 📊 Exploratory Data Analysis (EDA)
We analyzed:
- The balance between phishing and legitimate websites
- Correlation between features
- Visual trends using bar charts and heatmaps

---

## 🧠 Machine Learning Models
The following supervised learning models were used:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **XGBoost Classifier**

---

## 📈 Evaluation Metrics
To evaluate model performance:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-Score)**
- **ROC-AUC Curve**

---

## ✅ Results
- **Random Forest** and **XGBoost** performed best in terms of accuracy and generalization.
- **Logistic Regression** provided a strong baseline with good interpretability.
- Visualizations helped understand model decisions and class separability.

---

## 📎 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phishing-website-detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd phishing-website-detection
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Run `phishing_detection.ipynb` notebook step-by-step.

---

## 📚 Conclusion
This project proves that **supervised learning techniques** can effectively detect phishing websites. Using models like **Random Forest**, it’s possible to build real-time phishing detection systems that are accurate, scalable, and essential in cybersecurity today.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## ✨ Acknowledgements
- Open source phishing datasets
- scikit-learn & XGBoost community
