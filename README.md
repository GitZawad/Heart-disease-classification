# ❤️ Heart Disease Classification Using Logistic Regression and Random Forest Algorithms

A machine learning project that explores heart disease classification using **Logistic Regression** and **Random Forest** algorithms. The project compares both models using clinical data from the UCI Cleveland Heart Disease dataset.

## 📌 Project Overview

The objective of this project is to evaluate the performance of two supervised machine learning algorithms in classifying patients with and without heart disease. It involves data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Original Records:** 303 patients
- **Records After Preprocessing:** 297 patients
- **Features:** 13
- **Classification:** Binary (0 = No Heart Disease, 1 = Heart Disease)
- **Missing Records Removed:** 6

## ⚙️ Methodology

The project follows these steps:

1. **Data Preprocessing:** Handling missing values and preparing the dataset.
2. **Exploratory Data Analysis:** Analyzing class distribution, age distribution, and feature correlations using visualizations.
3. **Feature Standardization:** Standardizing features before training the Logistic Regression model.
4. **Model Training:** Implementing Logistic Regression and Random Forest classifiers.
5. **Model Evaluation:** Comparing model performance using accuracy, precision, recall, F1-score, and confusion matrices.

## 🤖 Machine Learning Algorithms

- **Logistic Regression:** A supervised classification algorithm used as a baseline model.
- **Random Forest:** An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

## 📈 Results

Both models were trained and evaluated using an 80:20 training-testing split.

| Evaluation Metric | Logistic Regression | Random Forest |
|:---|:---:|:---:|
| Accuracy | 86.67% | **88.33%** |
| Precision | 0.83 | **0.84** |
| Recall | 0.83 | **0.88** |
| F1-Score | 0.83 | **0.86** |

### 🏆 Key Findings

- Random Forest achieved the highest accuracy of **88.33%**.
- Random Forest performed better in recall and F1-score.
- Both algorithms demonstrated promising classification performance on the selected dataset.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/GitZawad/Heart-disease-classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Heart-disease-classification
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells to explore the data analysis, model training, and evaluation.

## 📁 Project Structure

```text
Heart-disease-classification/
│
├── Heart-Disease-Classification.ipynb
├── README.md
└── dataset/
```

*Note: The file and folder names above are illustrative. Adjust them to match the actual repository structure.*

## ⚠️ Disclaimer

This project is developed for educational and research purposes. Its predictions are not intended to replace professional medical diagnosis or clinical decision-making.

## 👨‍💻 Author

**Kazi Zawad Uddin**

BSc in Software Engineering  
Zhengzhou University, China

- GitHub: [@GitZawad](https://github.com/GitZawad)

---

⭐ If you find this project interesting, feel free to explore the repository!
