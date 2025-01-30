# 🎬 Movie Rating Prediction

This program predicts how a user would rate a given movie using **Machine Learning** models like **Linear Regression** and **Random Forest Regressor**.

It uses **movie ratings data** to train the model and provides a **prediction score** based on the user's previous ratings.

---

## 🚀 Features

- Loads and preprocesses **movie ratings data**
- Supports **two machine learning models**:
  - ✅ **Linear Regression** (more accurate, but slower)
  - ✅ **Random Forest Regressor** (faster, but slightly less accurate)
- Allows users to **input their User ID & Movie ID**
- Displays **Mean Squared Error (MSE) & R² Score**
- Predicts how much a **specific user will rate a specific movie**
- **Supports interactive REPL** mode

---

## ⚡️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ChimentiMatt/movie-rating-predictor-machine-learning
cd movie-rating-prediction
```

### 2️⃣ Create & Activate a Virtual Environment

Create a virtual environment using Python:
```bash
python -m venv myenv
```

#### **Activate the Environment**:
**For macOS/Linux:**
```bash
source myenv/bin/activate
```
**For Windows (CMD):**
```cmd
myenv\Scripts\activate
```
**For Windows (PowerShell):**
```powershell
myenv\Scripts\Activate.ps1
```

---

## 🛠 Install Dependencies
After activating the virtual environment, install required libraries:
```bash
pip install -r requirements.txt
```

---

## 🎯 How to Use

Once everything is set up, run the program using:
```bash
python main.py
```

### **Interactive REPL Mode**
The program will guide you through **selecting a model and entering User & Movie IDs**:

```
Movie Rating Prediction REPL
1: Use Linear Regression (More accurate but slower)
2: Use Random Forest Regressor (Less accurate but much faster)
Q: Quit
Select a model (1/2) or 'Q' to quit:
```

1. **Enter User ID** (from dataset)
2. **Enter Movie ID** (from dataset)
3. The program will **train the model** and **predict** your rating!

---

## 📊 Example Output
```
Running prediction
User: 1  
Movie: Toy Story (1995) (ID: 1)  
Model: Linear Regression

Loading...
Done Processing

--------------------------------------------- Results ----------------------------------------------
Mean Squared Error: 0.6494
R² Score: 0.4025

Predicted rating for User 1 on Movie 1: 4.53
Prediction Accuracy for 'Like' (Ratings ≥ 2.5): 88.72%
----------------------------------------------------------------------------------------------------
```

---

## 🔧 Troubleshooting

### 1️⃣ Virtual Environment Not Found?
If you see an error when activating the environment, ensure you are in the correct directory:
```bash
cd movie-rating-prediction
source myenv/bin/activate  # or myenv\Scripts\activate for Windows
```

### 2️⃣ Missing Dependencies?
Run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Wrong Python Version?
Ensure you are using **Python 3.8 or later**:
```bash
python --version
```
---

## 📎 Credits
- **Dataset**: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- **Libraries Used**:
  - `pandas`, `numpy`, `scikit-learn`, `threading`

---

## 🎮 Enjoy Predicting Movie Ratings! 🍿

