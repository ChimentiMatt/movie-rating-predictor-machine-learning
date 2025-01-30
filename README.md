# 🎬 Movie Rating Predictor

This program predicts how a user would rate a given movie by using **Machine Learning** models **Linear Regression** and **Random Forest Regressor**.

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

## 🔍 What Does This Program Do?

1. Collects and preprocesses movie and rating data  
2. Extracts relevant features (e.g., user rating history, movie genres, release year)  
3. Trains either a **Linear Regression** or **Random Forest Regressor** model  
4. Uses the trained model to predict how a user would rate a specific movie  
5. Outputs accuracy metrics like **MSE (Mean Squared Error)** and **R² Score**  

By comparing the two models, users can analyze which approach works best for predicting movie ratings.

---

## 🤖 Machine Learning Models Used

This program utilizes two machine learning models to predict movie ratings:

### 📈 Linear Regression

Linear Regression is a fundamental statistical method used to model the relationship between a dependent variable (rating) and independent variables (movie features and user behavior). It assumes a linear relationship and fits a straight line to minimize errors.

#### ✅ Pros:
- Easy to interpret  
- Works well when relationships are linear  
- Faster training time  

#### ❌ Cons:
- Limited in handling complex relationships  
- Sensitive to outliers  

### 🌲 Random Forest Regressor

Random Forest Regressor is an ensemble learning method that builds multiple decision trees and averages their predictions. It improves accuracy and reduces overfitting by randomly selecting subsets of data for training.

#### ✅ Pros:
- Handles non-linear relationships well  
- More accurate predictions for complex datasets  
- Resistant to overfitting  

#### ❌ Cons:
- Slower training time compared to Linear Regression  
- Harder to interpret than a single decision tree  

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
--------------------------------------------- Results ----------------------------------------------
User Id: 1 
Movie Title: Toy Story (1995)
Movie Id: 1


Mean Squared Error: 0.6494
R² Score: 0.4025

Predicted Rating: 4.53
Prediction Accuracy (Ratings ≥ 2.5): 88.72%
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

