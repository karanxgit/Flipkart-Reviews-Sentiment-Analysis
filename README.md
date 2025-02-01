# **Flipkart Reviews Sentiment Analysis Using Python**  

This project aims to determine whether a **Flipkart product review** is **positive or negative** using **machine learning techniques**. By analyzing customer feedback and ratings, we can extract valuable insights into **product quality** and suggest areas for improvement.  

---

## **📌 Table of Contents**  
1. [Introduction](#introduction)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Preprocessing](#preprocessing)  
5. [Analysis](#analysis)  
6. [Model Training](#model-training)  
7. [Evaluation](#evaluation)  
8. [Conclusion](#conclusion)  

---

## **📌 Introduction**  
This project leverages **Python libraries** such as **Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn, and WordCloud** for **data preprocessing, analysis, and visualization**. The workflow includes:  

✔ Importing and cleaning the dataset  
✔ Preprocessing text data for analysis  
✔ Converting text into **numerical vectors** using **TF-IDF**  
✔ Training a **Decision Tree Classifier**  
✔ Evaluating the model's **performance**  

---

## **📌 Installation**  
To run this project, ensure you have Python installed along with the required dependencies. You can install them using:  

```bash
pip install pandas scikit-learn nltk matplotlib seaborn wordcloud
```

Additionally, download the necessary **NLTK stopwords** by running the following command in Python:  

```python
import nltk
nltk.download('stopwords')
```

---

## **📌 Usage**  
1. **Clone the repository** to your local machine.  
2. **Download the dataset** and place it in the project directory.  
3. **Run the script** using Python.  

---

## **📌 Preprocessing**  
The text data undergoes **cleaning and transformation** to improve the model’s performance. This involves:  
✔ **Removing punctuation and special characters**  
✔ **Converting text to lowercase**  
✔ **Eliminating stopwords using NLTK**  

These steps help in structuring the text for better analysis and accurate model predictions.  

---

## **📌 Analysis**  
Data analysis includes:  
📊 **Exploring unique ratings**  
📊 **Visualizing the distribution of ratings using countplots**  
📊 **Converting ratings into binary labels** (e.g., reviews with a score ≤4 are considered **negative**, while 5+ are **positive**).  

---

## **📌 Model Training**  
1. **Transforming text data into numerical vectors** using **TF-IDF (Term Frequency-Inverse Document Frequency)**.  
2. **Splitting the dataset** into **training and testing sets** to evaluate model performance.  
3. **Training a Decision Tree Classifier** to categorize reviews as **positive or negative**.  

---

## **📌 Evaluation**  
To assess the model’s performance, we use:  
✔ **Accuracy Score** to measure overall performance.  
✔ **Confusion Matrix** to visualize:  
   - **True Positives (TP)**
   - **True Negatives (TN)**
   - **False Positives (FP)**
   - **False Negatives (FN)**  

---

## **📌 Conclusion**  
The **Decision Tree Classifier** performs effectively on the given dataset, achieving a **high accuracy score**.  
💡 Future enhancements could include:  
📌 **Training on a larger dataset** for improved generalization.  
📌 **Exploring other machine learning models** (e.g., Random Forest, SVM, or Deep Learning models) for better accuracy.  

---
