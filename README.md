# 📦 Olist E-commerce Supply Chain Optimization

## 🚀 Project Overview
This project analyzes the **Olist E-commerce dataset** (Brazilian marketplace) to optimize supply chain logistics. The goal is to predict delivery times, identify late shipments before they happen, and understand customer sentiment to reduce churn.

**Tech Stack:** Python, Pandas, Scikit-learn, Gradient Boosting, Matplotlib, Seaborn, Streamlit, Deep Translator

---

## 📊 Key Modules & Results

### Module 1: Delivery Time Prediction (Regression)
- **Objective:** Predict the exact number of days a package will take to arrive  
- **Model:** Random Forest Regressor  
- **Performance:**
  - **RMSE:** ~6.3 days  
- **Key Insight:** Distance and freight value are the strongest drivers of delivery time  

---

### Module 2: Late Delivery Classification (Risk Analysis)
- **Objective:** Classify whether an order will be **Late** or **On-Time** to proactively alert customers  
- **Challenge:** Highly imbalanced data (only ~7% of orders were late)  
- **Solution:** Implemented **cost-sensitive learning** using weighted loss with Gradient Boosting  
- **Performance:**
  - **Recall (Late Orders):** Improved from **1% to 53%**  
- **Business Impact:** The model successfully captures **over 50% of potential late deliveries**, enabling proactive customer communication and reducing negative customer experience  

---

### Module 3: Customer Segmentation & NLP *(In Progress)*
- **Clustering:** Customer segmentation using **RFM (Recency, Frequency, Monetary) Analysis**  
- **NLP:** Sentiment analysis of Portuguese customer reviews to correlate delivery delays with negative feedback  

---

## 🛠️ How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/kunalbaghelkb/Olist_Ecommerce_Project.git

2. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  

   *(Place the extracted files inside the `data/` directory)*
  
3. Navigate to the project directory:  
    ```bash
    cd Olist_Ecommerce_Supply_Chain

4. Create a virtual environment:  
   ```bash
   python3.11 -m venv venv

5. Install required dependencies:  
   ```bash
   pip install -r requirements.txt

6. Run the Streamlit application
   ```bash
   streamlit run app.py
   
---

## 📈 Future Scope
- **Customer Segmentation:** Implement K-Means clustering to identify VIP customers vs. churn-risk customers using RFM analysis  
- **NLP Sentiment Analysis:** Apply TF-IDF–based sentiment modeling on Portuguese reviews and correlate sentiment with delivery delays  
- **Sales Forecasting:** Implement time-series models (ARIMA / Prophet) to predict inventory requirements for upcoming quarters  

---

## 👤 Author
**Kunal Baghel**  
Aspiring Data Scientist | AI Engineer  

🔗 [LinkedIn](https://linkedin.com/in/kunalbaghelz)  
🔗 [GitHub](https://github.com/kunalbaghelkb)  

---

## 📜 License
This project is open-source and available under the **MIT License**.
