# 🌱 Smart Crop Selection Web App

## 📌 Overview

This **Smart Crop Selection** web app helps farmers and researchers determine the most suitable crop to plant based on **soil parameters**. It uses **machine learning** to make predictions based on nitrogen (N), phosphorus (P), potassium (K), and pH levels in the soil.

## 🚀 Features

- **User-friendly web interface** built with Streamlit.
- **Takes soil inputs**: Nitrogen (N), Phosphorus (P), Potassium (K), and pH.
- **Predicts the best crop** for given soil conditions.
- **Displays accuracy score** of the model.
- **Deployed on Streamlit Community Cloud** for easy access.

## 🛠️ Technologies Used

- **Python**
- **Streamlit** (for web UI)
- **scikit-learn** (for machine learning)
- **XGBoost** (for classification)
- **Pandas & NumPy** (for data processing)

## 📥 Installation & Usage

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/thanhtungvudata/smart_crop_selection_web_app.git
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**

```bash
streamlit run app.py
```

The app will open in your browser at **[http://localhost:8501](http://localhost:8501)**.

## 🌍 Deployment on Streamlit Community Cloud

To deploy the app on **Streamlit Community Cloud**:

1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Set `app.py` as the main file and deploy!

## 📊 Model Training

The machine learning model used in this app:

- Uses **XGBoost Classifier**.
- Was trained on soil datasets with **stratified K-fold cross-validation**.
- Encodes crop labels using **LabelEncoder**.
- Standardizes soil parameters using **StandardScaler**.

To retrain the model:

```bash
python scripts/train_model.py
```

The trained model is saved as `models/crop_model.pkl`.

## 🔗 Live Demo

You can access the deployed version of this app [https://smartcropselection.streamlit.app/](https://smartcropselection.streamlit.app/).

## 💡 Future Improvements

- Adding **weather conditions** as an input feature.
- Supporting **real-time soil data collection** via IoT sensors.
- Expanding the model to suggest **fertilizer recommendations**.

## 🤝 Contributing

Pull requests are welcome! Feel free to **fork this repository** and submit improvements.

## 📜 License

This project is **open-source** under the **MIT License**.

