# sms-spam-detector
A simple web application to classify SMS messages as **Spam** or **Not Spam** using **Machine Learning** and **Natural Language Processing (NLP)**.

Built with **Streamlit**, **Scikit-learn**, **spaCy**, and **TF-IDF**, and deployed on **Streamlit Cloud** 🚀.

---

## 🧠 How It Works

1. User inputs a message.
2. Message is preprocessed using spaCy:
   - Stopword removal
   - Punctuation and digit removal
   - Lemmatization
3. Transformed text is vectorized using **TF-IDF**.
4. A **Multinomial Naive Bayes** model classifies it as:
   - ✅ Not Spam
   - 🚨 Spam
5. Displays the **spam probability** with helpful visual cues.

---

## 🚀 Live Demo

👉 [Click here to try the app!](https://your-username-sms-spam-detector.streamlit.app/)

---


## 📦 Features

- 🧠 Text preprocessing with spaCy
- 📊 TF-IDF vectorization
- 🤖 Spam classification with Multinomial Naive Bayes
- 📉 Spam probability feedback
- 💡 Beautiful and interactive UI with animations

---

## 📁 Project Structure

sms-spam-detector/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🧰 Tech Stack

- **Python**
- **Streamlit**
- **spaCy**
- **scikit-learn**
- **pandas**
- **TF-IDF**
- **Multinomial Naive Bayes**

---

## ✅ Installation (Run Locally)

```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run app.py
