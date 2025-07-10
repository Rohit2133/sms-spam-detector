import streamlit as st
import pickle
import spacy
from streamlit_lottie import st_lottie
import requests

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# # Custom CSS for styling
# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://images.unsplash.com/photo-1497493292307-31c376b6e479");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url()


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def transform_text(text):
    doc = nlp(str(text))  # Ensure input is string
    tokens = []

    for token in doc:
        if (
            not token.is_stop  # Remove stopwords
            and not token.is_punct  # Remove punctuation
            and token.text.isalnum() # Keep only alphabetic tokens
        ):
            tokens.append(token.lemma_.lower())  # Lemmatize and lowercase

    return " ".join(tokens)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))



# Sidebar
with st.sidebar:
    st.image("spam.png", width=120)
    st.title("📡 Spam Classifier")
    st.markdown("""
    - Built using **Scikit-learn**, **spaCy**, **TF-IDF**, and **Multinomial Naive Bayes Classifier**
    - Enter a message to check if it's spam.
    - created by [Rohit Aggarwal](http://github.com/Rohit2133/sms-spam-detector)
    """)

# Load Lottie Animation
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

Lottie_URL = "https://assets1.lottiefiles.com/packages/lf20_x62chJ.json"

# Display animation
st_lottie(Lottie_URL, height=200)

# Main title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📬 SMS Spam Detector</h1>", unsafe_allow_html=True)

# Text input
input_text = st.text_area("✍️ Enter your Message", height=150, placeholder="e.g. Congratulations! You've won a prize...")



# Predict button
if st.button("🚀 Predict"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        main_text = transform_text(input_text)
        vector = tfidf.transform([main_text])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("🚨 This message is **Spam**")
        else:
            st.success("✅ This message is **Not Spam**")
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Show transformed text
        with st.expander("🔍 Show Preprocessed Text"):
            st.code(main_text, language='text')
        
        st.markdown("<br>", unsafe_allow_html=True)
        # Show spam probability
        prob = model.predict_proba(vector)[0]
        spam_score = prob[1]  # Index 1 = spam, 0 = not spam
        st.write(f"📊 **Spam Probability**: `{spam_score:.2f}`")
        if spam_score > 0.5:
            st.warning("⚠️ High probability of being spam!")
        else:
            st.info("ℹ️ Low probability of being spam.")
