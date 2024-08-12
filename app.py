import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text preprocessing function (same as used in training)
ps = PorterStemmer()

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit app
st.title("SMS Spam Classifier")

st.write("""
### Enter the SMS message below to check if it's spam or ham:
""")

# Input box for user to enter an SMS message
user_input = st.text_area("Enter SMS")

if st.button('Predict'):
    # Preprocess the input
    transformed_input = text_transform(user_input)
    vectorized_input = vectorizer.transform([transformed_input])

    # Make prediction
    prediction = model.predict(vectorized_input)

    # Output the result
    if prediction[0] == 1:
        st.error("This message is Spam!")
    else:
        st.success("This message is Not Spam!")

st.write("""
### Example Messages:
- Congratulations! You've won a free ticket to the Bahamas. Call now!
- Hey, are we still on for dinner tonight?
""")
