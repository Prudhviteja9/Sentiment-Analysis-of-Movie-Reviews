import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Function to preprocess the input text
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Load the model and vectorizer
@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    with open(r'C:\Users\yedla\OneDrive\Desktop\New folder -1\review_classifier_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open(r'C:\Users\yedla\OneDrive\Desktop\New folder -1\tf-idf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Streamlit application layout
st.title('Movie Review Sentiment Analyzer')
st.write('Enter a movie review to check if it is positive or negative:')

# Text input
user_input = st.text_area("Review", height=150)

if st.button('Classify'):
    # Preprocess the input
    processed_input = preprocess_text(user_input)
    processed_input = vectorizer.transform([processed_input])
    prediction = model.predict(processed_input)
    
    # Display results
    if prediction[0] == 1:
        st.write('This review is likely **positive**.')
    else:
        st.write('This review is likely **negative**.')

# Add some footer info
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

