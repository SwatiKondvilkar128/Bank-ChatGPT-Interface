from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import pandas as pd
import numpy as np
# !pip install streamlit
import streamlit as st
import fsspec 

# Load the preprocessed data
BankMerge = pd.read_csv('BankMerge.csv')
BankMerge = pd.read_csv('BankMerge.csv',encoding='ISO-8859-1')

# BankMerge = pd.concat([BankFAQs,BankData2])

# Define the TD-IDF vectorizer and fit it to the data
tdidf = TfidfVectorizer()
tdidf.fit(BankMerge['Question'].str.lower())

# Define the support vector machine model and fit it to the data
svc_model = SVC(kernel='linear')
svc_model.fit(tdidf.transform(BankMerge['Question'].str.lower()), BankMerge['Class'])

# Define a function to get the answer to a given question
def get_answer(question):
    # Vectorize the question
    question_tdidf = tdidf.transform([question.lower()])
    
    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(question_tdidf, tdidf.transform(BankMerge['Question'].str.lower()))

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Get the predicted class of the query
    predicted_class = svc_model.predict(question_tdidf)[0]
    
    # If the predicted class is not the same as the actual class, return an error message
    if predicted_class != BankMerge.iloc[most_similar_idx]['Class']:
        return 'Sorry could not find an appropriate answer. Kindly contact customer care number'
    
    # Get the answer and construct the response
    answer = BankMerge.iloc[most_similar_idx]['Answer']
    response = f"Answer: {answer}"
    
    return response


# Create a streamlit app

def app():
    # Set the app title
    st.set_page_config(page_title="Bank Chatbot Interface", page_icon=":bank:")

    # Add a title and description to the app
    st.title("Welcome to Bank ChatGPT Interface")
    st.markdown("Greetings! I am a ChatBot programmed to provide you with the information that you require.")
    st.write("I am here to assist you, please feel free to ask me any questions that you may have.")
    # st.write("What can I help you with today?")
    # ("This app uses a Machine Learning Model to answer the frequently asked questions about banking.")

    # Create a text input for the user to ask a question
    question = st.text_input("Enter your question below")

    # Add a button to submit the question
    if st.button("Submit"):
        # Check if the user has entered a question
        if question == "":
            st.warning("Please enter a question.")
        else:
            # Call the get_answer function to predict the answer to the question
            answer = get_answer(question)

            # Display the answer to the user
            st.success(answer)

# Run the streamlit app
if __name__ == '__main__':
    app()