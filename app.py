
import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import speech_recognition as sr
import easyocr as ocr
from PIL import Image
import numpy as np
from pydub import AudioSegment
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import openpyxl
from sklearn.metrics import accuracy_score
import nltk
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        #Stats Area
        num_messages,words,num_media_messages,num_links = helper.fetch_stats(selected_user,df)
        st.title("Top stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(num_links)
# finding the busiest users in the group (Group level)
    if selected_user == 'Overall':
        st.title('Most Busy Users')
        x, new_df = helper.most_busy_users(df)
        fig, ax = plt.subplots()

        col1, col2 = st.columns(2)

        with col1:
            ax.bar(x.index, x.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

#wordcloud
    st.title("Wordcloud")
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots()
    ax.imshow(df_wc)
    st.pyplot(fig)

#most_common_words
    most_common_df = helper.most_common_words(selected_user, df)
    # st.dataframe(most_common_df)
    fig, ax = plt.subplots()

    ax.barh(most_common_df[0], most_common_df[1])
    plt.xticks(rotation='vertical')
    st.title('Most commmon words')
    st.pyplot(fig)

#emoji analysis
    emoji_df = helper.emoji_helper(selected_user, df)
    st.title("Emoji Analysis")

    col1,col2 = st.columns(2)

    with col1:
        st.dataframe(emoji_df)
    with col2:
        fig,ax = plt.subplots()
        ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
        st.pyplot(fig)

#monthly timeline
    st.title("Monthly Timeline")
    timeline = helper.monthly_timeline(selected_user,df)
    fig,ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'],color='pink')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# daily timeline
    st.title("Daily Timeline")
    daily_timeline = helper.daily_timeline(selected_user, df)
    fig, ax = plt.subplots()
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# activity map
    st.title('Activity Map')
    col1,col2 = st.columns(2)

    with col1:
        st.header("Most busy day")
        busy_day = helper.week_activity_map(selected_user,df)
        fig,ax = plt.subplots()
        ax.bar(busy_day.index,busy_day.values,color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    with col2:
        st.header("Most busy month")
        busy_month = helper.month_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(busy_month.index, busy_month.values,color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    st.title("Weekly Activity Map")
    user_heatmap = helper.activity_heatmap(selected_user,df)
    fig,ax = plt.subplots()
    ax = sns.heatmap(user_heatmap)
    st.pyplot(fig)

    st.dataframe(df)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        #Stats Area
        num_messages,words,num_media_messages,num_links = helper.fetch_stats(selected_user,df)
        st.title("Top stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(num_links)
# finding the busiest users in the group (Group level)
    if selected_user == 'Overall':
        st.title('Most Busy Users')
        x, new_df = helper.most_busy_users(df)
        fig, ax = plt.subplots()

        col1, col2 = st.columns(2)

        with col1:
            ax.bar(x.index, x.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

#wordcloud
    st.title("Wordcloud")
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots()
    ax.imshow(df_wc)
    st.pyplot(fig)

#most_common_words
    most_common_df = helper.most_common_words(selected_user, df)
    # st.dataframe(most_common_df)
    fig, ax = plt.subplots()

    ax.barh(most_common_df[0], most_common_df[1])
    plt.xticks(rotation='vertical')
    st.title('Most commmon words')
    st.pyplot(fig)

#emoji analysis
    emoji_df = helper.emoji_helper(selected_user, df)
    st.title("Emoji Analysis")

    col1,col2 = st.columns(2)

    with col1:
        st.dataframe(emoji_df)
    with col2:
        fig,ax = plt.subplots()
        ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
        st.pyplot(fig)

#monthly timeline
    st.title("Monthly Timeline")
    timeline = helper.monthly_timeline(selected_user,df)
    fig,ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'],color='pink')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# daily timeline
    st.title("Daily Timeline")
    daily_timeline = helper.daily_timeline(selected_user, df)
    fig, ax = plt.subplots()
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

# activity map
    st.title('Activity Map')
    col1,col2 = st.columns(2)

    with col1:
        st.header("Most busy day")
        busy_day = helper.week_activity_map(selected_user,df)
        fig,ax = plt.subplots()
        ax.bar(busy_day.index,busy_day.values,color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    with col2:
        st.header("Most busy month")
        busy_month = helper.month_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(busy_month.index, busy_month.values,color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    st.title("Weekly Activity Map")
    user_heatmap = helper.activity_heatmap(selected_user,df)
    fig,ax = plt.subplots()
    ax = sns.heatmap(user_heatmap)
    st.pyplot(fig)

# Create a Streamlit app for text extraction from images
st.title("Extract text from Images")

# Image uploader
uploaded_image = st.file_uploader("Upload your image for text extraction", type=['png', 'jpg', 'jpeg'])

@st.cache_data
def load_ocr_model():
    reader = ocr.Reader(['en'], model_storage_directory='.')
    return reader

reader = load_ocr_model()

if uploaded_image is not None:
    input_image = Image.open(uploaded_image)  # Reading an image
    st.image(input_image)  # Displaying the image for user understanding

    with st.spinner("AI at work!!"):
        result = reader.readtext(np.array(input_image))
        result_text = []  # Empty list for storing results

        for text in result:
            result_text.append(text[1])
        st.write(result_text)
    st.success("Here are the extracted texts from the image.")
else:
    st.write("Upload an image or audio file for processing")
def transcribe_audio_chunks(audio_file, chunk_duration_ms=5000):
    r = sr.Recognizer()

    audio = AudioSegment.from_wav(audio_file)
    chunk_length_ms = chunk_duration_ms

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk.export("temp.wav", format="wav")  # Export the chunk to a temporary WAV file

        with sr.AudioFile("temp.wav") as source:
            try:
                transcription = r.record(source)
                result = r.recognize_google(transcription)
                yield "Transcription: " + result
            except sr.UnknownValueError:
                pass  # Ignore unrecognized chunks
            except sr.RequestError as e:
                yield "Error: " + str(e)

# Create a Streamlit app
st.title("Speech Recognition ")

# Upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Transcribing..."):
        for result in transcribe_audio_chunks(uploaded_file):
            if result.startswith("Error: "):
                st.error(result)
            else:
                st.success(result)

# Streamlit title and layout config
st.title('News Authenticator')
st.write('Enter a news title and author to check if it is Real or Fake.')

# Check if stopwords are downloaded, otherwise download
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load and preprocess data
news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)
x = news_dataset['content'].values
y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(x_train, y_train)

# Streamlit input and prediction
author = st.text_input("Author")
title = st.text_input("Title")
if st.button("Predict"):
    content = author + ' ' + title
    content = stemming(content)
    content_vectorized = vectorizer.transform([content])
    prediction = model.predict(content_vectorized)

    if prediction[0] == 0:
        st.write('The news is Real')
    else:
        st.write('The news is Fake')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
urls_data = pd.read_csv("urldata.csv")

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens =[]
    for i in tkns_BySlash:
        tokens = str(i).split('-')
        tkns_ByDot =[]
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens +tokens +tkns_ByDot
    total_Tokens = list(set(total_Tokens))
    if 'com' in total_Tokens:
        total_Tokens.remove('com')
    return total_Tokens

y = urls_data["label"]
url_list = urls_data["url"]

# Vectorize the URL data
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
logit = LogisticRegression()
logit.fit(X_train, y_train)

# Streamlit app
st.title("Phishing URL Detector")

# Input for user to enter a URL
user_input = st.text_input("Enter a URL:")

if user_input:
    X_predict = [user_input]
    X_predict = vectorizer.transform(X_predict)
    prediction = logit.predict(X_predict)

    st.write("Prediction:", prediction[0])


