import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from bokeh.palettes import Reds, Greens
import math

# Load the model :
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)

#  Define the functions


def remove_punctuation(news):
    punctuationfree = "".join(
        [i for i in news if i not in string.punctuation])
    return punctuationfree


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def tokenization(news):
    # tokenize the text using RegexpTokenizer
    # r'\w+' means that we want to tokenize the text by words (r'\w+')
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(news)
    return tokens


def remove_stopwords(news):
    stopword = stopwords.words('english')
    news_stopwords = [i for i in news if i not in stopword]
    return news_stopwords


# lemmitization using WordNetLemmatizer
def lemmatization(news):
    lemmatizer = WordNetLemmatizer()
    news_lemmatization = [lemmatizer.lemmatize(i) for i in news]
    return news_lemmatization

#  Define the predict_fake_news function


def predict_fake_news(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = tokenization(text)
    text = remove_stopwords(text)
    text = lemmatization(text)
    text = ' '.join(text)
    text = model[0].transform([text])
    prediction_prob = model[1].predict_proba(text)
    return prediction_prob[0][0], prediction_prob[0][1]


# Define the circle_progress function
def circle_progress(percentage, color):
    size = 200
    stroke_width = 20
    radius = (size - stroke_width) / 2
    circumference = 2 * math.pi * radius
    offset = circumference - percentage / 100 * circumference

    return f"""
    <style>
      .circle-animation {{
        stroke-dasharray: {circumference};
        stroke-dashoffset: {circumference};
        animation: progress 2s ease-out forwards;
      }}

      @keyframes progress {{
        to {{
          stroke-dashoffset: {offset};
        }}
      }}
    </style>
    <center>
    <svg width="{size}" height="{size}">
      <circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="#262730" stroke-width="{stroke_width}" fill="none" />
      <circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="{color}" stroke-width="{stroke_width}" stroke-dasharray="{circumference}" stroke-dashoffset="{circumference}" fill="none" transform="rotate(-90 {size/2} {size/2})" class="circle-animation">
        <animate attributeName="stroke-dashoffset" from="{circumference}" to="{offset}" dur="2s" fill="freeze" />
      </circle>
      <text x="{size/2}" y="{size/2}" font-size="56" fill="{color}" text-anchor="middle" dominant-baseline="central">{percentage}%</text>
    </svg>
    </center>
    """


def main():
    st.set_page_config(page_title='Fake News Classification App')
    st.title('Fake News Classification App ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Predict The News")

    # Check if predict button is clicked
    if predict_btt:
        # Predict the news
        prediction_class = predict_fake_news(sentence)
        print(prediction_class)
        fake_prob, real_prob = prediction_class

        # Remove the old circle and create a new one
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>Results</h3>", unsafe_allow_html=True)
        circle_div = st.empty()

        # Check the prediction and update the circle with the result
        if fake_prob > real_prob:
            st.markdown("<br>", unsafe_allow_html=True)
            st.error('Fake News')
            percentage = round(fake_prob * 100)
            color = Reds[4][2]
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.success('Real News')
            percentage = round(real_prob * 100)
            color = Greens[4][2]

        # Display the circle animation
        circle_div.markdown(circle_progress(
            percentage, color), unsafe_allow_html=True)


# Run the app
if __name__ == '__main__':
    main()

# to run the app : streamlit run app.py
