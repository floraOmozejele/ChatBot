import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

sales = sales[1].reset_index(drop = True)
cust = cust[1].reset_index(drop = True)

new_data = pd.DataFrame()
new_data['Question'] = cust
new_data['Answer'] = sales



# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)

new_data['tokenized Question'] = new_data['Question'].apply(preprocess_text)

xtrain = new_data['tokenized Question'].to_list()

# # Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

#------------------------STREAMLIT IMPLIMENTATION---------------------------------------------------------
st.header('Project Background Information', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ. ")

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>ORGANISATIONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Flora James</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


user_hist = []
reply_hist = []


robot_image, space1, space2, chats = st.columns(4)
with robot_image:
    robot_image.image('IMG_0164.JPG', width = 500)



with chats:
    user_message = chats.text_input('You can ask me anything')

    def responder(text):
        user_input_processed = preprocess_text(text)
        vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
        similarity_scores = cosine_similarity(vectorized_user_input, corpus)
        argument_maximum = similarity_scores.argmax()
        return (new_data['Answer'].iloc[argument_maximum])

    bot_greetings = ['Hello user, You are chatting with Flora....How may i help you',
                'Hi Dear, i am here if you need me',
                'Hey, what do need me to do',
                'Hello i am here for you',
                'How can i be of help' ]

    bot_farewell = ['Thanks for your usage....bye',
                'Alright dear...Hope to see you soon',
                'Hope i was of help to know...Bye',
                'Do you have anymore question you to ask....Bye',
                'Thanks for reaching out...Hope i answer all your question...Bye']

    human_greeting = ['Hi', 'Hello there', 'Hiyya', 'Hey', 'hello', 'Wassup']

    human_exits = ['Thanks bye', 'bye', 'quite', 'exit', 'bye bye', 'close']

    import random
    random_greeting = random.choice(bot_greetings)
    random_farewell = random.choice(bot_farewell)

    if user_message.lower() in human_exits:
        chats.write(f'\nChatbot: {random_farewell}!')
        user_hist.append(user_message)
        reply_hist.append(random_greeting)

    elif user_message.lower() in human_greeting:
        chats.write(f'\nChatbot:{random_greeting}!')
        user_hist.append(user_message)
        reply_hist.append(random_greeting)

    elif user_message =='':
        chats.write('')

    else:
        response = responder(user_message)
        chats.write(f'\nChatbot: {response}')
        user_hist.append(user_message)
        reply_hist.append(random_greeting)


# Save the history of user texts
import csv
with open('history.txt', 'a') as file:
        for item in user_hist:
            file.write(str(item) + '\n')

# Save the history bot reply
with open('reply.txt', 'a') as file:
       for item in reply_hist:
            file.write(str(item) + '\n')

# Import the file to display it in the frontend 
with open('history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('reply.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({'User Input': data1, 'Bot Reply': data2})

# History = pd.Series(data)
st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width= True)
# st.sidebar.write(data2)

     




