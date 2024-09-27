import nltk
import string
import matplotlib.pyplot as plt
from gensim import corpora
from wordcloud import WordCloud
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')

# Preprocess text for LDA
def preprocess_LDA(text):
    stop_words = set(stopwords.words('indonesian'))  # Use Indonesian stopwords if applicable
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]
    return tokens

def lda_topic_modeling(categorized_texts):
    topic_models = {}

    for file_name, sections in categorized_texts.items():
        # Extract sentences from each category
        latar_belakang_sentences = [item['sentence'] for item in sections['latar_belakang']]
        tujuan_sentences = [item['sentence'] for item in sections['tujuan']]
        metode_sentences = [item['sentence'] for item in sections['metode']]
        
        # Combine sentences into documents
        background_doc = ' '.join(latar_belakang_sentences)
        objectives_doc = ' '.join(tujuan_sentences)
        methods_doc = ' '.join(metode_sentences)
        
        # Preprocess the documents
        docs = [preprocess_LDA(background_doc), preprocess_LDA(objectives_doc), preprocess_LDA(methods_doc)]
        
        # Create a dictionary and corpus for LDA
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        
        # Build LDA model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, random_state=42)
        
        # Store the topics and their words
        topic_models[file_name] = lda_model.print_topics(num_topics=3)
    
    return topic_models

def plot_lda_wordcloud_for_file(lda_topics, file_name="yohanes kurnia_UK.txt"):
    if file_name in lda_topics:
        topics = lda_topics[file_name]

        # Set up the figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        # Generate a word cloud for each topic
        for idx, (topic_id, topic) in enumerate(topics):
            # Extract the words and their weights
            topic_words = {word.split('*')[1].replace('"', ''): float(word.split('*')[0]) for word in topic.split(' + ')}
            
            # Generate the word cloud
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(topic_words)
            
            # Plot the word cloud
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f"Topic {idx + 1}", fontsize=16)

        # Set the main title
        plt.suptitle(f"LDA Topic Visualization - {file_name}", fontsize=18)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No LDA topic data available for file: {file_name}")

def plot_lda_wordclouds(lda_topics):
    for file_name, topics in lda_topics.items():
        # Set up the figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

        # Generate a word cloud for each topic
        for idx, (topic_id, topic) in enumerate(topics):
            # Extract the words and their weights
            topic_words = {word.split('*')[1].replace('"', ''): float(word.split('*')[0]) for word in topic.split(' + ')}
            
            # Generate the word cloud
            wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(topic_words)
            
            # Plot the word cloud
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].axis('off')
            axes[idx].set_title(f"Topic {idx + 1}", fontsize=16)

        # Set the main title
        plt.suptitle(f"LDA Topic Visualization - {file_name}", fontsize=18)
        plt.tight_layout()
        plt.show()