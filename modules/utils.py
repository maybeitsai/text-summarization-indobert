"""
Module untuk ekstraksi fitur.
"""
import os
import re
import PyPDF2
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import warnings 
warnings.filterwarnings("ignore")
nltk.download('punkt')

def load_and_segment_texts(input_clean_folder):
    segmented_texts = {}
    
    for file_name in os.listdir(input_clean_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_clean_folder, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # Tokenize text into sentences
                sentences = nltk.sent_tokenize(text)
                segmented_texts[file_name] = sentences
    
    return segmented_texts

def categorize_sentences(segmented_texts):
    """
    Fungsi untuk mengkategorikan kalimat dalam dokumen ke dalam 'title', 'latar_belakang', 'tujuan', dan 'metode'.
    """
    categorized_texts = {}

    for file_name, sentences in segmented_texts.items():
        categorized_texts[file_name] = {
            'title': [],
            'latar_belakang': [],
            'tujuan': [],
            'metode': []
        }

        # State flags
        in_latar_belakang = False
        in_tujuan = False
        in_metode = False

        for i, sentence in enumerate(sentences):
            lower_sentence = sentence.lower()

            # 1. Title: always the first sentence
            if i == 0:
                categorized_texts[file_name]['title'].append(sentence)
                continue

            # 2. Latar Belakang
            if "latar belakang" in lower_sentence:
                in_latar_belakang = True

            if in_latar_belakang:
                if any(kw in lower_sentence for kw in ["tujuan penelitian", "tujuan masalah", "batasan dan tujuan"]):
                    in_latar_belakang = False
                    in_tujuan = True
                    categorized_texts[file_name]['tujuan'].append(sentence)
                    continue  # Skip this sentence (boundary marker)

                categorized_texts[file_name]['latar_belakang'].append(sentence)
                continue

            # 3. Tujuan
            if in_tujuan:
                if any(kw in lower_sentence for kw in ["metode penelitian", "metodologi"]):
                    in_tujuan = False
                    in_metode = True
                    categorized_texts[file_name]['metode'].append(sentence)
                    continue  # Skip this sentence (boundary marker)

                categorized_texts[file_name]['tujuan'].append(sentence)
                continue

            # 4. Metode
            if in_metode:
                categorized_texts[file_name]['metode'].append(sentence)
                continue

    return categorized_texts

def extract_features(sentences, title_words, cue_words):
    """
    Fungsi untuk mengekstrak fitur dari setiap kalimat dalam sebuah dokumen.
    Menghapus kalimat jika unimportant == 0 atau length == 0.
    """
    features = []
    
    # Define keywords for unimportant information
    unimportant_words = ["contohnya", "sebagai contoh", "contoh", "misalnya", "misal", "misalkan"]
    
    # Iterate over each sentence to extract features
    for i, sentence in enumerate(sentences):
        sentence_features = {}
        
        # Feature 1: Unimportant Information
        if any(word in sentence.lower() for word in unimportant_words):
            sentence_features['unimportant'] = 0
        else:
            sentence_features['unimportant'] = 1

        # Feature 2: Sentence Length
        words = sentence.split()
        sentence_features['length'] = 0 if len(words) < 6 else len(words)

        # Feature 3: Words in Title (Bi formula: intersection over union)
        title_count = len(set(words).intersection(set(title_words)))
        union_count = len(set(words).union(set(title_words)))
        sentence_features['words_in_title'] = title_count / (union_count if union_count > 0 else 1)

        # Feature 4: Sentence Position (B2i formula, higher at start/end, lower in the middle)
        Mp = len(sentences)  # Total number of sentences
        ji = i + 1  # Sentence index, starting from 1
        if Mp > 1:
            # New formula to give higher score at start or end, and lower in the middle
            sentence_features['position'] = 1 - (2 * abs((ji / Mp) - 0.5))
        else:
            sentence_features['position'] = 1  # If there's only one sentence

        # Feature 5: Cue Words (B3i formula)
        cue_count = sum(1 for word in cue_words if word in sentence.lower())
        Tfi = sum(1 for sentence in sentences for word in cue_words if word in sentence.lower())
        sentence_features['cue_words'] = cue_count / (Tfi if Tfi > 0 else 1)

        # Skip sentence if unimportant == 0 or length == 0
        if sentence_features['unimportant'] == 0 or sentence_features['length'] == 0:
            continue  # Skip this sentence and move to the next
        
        features.append(sentence_features)
    
    return features

def categorize_and_extract_features(segmented_texts, title_words_mapping, cue_words):
    """
    Fungsi utama untuk mengkategorikan dan mengekstrak fitur dari kalimat dalam dokumen.
    Menghapus kalimat jika unimportant == 0 atau length == 0.
    """
    # Dictionary to store categorized texts with features
    categorized_texts_with_features = {}

    # Iterate over each document and its sentences
    for file_name, sentences in segmented_texts.items():
        categorized_texts_with_features[file_name] = {
            'title': [],
            'latar_belakang': [],
            'tujuan': [],
            'metode': []
        }

        # Get title words for the current document
        title_words = title_words_mapping.get(file_name, [])

        # Categorize sentences first
        categorized_sentences = categorize_sentences({file_name: sentences})

        # Extract features and associate them with categorized sentences
        for category in ['title', 'latar_belakang', 'tujuan', 'metode']:
            category_sentences = categorized_sentences[file_name][category]
            category_features = extract_features(category_sentences, title_words, cue_words)
            
            # Store sentences along with their features
            categorized_texts_with_features[file_name][category] = [
                {'sentence': sent, 'features': feat}
                for sent, feat in zip(category_sentences, category_features)
                if feat['unimportant'] != 0 and feat['length'] != 0  # Additional check to remove sentences
            ]

    return categorized_texts_with_features


def compute_sentence_score(features, weight_title=2.0, weight_position=1.0, weight_cue=3.0):
    # Calculate a combined score for each sentence based on feature weights
    score = (weight_title * features['words_in_title'] +
             weight_position * features['position'] +
             weight_cue * features['cue_words'])
    return score

def select_best_sentences(sentences, category, percentage=0.3):
    # Compute scores for each sentence in the category
    scored_sentences = [(sentence['sentence'], compute_sentence_score(sentence['features'])) 
                        for sentence in sentences]
    
    # Sort sentences by their computed score in descending order
    scored_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    
    # Select top 30% best sentences
    num_best_sentences = max(1, int(len(scored_sentences) * percentage))
    best_sentences = [sentence for sentence, score in scored_sentences[:num_best_sentences]]
    
    return best_sentences

def summarize_text(categorized_texts, percentage=0.3):
    summaries = {}
    
    for file_name, categories in categorized_texts.items():
        summary = {}
        
        # Get the title (first sentence)
        title = categories['title'][0]['sentence'] if categories['title'] else "Title not found"
        summary['judul'] = title
        
        # Select 30% best sentences from latar_belakang, tujuan, and metode
        if categories['latar_belakang']:
            best_latar_belakang = select_best_sentences(categories['latar_belakang'], 'latar_belakang', percentage)
            summary['latar_belakang'] = best_latar_belakang
        else:
            summary['latar_belakang'] = ["Latar belakang tidak ditemukan."]
        
        if categories['tujuan']:
            best_tujuan = select_best_sentences(categories['tujuan'], 'tujuan', percentage)
            summary['tujuan'] = best_tujuan
        else:
            summary['tujuan'] = ["Tujuan tidak ditemukan."]
        
        if categories['metode']:
            best_metode = select_best_sentences(categories['metode'], 'metode', percentage)
            summary['metode'] = best_metode
        else:
            summary['metode'] = ["Metode tidak ditemukan."]
        
        summaries[file_name] = summary
    
    return summaries

def extract_text_from_pdfs(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = ''
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        # Save extracted text to a .txt file
        txt_filename = pdf_file.replace('.pdf', '.txt')
        txt_path = os.path.join(output_folder, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        print(f"Extracted text from {pdf_file} and saved to {txt_filename}")

def save_cleaned_texts(cleaned_texts, output_clean_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_clean_folder):
        os.makedirs(output_clean_folder)
    
    for filename, cleaned_text in cleaned_texts.items():
        # Create a new .txt filename in the cleaned folder
        clean_txt_path = os.path.join(output_clean_folder, filename)
        
        # Save the cleaned text into the .txt file
        with open(clean_txt_path, 'w', encoding='utf-8') as text_file:
            text_file.write(cleaned_text)
        print(f"Saved cleaned text to {clean_txt_path}")

def clean_text(text):
    # Remove unwanted characters, keeping only periods and commas
    cleaned_text = re.sub(r'[^\w\s.]', '', text)
    # Remove newline and tab characters
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\t', ' ')
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text

def preprocess_text_files(txt_folder):
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    cleaned_texts = {}
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
        cleaned_text = clean_text(text)
        cleaned_texts[txt_file] = cleaned_text
        print(f"Preprocessed {txt_file}")
    
    return cleaned_texts

def plot_cosine_similarity(file_name, sim_values):
        # Prepare data for the heatmap
        similarity_matrix = np.array([
            [1, sim_values['latar_belakang_vs_tujuan'], sim_values['latar_belakang_vs_metode']],
            [sim_values['latar_belakang_vs_tujuan'], 1, sim_values['tujuan_vs_metode']],
            [sim_values['latar_belakang_vs_metode'], sim_values['tujuan_vs_metode'], 1]
        ])

        # Define labels for the heatmap
        labels = ["Latar Belakang", "Tujuan", "Metode"]

        # Plot heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(similarity_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)

        # Set title and labels
        plt.title(f"Cosine Similarity Heatmap - {file_name}", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()

def cosine_similarity_analysis(categorized_texts):
    similarities = {}

    for file_name, sections in categorized_texts.items():
        similarities[file_name] = {}
        
        # Extract sentences from each category
        latar_belakang_sentences = [item['sentence'] for item in sections['latar_belakang']]
        tujuan_sentences = [item['sentence'] for item in sections['tujuan']]
        metode_sentences = [item['sentence'] for item in sections['metode']]

        # Combine sentences into documents
        background_doc = ' '.join(latar_belakang_sentences)
        objectives_doc = ' '.join(tujuan_sentences)
        methods_doc = ' '.join(metode_sentences)
        
        # Create a list of documents to compare
        docs = [background_doc, objectives_doc, methods_doc]
        
        # Apply TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        # Compute cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Store similarity between sections
        similarities[file_name]['latar_belakang_vs_tujuan'] = cosine_sim[0, 1]
        similarities[file_name]['latar_belakang_vs_metode'] = cosine_sim[0, 2]
        similarities[file_name]['tujuan_vs_metode'] = cosine_sim[1, 2]

    return similarities

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

def plot_cosine_similarity_for_file(similarities, file_name="yohanes kurnia_UK.txt"):
    if file_name in similarities:
        # Prepare data for the heatmap
        sim_values = similarities[file_name]
        similarity_matrix = np.array([
            [1, sim_values['latar_belakang_vs_tujuan'], sim_values['latar_belakang_vs_metode']],
            [sim_values['latar_belakang_vs_tujuan'], 1, sim_values['tujuan_vs_metode']],
            [sim_values['latar_belakang_vs_metode'], sim_values['tujuan_vs_metode'], 1]
        ])

        # Define labels for the heatmap
        labels = ["Latar Belakang", "Tujuan", "Metode"]

        # Plot heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(similarity_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True)

        # Set title and labels
        plt.title(f"Cosine Similarity Heatmap - {file_name}", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()
    else:
        print(f"No similarity data available for file: {file_name}")

