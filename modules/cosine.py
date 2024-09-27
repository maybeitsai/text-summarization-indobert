import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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