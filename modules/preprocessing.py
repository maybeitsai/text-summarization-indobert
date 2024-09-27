"""
Module untuk ekstraksi fitur.
"""
import os
import re

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
    # Remove unwanted characters, keeping only periods
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
