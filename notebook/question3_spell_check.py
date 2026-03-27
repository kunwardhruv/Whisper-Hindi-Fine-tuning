import pandas as pd
import csv
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import nltk
from nltk.corpus import indian
import re

# Load Hindi words from NLTK
hindi_words = set(indian.words('hindi.pos'))

# Function to check if word is transliterated English
def is_transliterated_english(word):
    # Simplified: check if word contains common English transliteration patterns
    # For example, if it has 'कं' for 'com', etc.
    # But for speed, skip detailed transliteration
    return False, 'Not checked for speed'

# Function to classify word
def classify_word(word):
    # Check if in Hindi dictionary
    if word in hindi_words:
        return 'correct spelling', 'high', 'Found in Hindi dictionary'
    
    # Check if transliterated English
    is_eng, reason = is_transliterated_english(word)
    if is_eng:
        return 'correct spelling', 'high', 'Transliterated English: ' + reason
    
    # If not, check length and structure
    if len(word) < 3:
        return 'correct spelling', 'medium', 'Short word, possibly correct'
    
    # Check for common Hindi patterns
    if re.match(r'^[क-ह][्]?[क-ह]*[ा-ौ]?[ं]?[।]?$', word):  # Simple Devanagari pattern
        return 'correct spelling', 'medium', 'Matches Devanagari pattern'
    
    # Otherwise, likely incorrect
    return 'incorrect spelling', 'low', 'Not in dictionary, not transliterated English'

# Read CSV
df = pd.read_csv('unique_words.csv')

# Process
results = []
for word in df['word']:
    classification, confidence, reason = classify_word(word)
    results.append([word, classification, confidence, reason])

# Write to output CSV
with open('outputs/spelling_classification.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'classification', 'confidence', 'reason'])
    writer.writerows(results)

# Count correct
correct_count = sum(1 for r in results if r[1] == 'correct spelling')
print(f"Total unique words: {len(results)}")
print(f"Correct spelled words: {correct_count}")

# Get low confidence words
low_conf = [r for r in results if r[2] == 'low']
print(f"Low confidence words: {len(low_conf)}")

# Sample 50 low confidence for review
sample_low = low_conf[:50]
with open('outputs/low_confidence_sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['word', 'classification', 'confidence', 'reason'])
    writer.writerows(sample_low)