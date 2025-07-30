import nltk
from nltk.corpus import wordnet as wn
import pickle

# One-time download
nltk.download('wordnet')

# Get all lemmas for nouns, adjectives, and verbs
def get_wordnet_words(pos_list=('n', 'v', 'a')):
    words = set()
    for pos in pos_list:
        for syn in wn.all_synsets(pos=pos):
            for lemma in syn.lemmas():
                word = lemma.name().replace('_', ' ')
                if word.isalpha():
                    words.add(word.lower())
    return sorted(words)

def save_words(words, filename='wordnet_words.pkl'):
    """Save word list to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(words, f)

def load_words(filename='wordnet_words.pkl'):
    """Load word list from a pickle file"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"No saved word list found at {filename}")
        return None

if __name__ == "__main__":
    # This gives you ~150k+ unique words
    filtered_words = get_wordnet_words()
    print(f"Collected {len(filtered_words)} words")

    # Save the filtered words
    path = "wordnet_words_n_v_a.pkl"
    save_words(filtered_words, path)
    print(f"Words saved to {path}")

