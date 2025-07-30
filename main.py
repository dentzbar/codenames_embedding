from utils import load_words
import random

# Load the word dataset
words = load_words('simple_word_dataset.pkl')

# Sample words for each category
red_agents = random.sample(words, 9)
blue_agents = random.sample([w for w in words if w not in red_agents], 8) 
inocents = random.sample([w for w in words if w not in red_agents + blue_agents], 8)
print(f"Red agents: {red_agents}")
print(f"Blue agents: {blue_agents}")
print(f"Inocents: {inocents}")