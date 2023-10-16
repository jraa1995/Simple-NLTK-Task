import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# Load the dataset
with open('8_BikeInjury.txt', 'r') as file:
    data = file.read()

# Tokenize the sentences into words
tokens = word_tokenize(data)

# Convert to lowercase and remove non-alphabetic tokens
words = [word.lower() for word in tokens if word.isalpha()]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Compute the frequency distribution of words
fdist = FreqDist(filtered_words)

# Display the 25 most common injury descriptions
common_injuries = fdist.most_common(25)
for word, frequency in common_injuries:
    print(f"{word}: {frequency}")

# Separate the words and frequencies
words, frequencies = zip(*common_injuries)

# Adjust the figure size
plt.figure(figsize=(15,10))

# Plot the frequencies using horizontal bars
bars = plt.barh(words, frequencies, color='#ffd166')

# Set the title
plt.title('Top 25 Biking Injury Descriptions')

# Annotate each bar with its respective count
for bar in bars:
    width = bar.get_width()
    plt.text(width + 5,  # Increase this value to move numbers further to the right
             bar.get_y() + bar.get_height() / 2,
             str(int(width)),
             ha='center',
             va='center')

# Invert the y-axis to have the word with the highest count on top
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# WordCloud
wordcloud = WordCloud(
    width=1500,
    height=800,
    background_color='white',
    max_words=50,
    min_font_size=10,
    contour_width=3,
    contour_color='steelblue'
).generate(' '.join(filtered_words))

plt.figure(figsize=(18,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Biking Injury Descriptions Word Cloud")
plt.show()

# Tokenize the text into sentences
tokenized_sentences = sent_tokenize(data)

# Display several injury description sentences
for i, sentence in enumerate(tokenized_sentences[:35]):  # Change 5 to the number of sentences you want to display
    print(f"Sentence {i+1}: {sentence}")

