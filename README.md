# Text Summarization and Chatbot

## Project Overview

### This project involves two key components:

- **Text Summarization (elc.py)**: A script that processes documents, extracting important sentences based on various features like term frequency-inverse 
document frequency (TF-IDF), sentence length, and position, among others.

- **Chatbot (chat.py)**: A simple rule-based chatbot that responds to user queries using TF-IDF and cosine 
similarity to provide context-aware responses from a corpus of text data.

## Requirements

- Python 3.0
- nltk
- numpy
- pandas
- scikit-learn

Before running the scripts, ensure you have installed the required libraries using pip:
```
pip install nltk numpy pandas scikit-learn
```

## File Descriptions
### Text Summarization(elc.py)
This script reads multiple text files from a specified directory, processes them, and generates a summary of the content.

**Key Features:**
- Sentence Tokenization: Breaks the text into individual sentences.
- TF-IDF Calculation: Creates term frequency and inverse document frequency matrices to determine word importance.
- Sentence Scoring: Sentences are scored based on multiple criteria:
  - Cue phrases (e.g., "for example", "in summary")
  - Numerical data
  - Sentence length
  - Position of the sentence within the document
  - Word Frequency
  - Proper noun count
  - Word matches with the heading
- Summarization: Sentences with scores above a certain threshold are included in the summary.
  
**Usage:**
To run the summarization process, specify the directory containing your documents within the script and run:
```
python elc.py
```

### Simple Chatbot(chat.py)
This script implements a basic chatbot that reads the contents of the same directory of text files and engages in 
conversation by using the TF-IDF vectorizer and cosine similarity for generating responses.

This script implements a basic chatbot that reads the contents of the same directory of text files and engages in conversation by using the TF-IDF vectorizer and cosine similarity for generating responses.

**Key Features:**
- Greeting Detection: Responds with a greeting if the user starts the conversation.
- TF-IDF and Cosine Similarity: Uses TF-IDF to find the most relevant response based on the user query.
- Fallback Responses: If the chatbot cannot find a suitable answer, it responds with a default message.

**Usage:**
Run the chatbot using:
```
python chat.py
```
> The chatbot will prompt you to enter queries. Type bye to exit the conversation.

## How It Works

### Text Summarization (elc.py)
- The script processes documents and calculates various metrics like TF-IDF scores, cue phrases, sentence position, etc.
- These metrics are used to score each sentence.
- Sentences with scores higher than a threshold are selected for inclusion in the final summary.

### Chatbot (chat.py)
- The chatbot reads text files and tokenizes the text into sentences.
- When a user inputs a query, it calculates the TF-IDF vectors and computes the cosine similarity to find the most
relevant response from the available sentences.
- It also handles simple greetings and exits when the user types bye.




  
