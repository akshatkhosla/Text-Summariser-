"""
ISHA GOEL - 102183055 - 2COE8
AKSHAT KHOSLA - 102053026 - 2COE8
ARYAN - 102003502 -  2COE20
"""

import nltk # natural language toolkit
import numpy as np
import pandas as pd
import os
import csv
import math
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def sent_preprocessing(sentences: list) -> list:
    cleaned_sentencs = [sent for sent in sentences if sent]
    for sent in sentences:
        if sent == '' or sent == ' ':
            print(1)
    return cleaned_sentencs


def text_preprocessing(sentences: list):

    #print('Preprocessing text')

    stop_words = set(stopwords.words('english'))

    clean_words = None
    for sent in sentences:
        words = word_tokenize(sent)
        words = [ps.stem(word.lower()) for word in words if word.isalnum()]
        clean_words = [word for word in words if word not in stop_words]

    return clean_words


def create_tf_matrix(sentences: list) -> dict:
    
    #print('Creating tf matrix.')

    tf_matrix = {}

    for sentence in sentences:
        tf_table = {}

        clean_words = text_preprocessing([sentence])
        words_count = len(word_tokenize(sentence))

        # Determining frequency of words in the sentence
        word_freq = {}
        for word in clean_words:
            word_freq[word] = (word_freq[word] + 1) if word in word_freq else 1

        # Calculating relative tf of the words in the sentence
        for word, count in word_freq.items():
            tf_table[word] = count / words_count

        tf_matrix[sentence[:15]] = tf_table

    return tf_matrix


def create_idf_matrix(sentences: list) -> dict:

    # print('Creating idf matrix.')

    idf_matrix = {}
    documents_count = len(sentences)
    sentence_word_table = {}

    # Getting words in the sentence
    for sentence in sentences:
        clean_words = text_preprocessing([sentence])
        sentence_word_table[sentence[:15]] = clean_words

    # Determining word count table with the count of sentences which contains the word.
    word_in_docs = {}
    for sent, words in sentence_word_table.items():
        for word in words:
            word_in_docs[word] = (word_in_docs[word] + 1) if word in word_in_docs else 1

    # Determining idf of the words in the sentence.
    for sent, words in sentence_word_table.items():
        idf_table = {}
        for word in words:
            idf_table[word] = math.log10(documents_count / float(word_in_docs[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def create_tf_idf_matrix(tf_matrix, idf_matrix) -> dict:
    
    # print('Calculating tf-idf of sentences.')

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def create_sentence_score_table(tf_idf_matrix) -> dict:

    # print('Creating sentence score table.')

    sentence_value = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        smoothing = 1
        sentence_value[sent] = (total_score_per_sentence + smoothing) / (count_words_in_sentence + smoothing)

    return sentence_value


def find_average_score(sentence_value):
    # print('Finding average score')

    sum = 0
    for val in sentence_value:
        sum += sentence_value[val]

    average = sum / len(sentence_value)

    return average

path='C:\\Users\\91981\\OneDrive\\Desktop\\elc\\unlabelled documents\\'
filelist = os.listdir(path)
for file in filelist:
    f=open(path+file, "r", encoding="latin-1")  
    text=f.read()
    sent_tokens = nltk.sent_tokenize(text)
    word_tokens = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    word_tokens_lower=[word.lower() for word in word_tokens]
    stopWords = list(set(stopwords.words("english")))
    word_tokens_refined=[x for x in word_tokens_lower if x not in stopWords]
    #print(len(word_tokens_refined))
    stem = [ ]
    ps = PorterStemmer( )
    for w in word_tokens_refined:
        stem.append(ps.stem(w))
    word_tokens_refined=stem
    
    #Cue - Phrases:
    QPhrases=["example", "anyway", "furthermore", "first", "second", "then", "now", "therefore", "hence", "lastly", "finally", "summary"]
    cue_phrases={ }
    for sentence in sent_tokens:
        cue_phrases[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for word in word_tokens:
            if word.lower() in QPhrases:
                cue_phrases[sentence] += 1
    maximum_frequency = max(cue_phrases.values())
    for k in cue_phrases.keys():
        try:
            cue_phrases[k] = (cue_phrases[k] / maximum_frequency)
            cue_phrases[k]=round(cue_phrases[k],3)
        except ZeroDivisionError:
            x=0
    #print(cue_phrases)
    
    #Numerical data
    
    numeric_data={ }
    for sentence in sent_tokens:
        numeric_data[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isdigit():
                numeric_data[sentence] += 1
    maximum_frequency = max(numeric_data.values())
    for k in numeric_data.keys():
        try:
            numeric_data[k] = (numeric_data[k]/maximum_frequency)
            numeric_data[k] = round(numeric_data[k], 3)
        except ZeroDivisionError:
            x=0
    #print(numeric_data)
    #print()
    
    #Sentence length
    sent_len_score={ }
    for sentence in sent_tokens:
        sent_len_score[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        if len(word_tokens) in range(0,10):
            sent_len_score[sentence]=1-0.05*(10-len(word_tokens))
        elif len(word_tokens) in range(7,20):
            sent_len_score[sentence]=1
        else:
            sent_len_score[sentence]=1-(0.05)*(len(word_tokens)-20)
    for k in sent_len_score.keys():
        sent_len_score[k]=round(sent_len_score[k],4)
    #print(sent_len_score.values())
    
    #Sentence Position
    sentence_position={ }
    d=1
    no_of_sent=len(sent_tokens)
    for i in range(no_of_sent):
        a=1/d
        b=1/(no_of_sent-d+1)
        sentence_position[sent_tokens[d-1]]=max(a,b)
        d=d+1
    for k in sentence_position.keys():
        sentence_position[k]=round(sentence_position[k],3)
    #print(sentence_position.values())
    
    #Word Frequency
    #Create a frequency table to compute the frequency of each word.
    freqTable = { }
    for word in word_tokens_refined:
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    for k in freqTable.keys():
        freqTable[k]= math.log10(1+freqTable[k])
        
    word_frequency={ }
    for sentence in sent_tokens:
        word_frequency[sentence]=0
        e=nltk.word_tokenize(sentence)
        f=[ ]
        for word in e:
            f.append(ps.stem(word))
        for word,freq in freqTable.items():
            if word in f:
                word_frequency[sentence]+=freq
    #print(word_frequency.values())
    
    
    #........................part 6 (upper cases).................................
    upper_case={ }
    for sentence in sent_tokens:
        upper_case[sentence] = 0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k.isupper():
                upper_case[sentence] += 1
    maximum_frequency = max(upper_case.values())
    for k in upper_case.keys():
        try:
            upper_case[k] = (upper_case[k]/maximum_frequency)
            upper_case[k] = round(upper_case[k], 3)
        except ZeroDivisionError:
            x=0
    #print(upper_case.values())
    
    #......................... Part 7 (number of proper noun)...................
    proper_noun={ }
    for sentence in sent_tokens:
        tagged_sent = pos_tag(sentence.split())
        propernouns = [word for word, pos in tagged_sent if pos == 'NNP'] 
        proper_noun[sentence]=len(propernouns)
    maximum_frequency = max(proper_noun.values())
    for k in proper_noun.keys():
        try:
            proper_noun[k] = (proper_noun[k]/maximum_frequency)
            proper_noun[k] = round(proper_noun[k], 3)
        except ZeroDivisionError:
            x=0
    #print(proper_noun)
    
    
    #................................. part 8 (word matches with heading) ...................
    head_match={ }
    heading=sent_tokens[0]
    for sentence in sent_tokens:
        head_match[sentence]=0
        word_tokens = nltk.word_tokenize(sentence)
        for k in word_tokens:
            if k not in stopWords:
                k = ps.stem(k)
                if k in ps.stem(heading):
                    head_match[sentence] += 1
    maximum_frequency = max(head_match.values())
    for k in head_match.keys():
        try:
            head_match[k] = (head_match[k]/maximum_frequency)
            head_match[k] = round(head_match[k], 3)
        except ZeroDivisionError:
            x=0
    #print(head_match)
    #print()
    
    
    # tf-idf
    tf_matrix = create_tf_matrix(sentences)
    #print('TF matrix', tf_matrix)

    idf_matrix = create_idf_matrix(sentences)
    #print('IDF matrix',idf_matrix)

    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print('TF-IDF matrix', tf_idf_matrix)
    #print('First document tfidf',tf_idf_matrix[list(tf_idf_matrix.keys())[0]])

    sentence_value = create_sentence_score_table(tf_idf_matrix)
    #print('Sentence Scores', sentence_value)

    threshold = find_average_score(sentence_value)
    #print('Threshold', threshold)
    
    # Total Score
    total_score={}
    for k in cue_phrases.keys():
        total_score[k]=cue_phrases[k]+numeric_data[k]+sent_len_score[k]+sentence_position[k]+word_frequency[k]+upper_case[k]+proper_noun[k]+head_match[k] + threshold
    #print(total_score.values()) 
    #print()
    
    
    sumValues = 0
    for sentence in total_score:
        sumValues += total_score[sentence]
    average = int(sumValues / len(total_score))
    # Storing sentences into our summary.
    summary = ' '
    for sentence in sent_tokens:
        if (sentence in total_score) and (total_score[sentence] > (1.5*average)):
            summary += " " + sentence
    print(summary)