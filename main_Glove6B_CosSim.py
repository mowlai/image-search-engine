import tensorflow as tf
tf.enable_v2_behavior()
import re
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import csv
import pandas as pd
from nltk.corpus import stopwords
import nltk
import re

from preprocess import *

all_captions = pd.read_csv("/content/gdrive/My Drive/all_captions.csv", sep=',') 

real_captions = all_captions['true_caption']
pred_captions = all_captions['pred_caption']

nltk.download('stopwords')
nltk.download('wordnet')

# Removing text from stopwords, lowering and 

def cleaning(data):
    no_punct=re.sub('[^a-zA-Z]', ' ',data)
    lower=str.lower(no_punct).split()
    
    stop_words=set(stopwords.words('english'))
    clean = [word for word in lower if not word in stop_words]
    
    return ( " ".join(clean)) 


from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.300d.txt'
word2vec_output_file = 'word2vec.txt'
# The first step is to convert the GloVe file format to the word2vec file format. 
# The only difference is the addition of a small header line. This can be done by calling the 
# glove2word2vec() function.

glove2word2vec(glove_input_file, word2vec_output_file)

#Loading vectors
embeddings_index = {}
with open('glove.6B.300d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

from gensim.models import KeyedVectors
glove_input_file = 'glove.6B.300d.txt'
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

#Setting up vocabulary of the model
w2v_vocab = set(model.vocab)

print("Loaded {} words in vocabulary".format(len(w2v_vocab)))

#Cleaning predicted sentences
prediction = all_captions['pred_caption']
for i in range(len(prediction)):
  prediction[i]=cleaning(prediction[i])


def count_distance(target_sentence, w2v_vocab = w2v_vocab,prediction=prediction):
    target_sentence = cleaning(target_sentence)
    # use n_similarity to compute a cosine similarity (should be reasonably robust)
    sentences_similarity = np.zeros(len(prediction))
    target_sentence_words = [w for w in target_sentence.split() if w in w2v_vocab]
    for idx, sentence in enumerate(prediction):
        sentence_words = [w for w in sentence.split() if w in w2v_vocab]
        sim = model.n_similarity(target_sentence_words, sentence_words)
        sentences_similarity[idx] = sim
    result = list(zip(sentences_similarity, prediction))
    
    print("Target:", target_sentence)
    return result

def create_submission_file(top_k, img_name_val, real_captions, pred_captions):
    
    with open('./submission_glove.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["caption", "image_list"])

        for idx in tqdm(range(len(img_name_val))):
            result = count_distance(real_captions[idx])
            scores_id = sorted(range(len(result)), key=lambda k: result[k],reverse=True)
            result = result.sort(key=lambda item:item[0], reverse=True)
            writer.writerow([' '.join(real_captions[idx]), ' '.join(list(map(lambda x: str(x), scores_id[:top_k])))])

create_submission_file(len(img_name_val), img_name_val, real_captions, prediction)

