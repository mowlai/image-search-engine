import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from preprocess import *

all_captions = pd.read_csv("/content/gdrive/My Drive/all_captions.csv", sep=',') 

real_captions = all_captions['true_caption']
pred_captions = all_captions['pred_caption']

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def get_similar_result_jaccard(idx, real_captions, pred_captions):
    
    s_score_list = []

    for idx_2 in range(len(pred_captions)):
      
      s_score = jaccard_similarity(real_captions[idx], pred_captions[idx_2])
      s_score_list.append((idx_2, s_score))

    s_score_list.sort(key=lambda x: x[1], reverse=True)

    return s_score_list

def show_image(image_fname, new_figure=True):
    if new_figure:
        plt.figure()
    np_img = cv2.imread(image_fname)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    plt.imshow(np_img) 

def show_qualitative_results(idx1, top_k=20):

    b_score_res = get_similar_result_jaccard(idx1, real_captions, pred_captions)

    print("Real capt:", ' '.join(real_captions[idx1]))
    print("Pred capt:", ' '.join(pred_captions[idx1]))
    sentence1 = [w for w in real_captions[idx1] if not w in stop_words]
    sentence2 = [w for w in pred_captions[idx1] if not w in stop_words]
    ss = jaccard_similarity(sentence1, sentence2)
    print("Score with True Predicted caption:", ss)
    print()

    show_image(img_name_val[idx1], new_figure=False)
    plt.grid(False)
    plt.ioff()
    plt.axis('off')


    fig = plt.figure(figsize=(10, 7))

    for idx2, (idx, sim_val) in enumerate(b_score_res[:20]):
        print(idx, sim_val, ' '.join(pred_captions[idx]))
        plt.subplot(4, 5, idx2+1)
        show_image(img_name_val[idx], new_figure=False)
        plt.grid(False)
        plt.ioff()
        plt.axis('off')
        plt.title('{}'.format(idx2+1))

show_qualitative_results(idx1 = 0)

all_idx = []
top_k = 1000

for ref_idx in tqdm(range(len(img_name_val))):
    s_score_res = get_similar_result_jaccard(ref_idx, real_captions, pred_captions)
    list_res = list(map(lambda x: x[0], s_score_res[:top_k]))
    index = list_res.index(ref_idx)
    all_idx.append(index)

n, bins, patches = plt.hist(all_idx, bins=1000)
plt.xlabel('top K')
plt.ylabel('Frequency')

plt.show()