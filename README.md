# image-search-engine
<h3>Google Training Camp at the Sapienza University of Rome on building an image search engine</h3>
Abstract: <br/>
How could we build an automated system that can find a photograph in a family album or an online photo collection given just a textual description? <br/>
In this training camp I have covered fundamental techniques in computer vision and natural language processing that will help us to address this question. The main aim of the camp was be to enable students to build their own prototype of the image search engine, and participate in online Kaggle competition organized for the course participants. <br/>

Specific techniques that the we learned and implemented:
- **image representation with convolutional neural networks (CNNs)**
- **building recurrent neural networks with LSTM and GRU units**
- **generating natural language image descriptions (=image captioning)**
- **representing words and sentences with vector embeddings (Word2Vec, GloVe, and BERT)** <br/>

Main goal was to build an image search engine on COCO dataset in two steps:
1. Train a model for image captioning.
2. Build a similarity function for generated queries.


The baseline (InceptionV3 and Jaccard-Distance; score 0.06597) could be found here https://github.com/SapienzaTrainingCamp/GoogleTrainingCamp

The pretrained InceptionV3 and pretrained Glove 6B with Cosine Similarity has been implemented. (score 0.13732)
