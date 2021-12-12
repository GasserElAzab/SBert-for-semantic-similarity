from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer('all-MiniLM-L6-v2')
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>> print embeddings for sentences <<<<<<<<<<<<<<<<<< ")

# #Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.', 
#     'The quick brown fox jumps over the lazy dog.']

# #Sentences are encoded by calling model.encode()
# sentence_embeddings = model.encode(sentences)

# #Print the embeddings
# # for sentence, embedding in zip(sentences, sentence_embeddings):
# #     print("Sentence:", sentence)
# #     print("Embedding:", embedding)
# #     print("")

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>> cosine similarity between 2 sentences only <<<<<<<<<<<<<<<<<< ")

# model = SentenceTransformer('all-MiniLM-L6-v2')

# #Sentences are encoded by calling model.encode()
# emb1 = model.encode("This is a red cat with a hat.")
# emb2 = model.encode("Have you seen my white dog?")

# cos_sim = util.cos_sim(emb1, emb2)
# # print("Cosine-Similarity:", cos_sim)

# #Add all pairs to a list with their cosine similarity score
# ##  Similarity scores between different sentences <<<<<<<<<<< START >>>>>>>>>>>>>>>>>
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Similarity scorees between different sentences <<<<<<<<<<<<<<<<<< ")

# sentences = ['A man is eating food.',
#           'A man is eating a piece of bread.',
#           'The girl is carrying a baby.',
#           'A man is riding a horse.',
#           'A woman is playing violin.',
#           'Two men pushed carts through the woods.',
#           'A man is riding a white horse on an enclosed ground.',
#           'A monkey is playing drums.',
#           'Someone in a gorilla costume is playing a set of drums.'
#           ]

# embeddings = model.encode(sentences)

# cos_sim = util.cos_sim(embeddings, embeddings)

# all_sentence_combinations = []
# for i in range(len(cos_sim)-1):
#     for j in range(i+1, len(cos_sim)):
#         all_sentence_combinations.append([cos_sim[i][j], i, j])

# #Sort list by the highest cosine similarity score
# all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

# print("Top-5 most similar pairs:")
# for score, i, j in all_sentence_combinations[0:5]:
#     print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))


# #  Similarity scores between different sentences <<<<<<<<<<< ENDD >>>>>>>>>>>>>>>>>


print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Start of the lab (as recommended) <<<<<<<<<<<<<<<<<< ")

from transformers import BertTokenizer, BertModel
from csv import QUOTE_NONE
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
# import tensorflow_datasets as tfds


#This is a pre trained bert model, but not fine tuned as the professor said

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text,return_tensors='pt')
# output = model(**encoded_input)
# # print(output)

#load datas

sts_train=pd.read_csv("stsbenchmark/sts-train.csv",sep='\t',header=None,usecols=[4, 5, 6], quoting=QUOTE_NONE,names=["label","sen1","sen2"])
sts_test=pd.read_csv("stsbenchmark/sts-test.csv",sep='\t',header=None,usecols=[4, 5, 6], quoting=QUOTE_NONE,names=["label","sen1","sen2"])
sts_dev=pd.read_csv("stsbenchmark/sts-dev.csv",sep='\t',header=None,usecols=[4, 5, 6], quoting=QUOTE_NONE,names=["label","sen1","sen2"])
print("loading data ok")
print("starting tokenizing the data")

train_sen1_encoding= tokenizer(list(sts_train["sen1"]), padding="max_length", truncation=True)
train_sen2_encoding= tokenizer(list(sts_train["sen2"]), padding="max_length", truncation=True)
train_labels=list(sts_train["label"])


test_sen1_encoding= tokenizer(list(sts_test["sen1"]), padding="max_length", truncation=True)
test_sen2_encoding= tokenizer(list(sts_test["sen2"]), padding="max_length", truncation=True)
test_labels=list(sts_test["label"])

dev_sen1_encoding= tokenizer(list(sts_dev["sen1"]), padding="max_length", truncation=True)
dev_sen2_encoding= tokenizer(list(sts_dev["sen2"]), padding="max_length", truncation=True)
dev_labels=list(sts_dev["label"])

print("End of tokenizing the data")
print(train_sen1_encoding)




# Regression training objective
# The normal bert should be tuned using STS dataset, cosine similiarty, and mapping using the provided table




#Classification objective (using the same embedding from the first task)
# using NLI dataset


#Combination --> training on NLI then tuning by STS



#Evaluation
# 1. Simliarty of spearmean correleation

# 2. Semantic search to make a file that allowe user to enter a string and return K similar

