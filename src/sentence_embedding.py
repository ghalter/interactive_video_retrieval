'''
FindSimiliarities(data, args):
--> input-format
    - query:                is the search query. Must be a list in the following format
                            -> query = ['your query is here']
    - caption_embeddings:   are the preprocessed caption-embeddings, which need to be loaded
    - captions, img_ids:    are the original captions and image_ids from captions.csv
                            -> should be handed to the function as lists with the values
    - model:                pass the pre-loaded model, i.e. the SentenceTransformer to the function
    - number_top_matches:   defines the how many best-matches are returned
    
--> returns
    - best_matches:         are the top-n matches in the format (img_ids, cosine_similarities)
    - results:              are all images and their corresponding cosine_similiarities sorted by
                            the highest cosine similarities

'''

import scipy
import numpy as np

from sentence_transformers import SentenceTransformer
import os
import pickle
import pandas as pd


# Load Model
_embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Load caption embeddings using pickle
dir_path = "data"
embeddings_path = dir_path + '/embedding.pickle'
file = open(embeddings_path, 'rb')
_caption_embeddings = pickle.load(file)
file.close()



# Load the caption and img_ids from the captions.csv file
csv_path = dir_path + '/captions.csv'
csv = pd.read_csv(csv_path)
_CAPTIONS = csv['caption'].tolist()
_IMAGE_IDS = csv['thumbnail_id'].tolist()

print(len(_caption_embeddings))
# Define the query
query = ['a group of people']


def find_similarity(query, captions, entries, number_top_matches = 100):
    
    query = [query]
    
    # Get the embedding of the query using SiameseBERT Networks
    query_embedding = _embedding_model.encode(query)
    
    # Get the distances for the query and save them into distances
    distances = list()
    
    for idx, caption in enumerate(_caption_embeddings):
        caption = np.expand_dims(caption, axis=0)
        distance = scipy.spatial.distance.cdist(caption, query_embedding, "cosine")
        distances.append(distance)
        
    # Combine cosine distances and ids using zip (change to ids later!!!)
    results = zip(captions, entries, distances)
    results = sorted(results, key = lambda x: x[2])
    
    # Retrieve best_matching results
    best_matches = list()
    
    for caption, entry, distance in results[0:number_top_matches]:
        # best_matches.append([entry, (1-distance)])
        best_matches.append(entry)

    return best_matches, results





# Get the best matching captions
# best_matches, results = FindSimiliarities(query,
#                                          caption_embeddings,
#                                          captions,
#                                          img_ids,
#                                          embedding_model
#                                          )

# print(best_matches)