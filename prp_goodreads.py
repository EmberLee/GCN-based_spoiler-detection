import numpy as np
import nltk
import re; import json
from tqdm import tqdm
import pickle

from dependency_parsing import *

'''
# books : 25475, # users : 18892
# reviews : 1378033, # reviews containing spoiler : 9M,
# sentences : 1700M, # spoiler sentences : 57M
## keys() : user_id, timestamp, review_sentences, rating, has_spoiler, book_id, review_id
'''

def make_graph(num_workers):
    data_path = './datas/goodreads_data/'
    data_name = 'goodreads_reviews_spoiler.json'

    print('Data loading...')
    data = [json.loads(line) for line in open(data_path + data_name, 'r')]
    print('Loading finished')

    lnt = len(data)
    pool = prpExecutor(max_workers=num_workers)

    graph_data = list(pool.map(context_to_tree_goodreads, tqdm(data)))

    # try:
    #     json.dump(graph_data, open(data_path + 'graph_goodreads.json', 'w'))
    # except:
    #     pickle.dump(graph_data, open(data_path + 'graph_goodreads.pickle', 'wb'))
    pickle.dump(graph_data, open(data_path + 'node_edge_info.pickle', 'wb'))

def extracter(num_workers):
    data_path = './datas/goodreads_data/'
    data_name = 'goodreads_reviews_spoiler.json'

    print('Data loading...')
    data = [json.loads(line) for line in open(data_path + data_name, 'r')]
    print('Loading finished')

    pool = prpExecutor(max_workers=num_workers)

    info_data = list(pool.map(context_to_tree_goodreads, tqdm(data[:10000])))

    try:
        json.dump(info_data, open(data_path + 'node_edge_info_10000.json', 'w'))
    except:
        pickle.dump(info_data, open(data_path + 'node_edge_info_10000.pickle', 'wb'))

if __name__ == '__main__':
    # make_graph(10)
    extracter(10)
