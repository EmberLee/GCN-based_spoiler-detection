import numpy as np
import nltk
import re; import json
from tqdm import tqdm
import pickle
import time
import os
from concurrent.futures import ProcessPoolExecutor as prpExecutor

def extract_info(graphs, is_directional=False):
    result = {}
    # graphs = ith_data['graph']
    (result['node'], result['adj']) = ([[] for _ in range(len(graphs))] for _ in range(2))
    result['edge'] = [{} for _ in range(len(graphs))]

    BRACKET_REGEX = re.compile(r'\(.*?\)')
    BIGQUOTE_REGEX = re.compile(r'\".*?\"')

    for i, ith_list in enumerate(graphs):
        graph = ith_list[1]
        result['edge'][i]['edge_attribute'] = []
        result['edge'][i]['edge_coord'] = []

        for each_info in graph[4:-1]:
            if '->' not in each_info:
                node = BRACKET_REGEX.findall(each_info)[0][1:-1] # containing empty node above ROOT : 'None'
                result['node'][i].append(node)

            else:
                ## edge label part
                edge_label = BIGQUOTE_REGEX.findall(each_info)[0][1:-1]
                result['edge'][i]['edge_attribute'].append(edge_label)

                ## edge tuple part
                end_idx = each_info.index('[label')
                content = each_info[:end_idx-1]
                middle_idx = content.index('->')
                start_node = int(content[:middle_idx-1])
                end_node = int(content[middle_idx+3:])

                edge_tuple = (start_node, end_node)
                result['edge'][i]['edge_coord'].append(edge_tuple)

        if not is_directional:
            ## init adjacency matrix
            num_nodes = len(result['node'][i]) - 1
            result['adj'][i] = np.zeros((num_nodes, num_nodes))

            for jth_coord in result['edge'][i]['edge_coord']:
                (start, end) = jth_coord
                if start == 0 or end == 0:
                    continue
                else:
                    result['adj'][i][start-1, end-1] = 1.0
                    result['adj'][i][end-1, start-1] = 1.0

    return result

def extract_info_buru(graphs):
    result = {}
    # graphs = ith_data['graph']
    result['node'] = [[] for _ in range(len(graphs))]
    result['edge'] = [{} for _ in range(len(graphs))]
    result['text_info'] = ['' for _ in range(len(graphs))]

    BRACKET_REGEX = re.compile(r'\(.*?\)')
    BIGQUOTE_REGEX = re.compile(r'\".*?\"')

    for i, ith_list in enumerate(graphs):
        graph = ith_list[1]
        result['edge'][i]['edge_attribute'] = []
        result['edge'][i]['edge_coord'] = []

        for each_info in graph[4:-1]:
            if '->' not in each_info:
                node = BRACKET_REGEX.findall(each_info)[0][1:-1] # containing empty node above ROOT : 'None'
                result['node'][i].append(node)

            else:
                ## edge label part
                edge_label = BIGQUOTE_REGEX.findall(each_info)[0][1:-1]
                result['edge'][i]['edge_attribute'].append(edge_label)

                ## edge tuple part
                end_idx = each_info.index('[label')
                content = each_info[:end_idx-1]
                try:
                    middle_idx = content.index('->')
                except:
                    continue
                start_node = int(content[:middle_idx-1])
                end_node = int(content[middle_idx+3:])

                edge_tuple = (start_node, end_node)
                result['edge'][i]['edge_coord'].append(edge_tuple)

        result['node'][i][0] = '<ROOT>'
        result['text_info'][i] = ' '.join(result['node'][i])
        result['text_info'][i] += '\t'

        content = []
        for k in range(len(result['edge'][i]['edge_coord'])):
            start, end = result['edge'][i]['edge_coord'][k]
            start_end = [str(start), str(end)]
            attribute = result['edge'][i]['edge_attribute'][k]
            start_end.append(attribute)
            content.append(':'.join(start_end))
        result['text_info'][i] += '\t'.join(content)

    real_result = {}
    real_result['text_info'] = result['text_info']
    return real_result

def extracter(num_workers):
    data_path = './datas/goodreads_data/'
    data_name = 'graph_goodreads.json'

    print('Data loading...')
    data = json.load(open(data_path + data_name, 'r'))
    print('Loading finished')

    pool = prpExecutor(max_workers=num_workers)

    info_data = list(pool.map(extract_info_buru, tqdm(data)))

    try:
        json.dump(info_data, open(data_path + 'node_edge_info.json', 'w'))
    except:
        pickle.dump(info_data, open(data_path + 'node_edge_info.pickle', 'wb'))

''' goodreads_reviews_spoiler + node_edge_info '''
def combine_original(original_ith, post_ith):
    result = {}
    content = []
    original = original_ith['review_sentences']
    post = post_ith['text_info']
    for i, info in enumerate(original):
        is_spoiler = str(info[0])
        pieces = post[i].split('\t')
        ith_content = pieces[0] + '\t' +  is_spoiler + '\t' + '\t'.join(pieces[1:])
        content.append(ith_content)

    result['text_info'] = content
    return result

def combiner(num_workers):
    data_path = './datas/goodreads_data/'

    print('Data loading...')
    data1 = [json.loads(line) for line in open(data_path + 'goodreads_reviews_spoiler.json', 'r')]
    # data1 = data1[:10000]
    data2 = json.load(open(data_path + 'node_edge_info.json', 'r'))
    print('Loading finished')

    pool = prpExecutor(max_workers=num_workers)

    info_data = list(pool.map(combine_original, tqdm(data1), data2))

    try:
        json.dump(info_data, open(data_path + 'node_edge_info_new.json', 'w'))
    except:
        pickle.dump(info_data, open(data_path + 'node_edge_info_new.pickle', 'wb'))

if __name__ == '__main__':
    combiner(10)
