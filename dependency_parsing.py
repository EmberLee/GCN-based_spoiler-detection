import numpy as np
import nltk
import re; import json
from tqdm import tqdm
import pickle
import time
import os
from concurrent.futures import ProcessPoolExecutor as prpExecutor
from nltk import Tree
from nltk.parse.corenlp import CoreNLPDependencyParser

from preprocessing import *

def main(num_workers):
    data = pickle.load(open('datas/word_level_data.pickle', 'rb'))

    lnt = len(data)
    piece = lnt // 20
    graph_path = './datas/graph_datas/'
    tree_path = './datas/tree_datas/'

    pool = prpExecutor(max_workers=num_workers)

    for i in tqdm(range(0, lnt, piece)):
        data = pickle.load(open(graph_path + 'graph_data{0}.pickle'.format(i//piece + 1), 'rb'))
        graph_data = list(pool.map(context_to_tree, data, list(range(len(data)))))
        pickle.dump(graph_data, open(tree_path + 'tree_graph_data{0}.pickle'.format(i//piece + 1), 'wb'))

def context_to_tree(ith_data, step, to_graph = False):
    start_time = time.time()

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    if to_graph:
        context = ith_data['context']
        graph = [[] for _ in range(len(context))]
    else:
        context = ith_data['context']
        tree = [[] for _ in range(len(context))]
        triple = [[] for _ in range(len(context))]
        # figure = [[] for _ in range(len(context))]

    result = {}

    for i in range(len(context)): ## ith context of input movie(divided in multple sentences)
        if to_graph:
            graph[i] = [[] for _ in range(len(context[i]))]
        else:
            tree[i] = [[] for _ in range(len(context[i]))]
            triple[i] = [[] for _ in range(len(context[i]))]
            # figure[i] = [[] for _ in range(len(context[i]))]

        for j, jth in enumerate(context[i]): ## jth sentence of ith context

            ## Tokenizing PLAN
            if to_graph:
                if jth != '':
                    graph[i][j] = []
                    parsed = dep_parser.raw_parse(jth)
                    for parse in parsed:
                        graph[i][j].append(parse.to_dot())
                    graph[i][j] = graph[i][j][0].split('\n')

                else:
                    graph[i][j] = jth

            else:
                if jth != '':
                    # doc = nlp(jth)
                    # tree[i][j] = doc.sentences[0] ## stanfordnlp
                    tree[i][j], triple[i][j] = [],[]
                    parsed = dep_parser.raw_parse(jth)
                    for parse in parsed:
                        tree[i][j].append(parse.tree())
                        triple[i][j].append(parse.triples())

                    # figure[i][j] = tree[i][j][0].pretty_print()
                    tree[i][j] = list(tree[i][j][0])
                    triple[i][j] = list(triple[i][j][0])

                else:
                    tree[i][j] = jth
                    triple[i][j] = jth
                    # figure[i][j] = jth
            # print("{0}th Movie Processing => ".format(step+1) + 'i & j: {0}/{2}, {1}/{3}'.format(i+1, j+1, len(context), len(context[i])))

    if to_graph:
        ith_data['graph'] = graph
        print("Parsing Runtime: %0.2f Minutes"%((time.time() - start_time)/60))
        return ith_data

    else:
        ith_data['tree'] = tree
        ith_data['triple'] = triple
        # ith_data['figure'] = figure
        # print("Parsing Runtime: %0.2f Minutes"%((time.time() - start_time)/60))
        return ith_data

def context_to_tree_goodreads(ith_data):
    start_time = time.time()

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    context = ith_data['review_sentences']
    graph = [[0, []] for _ in range(len(context))]
    # tree = [[] for _ in range(len(context))]
    # triple = [[] for _ in range(len(context))]

    for i, ith in enumerate(context): ## ith context of input movie(divided in multple sentences)
        if ith[0] == 1:
            graph[i][0] = 1

            ## Tokenizing PLAN
        if ith[1] != '':
            parsed = dep_parser.raw_parse(ith[1])
            for parse in parsed:
                graph[i][1].append(parse.to_dot())

            graph[i][1] = graph[i][1][0].split('\n')

        else:
            graph[i][1] = ith[1]
        # print("{0}th Movie Processing => ".format(step+1) + 'i & j: {0}/{2}, {1}/{3}'.format(i+1, j+1, len(context), len(context[i])))

    # ith_data['graph'] = graph
    result = extract_info_buru(graph)
    return result

if __name__ == '__main__':
    main(10)
