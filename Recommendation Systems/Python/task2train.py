import json
from string import punctuation, digits
from pyspark import SparkContext
import sys
import time
from collections import defaultdict
from math import log



def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def tf_preprocessing(words):
    tf_dict = defaultdict(int)
    for w in words:
        tf_dict[w] += 1

    max_count = max(tf_dict.values())

    for word, count in tf_dict.items():
        tf_dict[word] = count / max_count

    return list(tf_dict.items())


def idf_computing_function(words):
    idf_dict = defaultdict(int)
    for w in words:
        idf_dict[w] += 1
    for w, v in idf_dict.items():
        idf_dict[w] = log(10253 / v, 2)
    return idf_dict


def list_extend_function(x, y):
    output = []
    ext = output.extend
    ext(x)
    ext(y)
    return output


def write_output_function(output_data, typ, column):
    output = []
    app = output.append
    for k, v in output_data.items():
        app({
            "type": typ,
            column[0]: k,
            column[1]: v
        })
    return output


def main():
    if len(sys.argv) != 4:
        print("Please enter all the required agruments")
        exit(-1)

    train_review = str(sys.argv[1])
    stopwords = str(sys.argv[3])

    start = time.time()

    sc = SparkContext('local[*]', 'task2train')
    train_review_rdd = sc.textFile(train_review).map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], x["text"], x['user_id'])).persist()

    model = []
    ext = model.extend

    business_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    business_id_index_lookup = business_id_index_lookup_rdd.collectAsMap()
    # index_business_id_lookup = business_id_index_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()

    ext(write_output_function(business_id_index_lookup, "business_id_lookup", ["business_id", "business_index"]))

    user_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[2]).distinct().zipWithIndex()
    user_id_index_lookup = user_id_index_lookup_rdd.collectAsMap()

    ext(write_output_function(user_id_index_lookup, "user_id_lookup", ["user_id", "user_index"]))

    with open(stopwords, 'rt') as stopwords_file:
        stopwords_list = [word.strip() for word in stopwords_file]

    preprocessed_words = train_review_rdd \
        .map(lambda x: (business_id_index_lookup[x[0]],
                        [w.translate(str.maketrans('', '', digits + punctuation)).strip() for w in
                         x[1].lower().split()])) \
        .mapValues(lambda x: [w for w in x if w not in stopwords_list and len(w) > 0]).groupByKey() \
        .flatMapValues(lambda x: x).persist()  # --count = 34,508,045

    # word_count= preprocessed_words.flatMap(lambda x: x[1]).count()

    # words_and_count = preprocessed_words.flatMap(lambda x: [((x[0], w), 1) for w in x[1]]) \
    #     .reduceByKey(lambda x, y: x + y).filter(lambda x: ((x[1] * 100) / 506685) > 0.0001) \
    #     .map(lambda x: x[0]).collect()

    words = preprocessed_words.map(lambda x: x[1]).flatMap(lambda x: list(set(x))).collect()
    # .filter(lambda x: x not in words_and_count and len(x) > 0)

    idf_words_dict = idf_computing_function(words)

    tfidf_words = preprocessed_words \
        .filter(lambda x: len(x[1]) > 0) \
        .map(lambda x: (x[0], tf_preprocessing(x[1]))).flatMap(lambda x: [((x[0], t[0]), t[1]) for t in x[1]]) \
        .map(lambda x: (x[0], x[1], idf_words_dict[x[0][1]])) \
        .map(lambda x: (x[0], x[1] * x[2])).map(lambda x: (x[0][0], (x[0][1], x[1]))) \
        .combineByKey(to_list, append, extend) \
        .mapValues(lambda x: sorted(list(x), key=lambda y: y[1], reverse=True)) \
        .mapValues(lambda x: x[:200]).mapValues(lambda x: [i[0] for i in x])
    # .mapValues(lambda x: [w for w in x if w not in words_and_count and len(w) > 0])

    indexed_words = tfidf_words.flatMap(lambda x: set(x[1])).zipWithIndex().collectAsMap()

    business_profile = tfidf_words.mapValues(lambda x: [indexed_words[i] for i in x]).collectAsMap()

    ext(write_output_function(business_profile, "business_profile", ["business_index", "business_profile"]))

    user_profile_list = train_review_rdd.map(lambda x: (x[2], x[0])).combineByKey(to_list, append, extend) \
        .map(lambda x: (user_id_index_lookup[x[0]], [business_id_index_lookup[u] for u in set(x[1])])) \
        .flatMapValues(lambda x: [business_profile[i] for i in x]).reduceByKey(list_extend_function) \
        .filter(lambda x: len(x[1]) > 1).map(lambda x: (x[0], list(set(x[1])))).collectAsMap()

    ext(write_output_function(user_profile_list, "user_profile", ["user_index", "user_profile"]))

    with open(str(sys.argv[2]), "w+") as fileout:
        for m in model:
            fileout.writelines(json.dumps(m) + "\n")

    sc.stop()

    end = time.time()
    print("Duration:", end - start)


if __name__ == "__main__": main()
