import itertools
import json
import math
import sys
from pyspark import SparkContext
import time


def minhash_function(list_of_users, user_hash_lookup):
    min_hash_dict = {}
    for i, u in enumerate(list_of_users):
        if i != 0:
            l = user_hash_lookup[u]
            min_hash_dict = {k: min(v, min_hash_dict[k]) for k, v in l.items()}
        else:
            min_hash_dict = user_hash_lookup[u]
    return min_hash_dict


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def LSH_function(list_minhash_values, b):
    length_of_minhash_list = len(list_minhash_values)
    r = math.ceil(length_of_minhash_list / b)
    return [(i, hash(tuple(list_minhash_values[v:v + r]))) for i, v in enumerate(range(0, length_of_minhash_list, r))]


def sim(bus1_users, bus2_users):
    b1 = set(bus1_users)
    b2 = set(bus2_users)
    return len(b1.intersection(b2)) / len(b1.union(b2))


def similar_pairs_function(upb, ibl, cps, st):
    output = []
    app = output.append
    for cp in cps:
        js = sim(upb[cp[0]], upb[cp[1]])
        if js >= st:
            app({"b1": ibl[cp[0]], "b2": ibl[cp[1]], "sim": js})
    return output


def main():
    if len(sys.argv) != 3:
        print("Please enter all the required agruments")
        exit(-1)

    start = time.time()

    sc = SparkContext('local[*]', 'task1')
    no_of_hash_functions = 30
    no_of_bands = 30

    train_review_rdd = sc.textFile(str(sys.argv[1])).map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], x["user_id"])).persist()

    business_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    business_id_index_lookup = business_id_index_lookup_rdd.collectAsMap()
    index_business_id_lookup = business_id_index_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()

    user_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[1]).distinct().zipWithIndex()
    user_id_index_lookup = user_id_index_lookup_rdd.collectAsMap()
    index_user_id_lookup = user_id_index_lookup_rdd.map(lambda x: (x[1], x[0]))
    no_of_distinct_users = len(user_id_index_lookup)

    # characteristic matrix
    users_per_business = train_review_rdd.combineByKey(to_list, append, extend).map(
        lambda x: (x[0], list(set(x[1])))) \
        .map(lambda x: (business_id_index_lookup[x[0]], [user_id_index_lookup[u] for u in x[1]]))

    users_per_business_dict = users_per_business.collectAsMap()

    # user-hashfunction table
    hash_functions_per_user = user_id_index_lookup_rdd.map(lambda x: (x[1], {i:
        (((3583 + i) * x[1] + (4297 * i)) % 939193) % (no_of_distinct_users * 2) for i in
        range(1, no_of_hash_functions + 1)})).collectAsMap()

    # signature matrix
    minhash_values_per_business = users_per_business.map(
        lambda x: (x[0], minhash_function(x[1], hash_functions_per_user))).mapValues(lambda x: list(x.values()))

    # dividing signature matrix into bands

    candidate_pairs = set(minhash_values_per_business.flatMap(
        lambda x: [(hashed_rows, x[0]) for hashed_rows in LSH_function(x[1], no_of_bands)]) \
                          .combineByKey(to_list, append, extend).values().filter(lambda x: len(x) > 1).flatMap(
        lambda x: [pair for pair in itertools.combinations(x, 2)]).collect())
    print("length of candidate pairs", len(candidate_pairs))

    similar_pairs = similar_pairs_function(users_per_business_dict, index_business_id_lookup, candidate_pairs, 0.05)
    print("length of similar pair", len(similar_pairs))

    with open(str(sys.argv[2]), "w+") as fileout:
        for pair in similar_pairs:
            fileout.writelines(json.dumps(pair) + "\n")

    sc.stop()

    end = time.time()
    print("Duration:", end - start)


if __name__ == "__main__": main()
