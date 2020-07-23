import itertools
import json
import math
import sys
import time
from pyspark import SparkContext


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def pearson_similarity_function(x, y):
    co_rated_user = list(set(x.keys()) & (set(y.keys())))
    co_rated_users_length = len(co_rated_user)
    stars_sum1 = 0
    stars_sum2 = 0
    for u in co_rated_user:
        stars_sum1 += x[u]
        stars_sum2 += y[u]
    avg_stars1 = stars_sum1 / co_rated_users_length
    avg_stars2 = stars_sum2 / co_rated_users_length

    num = 0
    den1 = 0
    den2 = 0
    for u in co_rated_user:
        num += (x[u] - avg_stars1) * (y[u] - avg_stars2)
        den1 += (x[u] - avg_stars1) ** 2
        den2 += (y[u] - avg_stars2) ** 2
    den = math.sqrt(den1) * math.sqrt(den2)

    if den != 0 and num != 0:
        return num / den
    else:
        return 1


def minhash_function(list_of_business, business_hash_lookup):
    min_hash_dict = {}
    for i, u in enumerate(list_of_business):
        if i != 0:
            l = business_hash_lookup[u]
            min_hash_dict = {k: min(v, min_hash_dict[k]) for k, v in l.items()}
        else:
            min_hash_dict = business_hash_lookup[u]
    return min_hash_dict


def LSH_function(list_minhash_values, b):
    length_of_minhash_list = len(list_minhash_values)
    r = math.ceil(length_of_minhash_list / b)
    return [(i, hash(tuple(list_minhash_values[v:v + r]))) for i, v in enumerate(range(0, length_of_minhash_list, r))]


def sim(bus1, bus2):
    if bus1 is not None and bus2 is not None:
        b1 = set(bus1.keys())
        b2 = set(bus2.keys())
        return len(b1.intersection(b2)) / len(b1.union(b2))


def write_similar_pairs_function(output_data, type, index_lookup):
    output = []
    app = output.append
    for data in output_data:
        if type == "business":
            app({"b1": index_lookup[data[0][0]], "b2": index_lookup[data[0][1]], "sim": data[1]})
        elif type == "user":
            app({"u1": index_lookup[data[0][0]], "u2": index_lookup[data[0][1]], "sim": data[1]})
    return output


def main():
    if len(sys.argv) != 4:
        print("Please enter all the required agruments")
        exit(-1)

    start = time.time()
    no_of_hash_functions = 30
    no_of_bands = 30

    sc = SparkContext('local[*]', 'task3train')

    train_review_rdd = sc.textFile(str(sys.argv[1])).map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], x["user_id"], x["stars"])).persist()

    business_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    business_id_index_lookup = business_id_index_lookup_rdd.collectAsMap()
    index_business_id_lookup = business_id_index_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()
    no_of_distinct_business = len(business_id_index_lookup)

    user_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[1]).distinct().zipWithIndex()
    user_id_index_lookup = user_id_index_lookup_rdd.collectAsMap()
    index_user_id_lookup = user_id_index_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()
    # no_of_distinct_users = len(user_id_index_lookup)

    if str(sys.argv[3]) == "item_based":
        business_user_star = train_review_rdd \
            .map(lambda x: (business_id_index_lookup[x[0]], (user_id_index_lookup[x[1]], x[2]))) \
            .combineByKey(to_list, append, extend).filter(lambda x: len(x[1]) > 2).persist()

        business_dict = business_user_star.mapValues(lambda x: dict(x)).collectAsMap()
        business_candidates = business_user_star.map(lambda x: x[0]).collect()

        candidate_pairs = [x for x in itertools.combinations(business_candidates, 2)]

        similar_candidate_pairs = sc.parallelize(candidate_pairs) \
            .filter(lambda x: len(set(business_dict[x[0]].keys()) & set(business_dict[x[1]].keys())) > 2) \
            .map(lambda x: (x, pearson_similarity_function(business_dict[x[0]], business_dict[x[1]]))) \
            .filter(lambda x: x[1] > 0).collect()

        output = write_similar_pairs_function(similar_candidate_pairs, "business", index_business_id_lookup)

        with open(str(sys.argv[2]), "w+") as fileout:
            for out in output:
                fileout.writelines(json.dumps(out) + "\n")

    elif str(sys.argv[3]) == "user_based":

        user_business_star = train_review_rdd \
            .map(lambda x: (user_id_index_lookup[x[1]], (business_id_index_lookup[x[0]], x[2]))) \
            .combineByKey(to_list, append, extend).mapValues(lambda x: list(set(x))).filter(lambda x: len(x[1]) > 2) \
            .mapValues(lambda x: dict(x)).persist()

        user_dict = user_business_star.collectAsMap()

        hash_functions_per_business = business_id_index_lookup_rdd \
            .map(lambda x: (x[1],
                            {i: (((199933 + i) * x[1] + (115249 * i)) % 1000000007) % (no_of_distinct_business * 2) for i in
                             range(1, no_of_hash_functions + 1)})).collectAsMap()

        minhash_values_per_user = user_business_star.map(
            lambda x: (x[0], minhash_function(x[1].keys(), hash_functions_per_business))).mapValues(
            lambda x: list(x.values()))

        candidate_pairs_rdd = minhash_values_per_user.flatMap(
            lambda x: [(hashed_rows, x[0]) for hashed_rows in LSH_function(x[1], no_of_bands)]) \
            .combineByKey(to_list, append, extend).values().filter(lambda x: len(x) > 1).flatMap(
            lambda x: [pair for pair in itertools.combinations(x, 2)]).distinct()

        similar_candidate_pairs = candidate_pairs_rdd \
            .filter(lambda x: len(set(user_dict[x[0]].keys()) & set(user_dict[x[1]].keys())) > 2) \
            .map(lambda x: (x, sim(user_dict.get(x[0]), user_dict.get(x[1])))) \
            .filter(lambda x: x[1] > 0.01).keys() \
            .map(lambda x: (x, pearson_similarity_function(user_dict.get(x[0]), user_dict.get(x[1])))) \
            .filter(lambda x: x[1] > 0).collect()

        output = write_similar_pairs_function(similar_candidate_pairs, "user", index_user_id_lookup)

        with open(str(sys.argv[2]), "w+") as fileout:
            for out in output:
                fileout.writelines(json.dumps(out) + "\n")

    sc.stop()

    end = time.time()
    print("Duration:", end - start)


if __name__ == "__main__": main()
