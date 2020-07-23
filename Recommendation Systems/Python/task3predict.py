from collections import Counter
from itertools import product
import json
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


def predict_rating_function(test_data, similar_data, trained_data, avg_rating_dict, type):
    avg_stars = 3.823989
    if type == "item_based":
        other_business = list(similar_data.keys())
        pairs_dict = {pair[1]: trained_data.get(tuple(sorted(pair))) for pair in product([test_data], other_business) if
                      trained_data.get(tuple(sorted(pair))) is not None}
        num = 0
        den = 0
        top3neighbors_dict = Counter(pairs_dict)
        for k, v in top3neighbors_dict.most_common(3):
            w = v * (abs(v) ** 1.5)
            num += similar_data[k] * w
            den += w
        if num != 0 and den != 0:
            return num / den
        else:
            return avg_stars

    elif type == "user_based":
        test_user_avg_rating = avg_rating_dict.get(test_data, avg_stars)
        other_users = list(similar_data.keys())
        pairs_dict = {pair[1]: trained_data.get(tuple(sorted(pair))) for pair in product([test_data], other_users) if
                      trained_data.get(tuple(sorted(pair))) is not None}
        num = 0
        den = 0
        avg = 0
        top3neighbors_dict = Counter(pairs_dict)
        for k, v in top3neighbors_dict.most_common(3):
            avg += similar_data[k]
        avg = avg/3

        for k, v in top3neighbors_dict.most_common(3):
            w = v*(abs(v)**1.5)
            num += (similar_data[k] - avg) * w
            den += w

        if num != 0 and den != 0:
            return test_user_avg_rating + (num / den)
        else:
            return avg_stars


def write_predicted_rating_function(output_data):
    return [{"user_id": data[0][0], "business_id": data[0][1], "stars": data[1]} for data in output_data]


def main():
    if len(sys.argv) != 8:
        print("Please enter all the required agruments")
        exit(-1)

    start = time.time()

    train_review_json = str(sys.argv[1])
    test_review_json = str(sys.argv[2])
    trained_data_json = str(sys.argv[3])
    output_json = str(sys.argv[4])
    item_based_or_user_based = str(sys.argv[5])
    business_avg_rating_json = str(sys.argv[6])
    user_avg_rating_json = str(sys.argv[7])

    sc = SparkContext('local[*]', 'task3predict')

    train_review_rdd = sc.textFile(train_review_json).map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], x["user_id"], x["stars"])).persist()

    business_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[0]).distinct().zipWithIndex()
    business_id_index_lookup = business_id_index_lookup_rdd.collectAsMap()
    index_business_id_lookup = business_id_index_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()

    user_id_index_lookup_rdd = train_review_rdd.map(lambda x: x[1]).distinct().zipWithIndex()
    user_id_index_lookup = user_id_index_lookup_rdd.collectAsMap()
    index_user_id_lookup = user_id_index_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()

    test_data_rdd = sc.textFile(test_review_json).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["user_id"], x["business_id"])) \
        .map(lambda x: (user_id_index_lookup.get(x[0], None), business_id_index_lookup.get(x[1], None)))

    if item_based_or_user_based == "item_based":
        trained_data = sc.textFile(trained_data_json) \
            .map(lambda x: json.loads(x)) \
            .map(lambda x: ((business_id_index_lookup[x["b1"]], business_id_index_lookup[x["b2"]]), x["sim"])) \
            .map(lambda x: (tuple(sorted(x[0])), x[1])) \
            .collectAsMap()

        business_avg_rating_dict = sc.textFile(business_avg_rating_json) \
            .map(lambda x: json.loads(x)).flatMap(lambda x: x.items()) \
            .filter(lambda x: x[0] not in "UNK") \
            .map(lambda x: (business_id_index_lookup[x[0]], x[1])) \
            .collectAsMap()

        user_business_star = train_review_rdd \
            .map(lambda x: (user_id_index_lookup[x[1]], (business_id_index_lookup[x[0]], x[2]))) \
            .combineByKey(to_list, append, extend).mapValues(lambda x: list(set(x))) \
            .mapValues(lambda x: dict(x)).collectAsMap()

        prediction = test_data_rdd \
            .map(
            lambda x: (x, predict_rating_function(x[1], user_business_star[x[0]], trained_data, business_avg_rating_dict,"item_based") if x[1] is not None and x[0] is not None else 3.823989)) \
            .map(lambda x: ((index_user_id_lookup.get(x[0][0]), index_business_id_lookup.get(x[0][1])), x[1])).collect()

        output = write_predicted_rating_function(prediction)

        with open(output_json, "w+") as fileout:
            for out in output:
                fileout.writelines(json.dumps(out) + "\n")

    elif item_based_or_user_based == "user_based":
        trained_data = sc.textFile(trained_data_json) \
            .map(lambda x: json.loads(x)) \
            .map(lambda x: ((user_id_index_lookup[x["u1"]], user_id_index_lookup[x["u2"]]), x["sim"])) \
            .map(lambda x: (tuple(sorted(x[0])), x[1])) \
            .collectAsMap()

        user_avg_rating_dict = sc.textFile(user_avg_rating_json) \
            .map(lambda x: json.loads(x)).flatMap(lambda x: x.items()) \
            .filter(lambda x: x[0] not in "UNK") \
            .map(lambda x: (user_id_index_lookup[x[0]], x[1])) \
            .collectAsMap()

        business_user_star = train_review_rdd \
            .map(lambda x: (business_id_index_lookup[x[0]], (user_id_index_lookup[x[1]], x[2]))) \
            .combineByKey(to_list, append, extend).mapValues(lambda x: list(set(x))) \
            .mapValues(lambda x: dict(x)).collectAsMap()

        prediction = test_data_rdd \
            .map(
            lambda x: (x, predict_rating_function(x[0], business_user_star[x[1]], trained_data, user_avg_rating_dict,
                                                  "user-based") if x[1] is not None and x[0] is not None else 3.823989)) \
            .map(lambda x: ((index_user_id_lookup.get(x[0][0]), index_business_id_lookup.get(x[0][1])), x[1])).collect()

        output = write_predicted_rating_function(prediction)

        with open(output_json, "w+") as fileout:
            for out in output:
                fileout.writelines(json.dumps(out) + "\n")

    sc.stop()

    end = time.time()
    print("Duration:", end - start)


if __name__ == "__main__": main()
