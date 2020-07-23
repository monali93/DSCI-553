import json
from pyspark import SparkContext
import sys
import time
from math import sqrt
import string


def write_output_function(output_data):
    output = []
    app = output.append
    for d in output_data:
        app({
            "user_id": d[0],
            "business_id": d[1],
            "sim": d[2]
        })
    return output


def cosine_similarity_function(up, bp):
    if len(up) is not None and len(bp) is not None:
        num = len(set(up) & set(bp))
        den = sqrt(len(up)) * sqrt(len(bp))
        return num / den
    else:
        return 0


def main():
    if len(sys.argv) != 4:
        print("Please enter all the required agruments")
        exit(-1)

    text_file = str(sys.argv[1])
    model_file = str(sys.argv[2])
    out_file = str(sys.argv[3])

    start = time.time()

    sc = SparkContext('local[*]', 'task2predict')
    model = sc.textFile(model_file).map(lambda x: json.loads(x))

    business_lookup_rdd = model.filter(lambda x: x["type"] == "business_id_lookup") \
        .map(lambda x: (x["business_id"], x["business_index"])).persist()
    business_lookup = business_lookup_rdd.collectAsMap()
    inverse_business_lookup = business_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()

    business_profile_lookup = model.filter(lambda x: x["type"] == "business_profile") \
        .map(lambda x: (x["business_index"], x["business_profile"])).collectAsMap()

    user_lookup_rdd = model.filter(lambda x: x["type"] == "user_id_lookup") \
        .map(lambda x: (x["user_id"], x["user_index"]))
    user_lookup = user_lookup_rdd.collectAsMap()
    inverse_user_lookup = user_lookup_rdd.map(lambda x: (x[1], x[0])).collectAsMap()

    user_profile_lookup = model.filter(lambda x: x["type"] == "user_profile") \
        .map(lambda x: (x["user_index"], x["user_profile"])).collectAsMap()

    cosine_similarity = sc.textFile(text_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["user_id"], x["business_id"])) \
        .map(lambda x: (user_lookup.get(x[0], None), business_lookup.get(x[1]))) \
        .filter(lambda x: x[0] is not None and x[1] is not None) \
        .map(lambda x: (
    x, cosine_similarity_function(user_profile_lookup.get(x[0]), business_profile_lookup.get(x[1])))) \
        .filter(lambda x: x[1] > 0.01).map(
        lambda x: (inverse_user_lookup[x[0][0]], inverse_business_lookup[x[0][1]], x[1])).collect()

    predicted_result = write_output_function(cosine_similarity)

    with open(out_file, "w+") as fileout:
        for r in predicted_result:
            fileout.writelines(json.dumps(r) + "\n")

    sc.stop()

    end = time.time()
    print("Duration:", end - start)


if __name__ == "__main__": main()
