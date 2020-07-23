import itertools
import sys
import os
import time
from graphframes import *
from pyspark import SparkContext
from pyspark.shell import sqlContext

os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.7.0-spark2.4-s_2.11")


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def main():
    if len(sys.argv) != 4:
        print("Please enter all the required agruments")
        exit(-1)

    threshhold = int(sys.argv[1])
    csv_file = str(sys.argv[2])
    out_file = str(sys.argv[3])

    start = time.time()
    sc = SparkContext.getOrCreate()
    csv_rdd = sc.textFile(csv_file)

    usr_bus_dict = csv_rdd.filter(lambda x: 'user_id,business_id' not in x)\
        .map(lambda x: x.split(',')) \
        .combineByKey(to_list, append, extend).mapValues(lambda x: sorted(x)).collectAsMap()

    edge_list = [tuple((pair)) for pair in itertools.permutations(sorted(usr_bus_dict.keys()), 2) if
                 len(set(usr_bus_dict[pair[0]]) & set(usr_bus_dict[pair[1]])) >= threshhold]

    edges = sqlContext.createDataFrame(edge_list, ["src", "dst"])

    vertices = sqlContext.createDataFrame(sorted(zip(set(itertools.chain(*edge_list)))), ["id"])

    g = GraphFrame(vertices, edges)

    communities = g.labelPropagation(maxIter=5).rdd.coalesce(1) \
        .map(lambda x: (x[1], x[0])) \
        .combineByKey(to_list, append, extend).map(lambda x: sorted(x[1])) \
        .sortBy(lambda x: (len(x), x)).collect()

    with open(out_file, 'w+') as fileout:
        for c in communities:
            fileout.writelines(str(c)[1:-1] + "\n")

    end = time.time()
    print("Duration: " + str(end - start))


if __name__ == "__main__": main()
