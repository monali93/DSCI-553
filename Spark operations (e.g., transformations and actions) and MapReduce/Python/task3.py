from pyspark import SparkContext
import json
import sys


def sub_tasks(review_rdd, output_file, partition_type, n_part, n):
    if str(partition_type) == "default":
        n_partitions = review_rdd.map(lambda x: (x['business_id'], 1)).groupByKey().map(lambda x: (x[0], len(x[1]))) \
            .filter(lambda x: x[1] > int(n)).getNumPartitions()

        n_items = review_rdd.map(lambda x: (x['business_id'], 1)).groupByKey().map(lambda x: (x[0], len(x[1]))) \
            .filter(lambda x: x[1] > int(n)).glom().map(len).collect()

        result = review_rdd.map(lambda x: (x['business_id'], 1)).groupByKey().map(lambda x: (x[0], len(x[1]))) \
            .filter(lambda x: x[1] > int(n)).collect()

        output = {
            "n_partitions": n_partitions,
            "n_items": n_items,
            "result": result,
        }

        with open(str(output_file), "w+") as fileout:
            json.dump(output, fileout, indent=4)

    elif str(partition_type) == "customized":
        r = review_rdd.map(lambda x: (x['business_id'], 1)).partitionBy(int(n_part),
                                                                        lambda x: ord(x[-1])).reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] > int(n)).persist()
        n_items = r.glom().map(len).collect()

        result = r.collect()

        output = {
            "n_partitions": int(n_part),
            "n_items": n_items,
            "result": result,
        }

        with open(str(output_file), "w+") as fileout:
            json.dump(output, fileout, indent=4)


def main():
    if len(sys.argv) != 6:
        print("Please enter all the required agruments")
        exit(-1)

    sc = SparkContext('local[*]', 'task3')
    review_rdd = sc.textFile(str(sys.argv[1])).map(lambda x: json.loads(x))

    sub_tasks(review_rdd, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    sc.stop()


if __name__ == "__main__": main()
