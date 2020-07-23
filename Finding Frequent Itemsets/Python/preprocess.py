import csv
import json
import sys
import time

from pyspark import SparkContext


def main():
    if len(sys.argv) < 2:
        print("Please enter all the required agruments")
        exit(-1)

    start = time.time()
    sc = SparkContext('local[*]', 'task1')

    review_rdd = sc.textFile(str(sys.argv[1])).map(lambda x: json.loads(x)).cache()
    business_rdd = sc.textFile(str(sys.argv[2])).map(lambda x: json.loads(x)).cache()

    b = business_rdd.filter(lambda x: x['state'] == "NV").map(lambda x: (x['business_id'], x['state'])).cache()

    r = review_rdd.map(lambda x: (x['business_id'], x['user_id'])).cache()

    s = r.join(b)

    top_categories = s.map(lambda x: (x[1][0], x[0])).collect()

    fields = ['user_id', 'business_id']

    with open(str(sys.argv[3]), mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for k,v in top_categories:
            d = {'user_id': str(k), 'business_id': str(v)}
            writer.writerow(d)

    sc.stop()


if __name__ == "__main__": main()
