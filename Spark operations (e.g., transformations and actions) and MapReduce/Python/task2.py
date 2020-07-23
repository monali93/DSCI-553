from pyspark import SparkContext
import json
import sys


def sub_tasks(review, business, output_file, if_spark, n):
    if str(if_spark) == "spark":
        sc = SparkContext('local[*]', 'task2')

        review_rdd = sc.textFile(str(review)).map(lambda x: json.loads(x)).cache()
        business_rdd = sc.textFile(str(business)).map(lambda x: json.loads(x)).cache()

        b = business_rdd.filter(lambda x: x['categories'] is not None) \
            .flatMap(lambda x: [(x['business_id'], i.strip()) for i in x['categories'].split(',')]).cache()

        r = review_rdd.map(lambda x: (x['business_id'], x['stars'])).cache()

        s = r.join(b)

        top_categories = s.map(lambda x: (x[1][1], x[1][0])).groupByKey().mapValues(lambda x: sum(x) / len(x)) \
            .sortBy(lambda x: (-x[1], x[0])).take(int(n))

        output = {
            "result": top_categories
        }

        with open(str(output_file), "w+") as fileout:
            json.dump(output, fileout, indent=4)

        sc.stop()

    elif str(if_spark) == "no_spark":
        review_dict = {}
        with open(str(review)) as f:
            for l in f:
                obj = json.loads(l)
                id = str(obj['business_id'])
                stars = obj['stars']
                if id not in review_dict:
                    review_dict[id] = (stars, 1)
                else:
                    val = review_dict.get(id)
                    s = val[0] + stars
                    count = val[1] + 1
                    review_dict[id] = (s, count)

        category_dict = {}
        with open(str(business), encoding="utf8") as f:
            for l in f:
                obj = json.loads(l)
                id = str(obj['business_id'])
                if obj['categories'] is not None:
                    category_list = obj['categories'].split(',')
                    for cat in category_list:
                        cat = cat.strip()
                        if cat not in category_dict:
                            if id in review_dict:
                                category_dict[cat] = review_dict[id]
                        else:
                            val = category_dict.get(cat)
                            rev = review_dict.get(id)
                            if rev is not None and rev[0] is not None and rev[1] is not None:
                                s = val[0] + rev[0]
                                count = val[1] + rev[1]
                                category_dict[cat] = (s, count)

        for key, value in category_dict.items():
            category_dict[key] = value[0] / value[1]

        top_categories = sorted(category_dict.items(), reverse=False, key=lambda x: (-x[1], x[0]))

        output = {
            "result": top_categories[:int(n)]
        }

        with open(str(output_file), "w+") as fileout:
            json.dump(output, fileout, indent=4)


def main():
    if len(sys.argv) != 6:
        print("Please enter all the required agruments")
        exit(-1)

    sub_tasks(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


if __name__ == "__main__": main()
