import org.apache.spark.sql.SparkSession
import java.io._
import org.json4s
import scala.collection.mutable
import scala.io.Source
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._


object task2 {

  def main(args: Array[String]): Unit ={

    if (args.length != 5) {
      println("Please enter all the required agruments")
      sys.exit(1)
    }

    val ss = SparkSession
      .builder()
      .appName(name = "task2")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val sc = ss.sparkContext
    sc.setLogLevel("ERROR")

    if (args(3) == "spark"){

      val review_rdd = sc.textFile(args(0).toString).map(string => parse(string).values.asInstanceOf[Map[String,String]])

      val business_rdd = sc.textFile(args(1).toString).map(string => parse(string).values.asInstanceOf[Map[String,String]])

      val b =business_rdd.filter(x=> x("categories") != null)
        .flatMap(x=> for (i <- x("categories").split(",")) yield (x("business_id"), i.trim))

      val r =review_rdd.map(x=>(x("business_id"), x("stars")))

      val s= r.join(b)

      val top_categories = s.map(x => (x._2._2, x._2._1)).map(x => (x._1, (x._2.asInstanceOf[Double], 1)))
        .reduceByKey((x, y) => {
          (y._1.asInstanceOf[Double] + x._1.asInstanceOf[Double], x._2 + y._2)
        }).mapValues{ case (sum , count) => sum/count}.sortBy(x => (-x._2,x._1)).take(args(4).toInt)

      val output_file = new File(args(2))
      val output_writer = new BufferedWriter(new FileWriter(output_file))
      output_writer.write("{"+ "\"result\":"+top_categories.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+"}")
      output_writer.close()

    }

    if (args(3) == "no_spark"){
      val review_map:mutable.Map[String,Tuple2[Double, Int]] = mutable.Map()
      val json_values = Source.fromFile(args(0))("UTF-8")
      for(line <- json_values.getLines()) {
        val json_line = parse(line)
        val b = (json_line \ "business_id").asInstanceOf[json4s.JString].s
        val stars = (json_line \ "stars").asInstanceOf[_root_.org.json4s.JsonAST.JDouble].num
        if(!review_map.contains(b)) {
          review_map += b -> (stars,1)
        }
        else {
          val t = review_map(b)
          val sum  = t._1 + stars
          val count = t._2 + 1
          review_map(b) = (sum, count)
        }
      }

      val categories_map:mutable.Map[String,Tuple2[Double, Int]] = mutable.Map()
      val categories_values = Source.fromFile(args(1))("UTF-8")
      for(lines <- categories_values.getLines()) {
        val line = parse(lines)
        val b = (line \ "business_id").asInstanceOf[json4s.JString].s
        val category = (line \ "categories")
        var categories:String = null

        if (category.values!= null) {
          categories = category.asInstanceOf[json4s.JString].s
        }
        if(categories != null && categories.nonEmpty) {
          val categories_array= categories.split(",")
          for (cat <- categories_array) {
            if(!categories_map.contains(cat.trim)){
              if(review_map.contains(b)){
                categories_map(cat.trim) = (review_map(b)._1, review_map(b)._2)
              }
            }
            else {
              val v = categories_map(cat.trim)
              if(review_map.contains(b)){
                val rev = review_map(b)
                val s = v._1 + rev._1
                val c = v._2 + rev._2
                categories_map(cat.trim) = (s,c)
              }
            }
          }
        }
      }

      val final_map:mutable.Map[String,Double] = mutable.Map()
      for((k, v) <- categories_map) {
        final_map(k) = v._1/v._2
      }
      val sort_map = final_map.toList.sortBy(r=>(-r._2, r._1))

      val json : JObject = "result" -> sort_map.slice(0,args(4).toInt)

      val output_file = new File(args(2))
      val output_writer = new BufferedWriter(new FileWriter(output_file))
      val v =pretty(render(json)).replaceAll("\\{", "[").replaceAll("\\}", "]") +"}"
      output_writer.write(v.replaceFirst("\\[", "{").replaceAll("\\]\\}", "}"))
      output_writer.close()

    }

    sc.stop()
  }

}
