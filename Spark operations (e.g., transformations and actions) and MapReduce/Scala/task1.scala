import java.io.{BufferedWriter, File, FileWriter}
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.parse
import scala.io._


object task1 {

  def main(args: Array[String]): Unit ={

    if (args.length != 6) {
      println("Please enter all the required agruments")
      sys.exit(1)
    }

    val lines = Source.fromFile(args(2)).getLines.toList
    val pun_list = List("(", "[", ",", ".", "!", "?", ":", ";", "]", ")")
    val reg_pattern = "[^a-zA-Z0-9\\s]".r

    val ss = SparkSession
      .builder()
      .appName(name = "task1")
      .config("spark.master", "local[*]")
      .getOrCreate()

    val sc = ss.sparkContext
    sc.setLogLevel("ERROR")

    val inputFile = args(0)

    val rdd = sc.textFile(inputFile)
    val dataRdd =rdd.map(string => parse(string).values.asInstanceOf[Map[String,String]])

    val count = dataRdd.map(x=>x("review_id")).count()

    val year = dataRdd.filter(x =>x("date").contains(args(3).toString)).count()

    val distinct_user = dataRdd.map(x=>x("user_id")).distinct().count()

    val top_reviewers = dataRdd.map(x=>(x("user_id"),1)).reduceByKey((x, y) => x + y).sortBy(_._2,ascending = false).take(args(4).toInt)

    val top_words = dataRdd.flatMap(x=>x("text").toLowerCase.split(" "))
      .filter(x=> ! (pun_list.contains(x)))
      .map(x=>  reg_pattern.replaceAllIn(x,"")).filter(x=> ! (lines.contains(x)))
      .filter(x=> x.length>0).map(x => (x,1)).reduceByKey((x,y)=>x+y).sortBy(x =>(-x._2,x._1)).map(x => x._1).take(args(5).toInt)


    val output_file = new File(args(1))
    val output_writer = new BufferedWriter(new FileWriter(output_file))
    output_writer.write("{"+ "\"A\":"+count.toInt+","+"\"B\":"+year+","+"\"C\":"+distinct_user+","+"\"D\":"+top_reviewers.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+","+"\"E\":"+top_words.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+"}")
    output_writer.close()

    sc.stop()

  }

}
