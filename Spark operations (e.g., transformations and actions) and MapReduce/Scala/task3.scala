import java.io._
import org.apache.spark.Partitioner
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._


  object task3 {
    def main(args: Array[String]): Unit = {

      if (args.length != 5) {
        println("Please enter all the required agruments")
        sys.exit(1)
      }

      val ss = SparkSession
        .builder()
        .appName(name = "task3")
        .config("spark.master", "local[*]")
        .getOrCreate()

      val sc = ss.sparkContext
      sc.setLogLevel("ERROR")

      val inputFile = args(0)
      val rdd = sc.textFile(inputFile)
      val dataRdd = rdd.map(string => parse(string).values.asInstanceOf[Map[String, String]])

      if(args(2) == "default"){
        val n_partitions = dataRdd.map(x=>(x("business_id"),1)).groupByKey().map(x=>(x._1,x._2.size)).filter(x=> x._2>args(4).toInt).getNumPartitions
        val n_items = dataRdd.map(x=>(x("business_id"),1)).groupByKey().map(x=>(x._1,x._2.size)).filter(x=> x._2>args(4).toInt).glom().map(_.length).collect()
        val result = dataRdd.map(x=>(x("business_id"),1)).groupByKey().map(x=>(x._1,x._2.size)).filter(x=> x._2>args(4).toInt).collect()

        val output_file = new File(args(1))
        val output_writer = new BufferedWriter(new FileWriter(output_file))
        output_writer.write("{"+ "\"n_partitions\":"+n_partitions.toInt+","+"\"n_items\":"+n_items.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+","+"\"result\":"+result.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+"}")
        output_writer.close()
      }


      if (args(2) == "customized")
      {
        class customizedPartition(totalPartition: Int) extends Partitioner{
          override def numPartitions: Int = totalPartition
          override def getPartition(key: Any): Int = key match {
            case null => 0
            case s:String =>
              val rawMod = s.takeRight(1).hashCode % numPartitions
              rawMod + (if (rawMod < 0) s.takeRight(1).hashCode else 0)
          }
        }

        val patitioned = dataRdd.map(x=>(x("business_id"),1)).partitionBy(new customizedPartition(args(3).toInt)).persist()
        val reduced_data = patitioned.reduceByKey((x,y)=>x+y).filter(x=> x._2 > args(4).toInt)

        val n_items = reduced_data.glom().map(_.length).collect()
        val result = reduced_data.collect()

        val output_file = new File(args(1))
        val output_writer = new BufferedWriter(new FileWriter(output_file))
        output_writer.write("{"+ "\"n_partitions\":"+args(3).toInt+","+"\"n_items\":"+n_items.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+","+"\"result\":"+result.toList.toString().replaceAll("List","").replaceAll("\\(", "[").replaceAll("\\)", "]")+"}")
        output_writer.close()
      }
      sc.stop()
    }
  }
