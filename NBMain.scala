import scala.io.Source
import collection.mutable.HashMap
import collection.mutable.MutableList

class NaiveBayes(fileName: String){
  //parse file into a list 
  val fileLines = io.Source.fromFile(fileName).getLines.toList;
  
  //vars to hold number of correct/incorrect instances after classification
  var correct = 0
  var incorrect = 0
    
  //var used for preprocessing of file, contains list of labels and attributes
  var headerInfo: List[(String,List[String])] = extractAttributes()
  
  //extract list of labels from headerInfo
  var labels = headerInfo(headerInfo.length-1)._2
  
  //remove the labels(last element), leaving only attribute categories and values
  var attributeList = headerInfo.dropRight(1)
  
  //iterate through attributeList, take only attribute categories
  var attributes = for(al <- attributeList) yield al._1
  
  //used to hold the counts of how often an attribute entry occurs
  var attributeCounts = new HashMap[(String, String, String), Int]()
  
  //used to hold the counts of how often a label entry occurs
  var label_counts = new HashMap[String, Int]()
  
  //parse file list, extract only data entries
  var dataVectors : List[List[String]] = for(line <- fileLines if (line(0) != '@') ) yield line.split(",").toList
  
  
  //initialize counts to 1
  {
    for(l <- labels) yield label_counts += (l -> 0)
    for(l <- labels; a <- attributeList; v <- a._2) yield {
      attributeCounts += ((l, a._1, v) -> 0)
    }
  }
  
  //function used to parse header part of arff file to deduce the attribute categories/values and labels
  def extractAttributes() = {
    var orderedAttribute : List[(String,List[String])] = List()	
    for(line <- fileLines ){
		  var tok = line.split(" ")(0).toLowerCase() //****
		  if (tok== "@attribute"){
			  var cleanline = for(ch <- line  if (ch != '{' && ch != '}' && ch != ',')) yield ch;
			  var attribute = cleanline.split(" ").drop(1).toList(0)
			  orderedAttribute = orderedAttribute:+((attribute, cleanline.split(" ").drop(2).toList ))	  
		  }  
    }
    orderedAttribute
  }
  
  //function that looks at data part of arff file and trains the classifier by counting the 
  //number of times an attribute entry and label occurs
  def train() = {
    for(line <- fileLines ){
		  var tok = line.split(" ")(0).toLowerCase() //***********
		    if ( (tok != "@attribute") && (tok != "@relation") && (tok != "@data") ){
		      
		      var cleanline = for(ch <- line  if (ch != '{' && ch != '}')) yield ch;
			    var linelist = cleanline.toLowerCase().split(",").toList
			    var line_label = linelist(linelist.length-1)
			    label_counts(line_label) += 1

			    for(ll <- 0 to linelist.length-2){
			      attributeCounts(line_label, attributes(ll), linelist(ll)) += 1
			    }
			    
		    }
		  }
  }
  
  //for every data vector there is, classify it
  def classify() = {
    for(dV <- dataVectors){
      classifyVector(dV) 
    }
  }
  
  //function that takes a data vector and classifies it by calculating: 
  //argMax( p(label)*PI[p(attribute_i | label)] )
  def classifyVector(attributeVector: List[String]) = {
    var trueLabel = attributeVector.last
    //println(trueLabel)
    var labelProbs = MutableList[Double]()
    for(l <- labels){ 
      var thisProb:Double = 1
      for(av <- 0 to attributes.length-1){

      thisProb += Math.log((attributeCounts(l.toLowerCase(), 
          attributes(av).toLowerCase(), attributeVector(av).toLowerCase()) ).toDouble / label_counts(l).toDouble)
      
      }  
      labelProbs.+=(thisProb)
    }
    var predictedLabel = labels(labelProbs.indexOf(labelProbs.reduceLeft(_ max _)))

    if(trueLabel.toLowerCase() == predictedLabel.toLowerCase()) correct += 1
    else incorrect += 1
    println("Real label: "+trueLabel)
    println("Predicted: "+ predictedLabel)
    println("************************")
    
  }
  
  //print the accuracy of the trained classifier
  def report() = {
    println("\nCorrect: "+ correct + "\n"+
           "Incorrect: " + incorrect + "\n"+
           "Accuracy: " + (correct/(correct + incorrect).toDouble)
    )
  }     
}


object NBMain {
  
  def main(args: Array[String]){
    var myNB = new NaiveBayes("contact_lens.arff") //dataset file
    myNB.train()
    myNB.classify()
    myNB.report()
  } 
  
}