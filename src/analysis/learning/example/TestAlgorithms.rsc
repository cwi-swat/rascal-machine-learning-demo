module analysis::learning::example::TestAlgorithms

import analysis::learning::DataPoints;
import analysis::learning::KMeans;
import analysis::learning::KNearestNeighbors;
import analysis::learning::DecisionTrees;
import analysis::learning::BootstrapAggregation;
import analysis::learning::RandomForest;

import analysis::statistics::Descriptive;

import util::Math;
import IO;
import Set;
import List;

data Algorithm[&M] = algorithm(str name, &M (set[Point]) train, set[Response] (&M, Point) predict);

list[Algorithm[value]] algorithms
  = [algorithm("kMeans", 
               set[set[Point]] (set[Point] trainers) {      
                 return kmeans(simplePartition(trainers, 5)); 
               },
               set[Response] (set[set[Point]] clusters, Point p) { 
                 return respond(clusters, p); 
               }
              ),
                         
     algorithm("kNN",    
               set[Point] (set[Point] trainers) { 
                 return trainers; 
               }, 
               set[Response] (set[Point] corpus, Point p) { 
                 return respond(corpus, p, 5); 
               }),
               
     algorithm("Decision Tree",    
               DecTree (set[Point] trainers) { 
                  return buildDecTree(trainers, maxDepth=5, minSize=10); 
               }, 
               set[Response] (DecTree t, Point p) { 
                 return respond(t, p); 
               }),
               
     algorithm("Bagged Decision Trees",    
               set[DecTree] (set[Point] trainers) { 
                  return buildDecTrees(trainers, maxDepth=5, minSize=10); 
               }, 
               set[Response] (set[DecTree] f, Point p) { 
                 return respond(f, p); 
               }),
               
     algorithm("Random Forest",    
               set[DecTree] (set[Point] trainers) { 
                   return buildRandomForest(trainers, maxDepth=5, minSize=10, sampleRatio=0.5, treeCount=20, featureCount=dim(trainers) / 2);
               }, 
               set[Response] (set[DecTree] f, Point p) { 
                 return respond(f, p); 
               })  
                                          
    ];

void testAlgorithmAccuracy(set[Point] corpus, int repeats = 3) {
  println("Running <size(algorithms)> algorithms, each <repeats> times on randomly split data.");
  
  result = algList:for (Algorithm[&M] a <- algorithms) {
    accuracy = roundList: for (i <- [0..repeats]) {
      pref = "Round <i + 1>/<repeats>:";
    
      <trainers, tests> = split(corpus, 2r3);
      println("<pref> Split the corpus of <size(corpus)> elements into a training set (<size(trainers)>) and a test set (<size(tests)>).");
      
      println("<pref> Training <a.name> using <size(trainers)> points");
      m = a.train(corpus);
      
      println("<pref> Testing <a.name> on <size(tests)> points");
      append roundList: (0 | r == t.resp ? it + 1 : it | t <- tests, {r, *_} := a.predict(m, t));
    }
    
    println("Reporting <a.name> average of <repeats> rounds: <round(mean(accuracy))>% accurate");
    append algList: <a.name, round(mean(accuracy))>;
  }
  
  println("------------------------
          'Done testing algorithms:
          '<for (<alg, acc> <- result) {><acc>% using <alg>
          '<}>");
}
   
