@doc{
.Synopsis machine learning algorithm based on the idea of Random Forests

.Description

RandomForests are a refinement on Decision Trees with Bootstrap Aggregation. Instead
of letting many decision trees, for randomly subsamples parts of the corpus, vote, each
tree is now also randomly optimizing for different dimensions at each level of the tree.
The thinking is that due to the optimization for a best split, and with typical variance
in the corpus, most trees will look similar and the bootstrap aggregation can't be of 
much help to improve predictor accuracy. Maybe by selecting different dimensions to 
optimize on, the randomized trees can better complement each other, at least they will
be guaranteed to be different. It depends on the data whether or not this will help.  

A forest is a collection of trees. A "random forest" is a collection of random trees.  

We learned this code from Python examples on <https://machinelearningmastery.com/> by Jason Brownlee
and translated the Python to Rascal. Bootstrap aggregation generates a number of decision trees
based on randomized subsets of the training corpus and then lets the trees vote democratically for
the right response.
}
module analysis::learning::RandomForest

extend analysis::learning::BootstrapAggregation; // which extends DecisionTrees
import IO;

private DecTree buildRandomDecTree(set[Point] corpus, 0, int _, int _) = end(corpus);
private DecTree buildRandomDecTree(set[Point] corpus, int _, int minSize, int _) = end(corpus) when size(corpus) <= minSize;

@memo
private default DecTree buildRandomDecTree(set[Point] corpus, int md, int ms, int fs) {
  classes = {p.resp | p <- corpus};
  init = split(0, 0.0, end({}), end({})); // has a worst gini of 1.0 due to empty sets
  
  // this is where the algorithm is different from normal decision trees;
  // a number of features is randomly selected among the available dimensions
  features = subsample({*[0..dim(corpus)]}, fs);
  
  // this is a brute-force search for a best splitting pivot (dimension & element)
  // for the current corpus. The pivot which distributes the responses (classes) best
  // over the left and right sides of the new tree, wins.
  best = ( init 
         | gini(n, classes) < gini(it, classes) ? n : it 
         | dim <- features, pivot <- corpus, n := split(dim, pivot.vec[dim], corpus)
         );
         
  // the current split is best, now we recursively split the left and right sides again
  // until the corpus is too small to split or until the max depth of the tree has been reached       
  best.lhs = buildRandomDecTree(best.lhs.cluster, md - 1, ms, fs);
  best.rhs = buildRandomDecTree(best.rhs.cluster, md - 1, ms, fs);
  return best; 
}  

// creating the forest is done exactly as in BootstrapAggregation, also computing a response is
// the same (the response function was inherited and is unchanged as it works on a set[DecTree] already)
set[DecTree] buildRandomForest(set[Point] corpus, int maxDepth=5, int minSize=10, num sampleRatio=0.2, int treeCount=10, int featureCount = dim(corpus) / 2)
  = {buildRandomDecTree(subsample(corpus, round(size(corpus) * 1.0 * sampleRatio)), maxDepth, minSize, featureCount) | _ <- [0..treeCount]}; 