@doc{
.Synopsis machine learning algorithm based on the idea of decision trees

.Description

We learned this code from Python examples on <https://machinelearningmastery.com/> by Jason Brownlee
and translated the Python to Rascal.
}
module analysis::learning::DecisionTrees

import analysis::learning::DataPoints;
import analysis::statistics::Inference;
import analysis::statistics::Frequency;
import List;
import Set;
import util::Math;

data DecTree 
  = split(int dim, real pivot, DecTree lhs, DecTree rhs)
  | end(set[Point] cluster)
  ;

private DecTree split(int dim, real pivot, set[Point] corpus) 
  = split(dim, pivot, end(lhs), end(corpus - lhs))
  when lhs := {p | p <- corpus, p.vec[dim] < pivot};

@doc{Computes how even the classes are distributed between the left and right parts of the candidate tree.}
private real gini(DecTree cand, set[Response] classes) {
   if (cand.lhs.cluster == {} || cand.rhs.cluster == {}) { 
     return 1.0;
   }
   
   real sqr(real r) = r * r;
   real cumFreq(set[Point] corpus, set[Response] classes)
      = (0.0 | sqr((0 | it + 1 | p <- corpus, p.resp == cl) / (1.000 * S)) | S := size(corpus), cl <- classes);
      
   total = size(cand.lhs.cluster) + size(cand.rhs.cluster);

   return (1.0 - cumFreq(cand.lhs.cluster, classes)) * (size(cand.lhs.cluster) / total)
        + (1.0 - cumFreq(cand.rhs.cluster, classes)) * (size(cand.rhs.cluster) / total)
        ;
}

DecTree buildDecTree(set[Point] corpus, int maxDepth = 5, int minSize = 10)
  = buildDecTree(corpus, maxDepth, minSize);
    
private DecTree buildDecTree(set[Point] corpus, 0, int _) = end(corpus);
private DecTree buildDecTree(set[Point] corpus, int _, int minSize) = end(corpus) when size(corpus) <= minSize;

@memo
private default DecTree buildDecTree(set[Point] corpus, int md, int ms) {
  classes = {p.resp | p <- corpus};
  init = split(0, 0.0, end({}), end({})); // has a worst gini of 1.0 due to empty sets
  
  // this is a brute-force search for a best splitting pivot (dimension & element)
  // for the current corpus. The pivot which distributes the responses (classes) best
  // over the left and right sides of the new tree, wins.
  best = ( init 
         | gini(n, classes) < gini(it, classes) ? n : it 
         | dim <- [0..dim(corpus)], pivot <- corpus, n := split(dim, pivot.vec[dim], corpus));
         
  // the current split is best, now we recursively split the left and right sides again
  // until the corpus is too small to split or until the max depth of the tree has been reached       
  best.lhs = buildDecTree(best.lhs.cluster, md - 1, ms);
  best.rhs = buildDecTree(best.rhs.cluster, md - 1, ms);
  return best; 
}  

set[Response] respond(end(set[Point] cluster), Point _) = vote(cluster);
set[Response] respond(split(int dim, real pivot, DecTree lhs, DecTree rhs), Point instance)
  = instance.vec[dim] < pivot ? respond(lhs, instance) : respond(rhs, instance);
