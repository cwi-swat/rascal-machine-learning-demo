@doc{
.Synopsis machine learning algorithm based on the idea of decision trees

.Description

We learned this algorithm from the description on Wikipedia <https://en.wikipedia.org/wiki/K-means_clustering> and coded 
it in Rascal.
}
module analysis::learning::KMeans

extend analysis::learning::DataPoints;

alias Clusters = set[set[Point]];
   
@doc{
.Synopsis: computes the average of all dimensions of all points
.Description:
A centroid is the "average" point of a set of points, composed as the
pairwise mean value of all points for each dimenions.
}
@memo  
private Point centroid(set[Point] cl:{Point h, *_}) =
  point([ sum([p.vec[i] | p <- cl]) / size(cl) | i <- index(h.vec)]); 

@doc{
.Synopsis: simple partition simply breaks the input up in almost even parts, in order of appearance
.Description:
This partioning algorithm serves as a demonstration of what an initial partioning in clusters
should do before the kmeans algorithm is applied.

The current algorithm is not very smart and will lead to longer running times if kmeans
is applied to it, because the chances are that many points have to be moved to new clusters
in many iterations before the optimum is reached.

The advice is to create your own partioning function. Such a function would first guess
a set of initial centroids using any distribution (using knowledge on the input data), 
and then sample efficiently sample around these centroids using another 
distribution function.
}
Clusters simplePartition(list[Point] cl, int k) = kFolds(cl, k);

@memo
private rel[Point centroid, Point point] centroids(Clusters clusters)
  = {<c, p> | cl <- clusters, c := centroid(cl), p <- cl};
  
@doc{
.Synopsis Kmeans optimizes the initial arbitrary clustering by reassigning points to a nearest averaged middle point for each cluster (centroid).
.Description

The efficiency of kmeans is greatly influenced by the initial clustering provided. The closer
the initial cluster is to the optimum which kmeans computes, the fewer steps it takes to reach it.

Kmeans will return no more clusters than the initial clustering, but it may return fewer
clusters. This may happen when the final elements in a cluster are closer to another centroid
than they were to the previous centroid of that cluster: then one cluster completely eats up another.
It's an indication that the initial amount of clusters was not appropriate for the input data, 
but it can also happen accidentally in weird cases when the initial clustering is completely 
unbalanced.
}
Clusters kmeans(Clusters initial) {
  // we assign each point to a cluster, identified by its current centroid
  assignment = centroids(initial);
  
  solve (assignment) {
    // the current centroids are used to index each cluster for each iteration
    cs  = assignment<0>;
    
    // we find which other cluster is currently closer for each point in each cluster:
    movement   = {<(c | dist(d, p) < dist(it, p) ? d : it | d <- cs), p> | c <- cs, p <- assignment[c]};
    
    // then we compute for each new cluster what the new centroid should be
    assignment = {<nc, p>  | c <- cs, cl := movement[c], cl != {}, nc := centroid(cl), p <- cl};
  }
  
  // no data point should have been lost
  assert {*cl | cl <- initial} == {*assignment[cl] | cl <- assignment<0>};
  
  // drop the centroids and simply return a set of clusters
  return { assignment[cl] | cl <- assignment<0>};
}

@doc{
.Synopsis find the closest cluster to the given instance and return its most frequent responses.
}
set[Response] respond(Clusters clusters, Point instance) {
   assignment = centroids(clusters);
   cs = assignment<centroid>;
   closest = (getOneFrom(cs) | dist(c, instance) < dist(it, instance) ? c : it | c <- cs);
   return vote(assignment[closest]);
}
   