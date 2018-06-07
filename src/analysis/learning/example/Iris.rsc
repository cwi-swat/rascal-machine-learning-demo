@doc{
.Synopsis
Example application of different machine learning algorithms to a typical dataset.

The example data is from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml), by Dua Dheeru and Efi Karra Taniskidou of
University of California, Irvine, School of Information and Computer Sciences.

Example inspired by <https://machinelearningmastery.com/> by Jason Brownlee.
}
module analysis::learning::example::Iris

import analysis::learning::example::TestAlgorithms;
import analysis::learning::DataPoints;
import lang::csv::IO;

alias Iris = rel[real sepalLength, real sepalWidth, real petalLength, real petalWidth, str class];

void irisDemo() {
  raw = readCSV(#Iris, |project://rascal-machine-learning-demo/src/analysis/learning/example/iris.csv|);
  points = {point([d.sepalLength, d.sepalWidth, d.petalLength, d.petalWidth], resp=class(d.class)) | d <- raw};
  testAlgorithmAccuracy(points);
}