@doc{
.Synopsis
Example application of different machine learning algorithms to a typical dataset.

The example data is from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml), by Dua Dheeru and Efi Karra Taniskidou of
University of California, Irvine, School of Information and Computer Sciences.

Example inspired by <https://machinelearningmastery.com/> by Jason Brownlee.
}
module analysis::learning::example::Banknote

import analysis::learning::example::TestAlgorithms;
import analysis::learning::DataPoints;
import lang::csv::IO;

alias Banknote 
  = rel[real variance, real skewness, real curtosis, real entropy, int class];

data Response = class(int number);
  
void banknoteDemo() {
  raw = readCSV(#Banknote, |project://rascal-machine-learning-demo/src/analysis/learning/example/banknote.csv|);
  
  points = {point([d.variance, d.skewness, d.curtosis, d.entropy], resp=class(d.class)) | d <- raw};
  <trainers, tests> = split(points, 2r3);

  testAlgorithms(trainers, tests);
}