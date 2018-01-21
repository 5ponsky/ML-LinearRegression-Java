


public class LayerLinear extends Layer {

  LayerLinear(int inputs, int outputs) {
    super(inputs, outputs);
  }

  void activate(Vec weights, Vec x) {
    int totalEntries = weights.size();
    int computedEntries = 0;
    int i = 0;

    double[] data = new double[outputs];
    Vec b = new Vec(weights, 0, outputs);
    computedEntries = computedEntries + outputs;

    while(computedEntries < totalEntries && i < outputs) {
      Vec temp = new Vec(weights, computedEntries, inputs);
      double newEntry = x.dotProduct(temp);
      activation.set(i, newEntry);
      computedEntries = computedEntries + inputs;
      ++i;
    }
    activation.add(b);

  }

  void ordinary_least_squares(Matrix x, Matrix y, Vec weights) {
    /// x are features
    /// y are labels

    Vec averagedYVec = new Vec(y.rows());
    for(int i = 0; i < y.cols(); ++i) {
      double yMean = y.columnMean(i);
      for(int j = 0; j < y.rows(); ++j) {
        double k = y.row(i).get(j);
        averagedYVec.set(j, k);
      }
    }

    Vec averagedXVec = new Vec(x.rows());
    for(int i = 0; i < x.cols(); ++i) {
      double xMean = x.columnMean(i);
      for(int j = 0; j < x.rows(); ++j) {
        // x.getattribute(i)
        double k = x.row(i).get(j);
        averagedXVec.set(j, k);
      }
    }


    // For each pattern in our data set
    // Feed it into our model (activate) to compute predicted label
    // measure distance between predicted label and actual label and square it

  }


}
