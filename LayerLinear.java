import java.util.Arrays;


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

    Matrix averagedXMatrix = new Matrix();
    Matrix averagedYMatrix = new Matrix();
    Matrix xCentroid = new Matrix();
    Matrix yCentroid = new Matrix();

    //
    // Subtract column averages from FEATURE matrix
    averagedXMatrix.newColumns(x.rows());
    xCentroid.newColumns(x.cols());
    double[] tempXCentroidCol = new double[x.cols()];
    for(int i = 0; i < x.cols(); ++i) {
      double xMean = x.columnMean(i);

      double[] tempColumn = new double[x.rows()];
      Arrays.fill(tempColumn, xMean);
      tempXCentroidCol[i] = xMean;
      averagedXMatrix.takeRow(tempColumn);
    }
    xCentroid.takeRow(tempXCentroidCol);

    averagedXMatrix = averagedXMatrix.transpose();
    x.addScaled(averagedXMatrix, -1.0);

    //
    // Subtract column averages from LABEL matrix
    averagedYMatrix.newColumns(y.rows());
    yCentroid.newColumns(y.cols());
    double[] tempYCentroidCol = new double[y.cols()];
    for(int i = 0; i < y.cols(); ++i) {
      double yMean = y.columnMean(i);

      double[] tempColumn = new double[y.rows()];
      tempYCentroidCol[i] = yMean;
      Arrays.fill(tempColumn, yMean);
      averagedYMatrix.takeRow(tempColumn);
    }
    yCentroid.takeRow(tempYCentroidCol);

    y = y.transpose();
    y.addScaled(averagedYMatrix, -1.0);

    //
    // Matrix multiplication for OLS
    Matrix featuresCrossLabels = Matrix.multiply(y, x, false, false);
    Matrix xTranspose = x.transpose();
    Matrix featuresCrossFeatures = Matrix.multiply(xTranspose, x, false, false);
    Matrix fcfInverse = featuresCrossFeatures.pseudoInverse();
    Matrix weightsMatrix = Matrix.multiply(featuresCrossLabels, fcfInverse, false, false);

    //
    // Calculate bias
    Matrix mx = Matrix.multiply(weightsMatrix, xCentroid.transpose(), false, false);
    yCentroid.addScaled(mx, -1);

    //
    // Push the bias Matrix (yCentroid) and weightsMatrix into one long vector
    int weightsIndex = 0;
    for(int i = 0; i < yCentroid.rows(); ++i) {
      Vec temp = yCentroid.row(i);
      for(int j = 0; j < yCentroid.cols(); ++j) {
        weights.set(weightsIndex, temp.get(j));
        ++weightsIndex;
      }
    }

    for(int i = 0; i < weightsMatrix.rows(); ++i) {
      Vec temp = weightsMatrix.row(i);
      for(int j = 0; j < weightsMatrix.cols(); ++j) {
        weights.set(weightsIndex, temp.get(j));
        ++weightsIndex;
      }
    }

    System.out.println(weights.toString());

    // Adust inputs and outputs sizes
    //inputs = yCentroid.rows();
    //outputs = weightsMatrix.cols();

    //
    // Calculate Y given X
    double[] temp = {0.0, 1.0};
    Vec tX = new Vec(temp);
    activate(weights, tX);
    System.out.println(activation.toString());
  }


}

// System.out.println(weights.toString());
// System.out.println("X-Centroid Rows: " + xCentroid.rows());
// System.out.println("X-Centroid Columns: " + xCentroid.cols());
// System.out.println("ycr " + yCentroid.rows() + " ycc " + yCentroid.cols());
// System.out.println("xr " + averagedXMatrix.rows() + " xc " + averagedXMatrix.cols());
// System.out.println("wr " + weightsMatrix.rows() + " wc " + weightsMatrix.cols());
