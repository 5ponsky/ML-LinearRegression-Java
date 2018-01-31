// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;

class Main
{
	static void test(SupervisedLearner learner, String challenge) {
		// Load the training data
		String fn = "data/" + challenge;
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF(fn + "_train_feat.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF(fn + "_train_lab.arff");

		// Train the model
		learner.train(trainFeatures, trainLabels);

		// Load the test data
		Matrix testFeatures = new Matrix();
		testFeatures.loadARFF(fn + "_test_feat.arff");
		Matrix testLabels = new Matrix();
		testLabels.loadARFF(fn + "_test_lab.arff");

		// Measure and report accuracy
		int misclassifications = learner.countMisclassifications(testFeatures, testLabels);
		System.out.println("Misclassifications by " + learner.name() + " at " + challenge + " = " + Integer.toString(misclassifications) + "/" + Integer.toString(testFeatures.rows()));
	}

	static void testRegression(SupervisedLearner learner) {
		Random random = new Random(1234);

		// Load the training data
		Matrix featureData = new Matrix();
		featureData.loadARFF("data/housing_features.arff");
		Matrix labelData = new Matrix();
		labelData.loadARFF("data/housing_labels.arff");

		// Cross-Validation indices
		int repititions = 5;
		int folds = 10;
		double foldRatio = 1.0 / (double)folds;
		int beginStep = 0;
		int endStep = 1;
		int length = featureData.rows() - 2;
		int testBlockSize = (int)(length * foldRatio);
		int beginIndex = (int)((double)length * foldRatio * (double)beginStep);
		int endIndex = (int)((double)length * foldRatio * (double)endStep);

		// Create train matrices
		Matrix trainFeatures = new Matrix((int)(featureData.rows() - featureData.rows()*foldRatio), featureData.cols());
		Matrix trainLabels = new Matrix((int)(featureData.rows() - featureData.rows()*foldRatio), featureData.cols());

		// Create test matrices
		Matrix testFeatures = new Matrix((int)(featureData.rows()*foldRatio), featureData.cols());
		Matrix testLabels = new Matrix((int)(featureData.rows()*foldRatio), labelData.cols());

		// Partition the data by folds
		double sse = 0; // Sum squared error
		double mse = 0; // Mean squared error
		double rmse = 0; // Root mean squared error


		//System.out.println("trainFeatures" + trainFeatures.rows());
		//System.out.println("testFeatures" + testFeatures.rows());

		for(int k = 0; k < repititions; ++k) {
			for(beginStep = 0; beginStep < folds; ++beginStep) {
				beginIndex = beginStep * (length / folds);
				endIndex = (beginStep + 1) * (length / folds);
				//System.out.println("beginIndex " + beginIndex);
				//System.out.println("endIndex " + endIndex);

				// First Training block
				trainFeatures.copyBlock(0, 0, featureData, 0, 0, beginIndex, 13);
				trainLabels.copyBlock(0, 0, featureData, 0, 0, beginIndex, 1);

				// Test block
				testFeatures.copyBlock(0, 0, featureData, beginIndex+1, 0, endIndex-beginIndex, 13);
				testLabels.copyBlock(0, 0, featureData, beginIndex+1, 0, endIndex-beginIndex, 1);

				// 2nd Training block
				trainFeatures.copyBlock(beginIndex+1, 0, featureData,
					beginIndex+1, 0, length - endIndex, 13);
				trainLabels.copyBlock(beginIndex+1, 0, featureData,
					beginIndex+1, 0, length - endIndex, 1);

				// System.out.println("Tr-Block 1: " + 0 + " to " + firstTrainBlockSize);
				// System.out.println("Te-Block: " + beginIndex + " to " + testBlockSize);
				// System.out.println("Tr-Block 2: " + (firstTrainBlockSize+testBlockSize) + " to " + 505 + '\n');

				learner.train(trainFeatures, trainLabels);

				sse = sse + learner.sum_squared_error(testFeatures, testLabels);
			}

			mse = mse + (sse / length);
			sse = 0;

			for(int i = 0; i < featureData.rows(); ++i) {
				int selectedRow = random.nextInt(length);
				int destinationRow = random.nextInt(length);
				featureData.swapRows(selectedRow, destinationRow);
				labelData.swapRows(selectedRow, destinationRow);
			}
		}

		rmse = Math.sqrt(mse/repititions);
		System.out.println("RMSE: " + rmse);

	}

	public static void testOLS2() {
		LayerLinear ll = new LayerLinear(13, 1);
		Random random = new Random(1234);
		Vec weights = new Vec(14);

		for(int i = 0; i < 14; ++i) {
			weights.set(i, random.nextGaussian());
		}

		Matrix x = new Matrix(100, 13);
		for(int i = 0; i < 100; ++i) {
			double[] temp = new double[13];
			for(int j = 0; j < 13; ++j) {
				temp[j] = random.nextGaussian();
			}
			x.takeRow(temp);
		}

		Matrix y = new Matrix(100, 1);
		for(int i = 0; i < y.rows(); ++i) {
			ll.activate(weights, x.row(i));
			for(int j = 0; j < ll.activation.size(); ++j) {
				double temp = ll.activation.get(j) + random.nextGaussian();
				y.row(i).set(j, temp);
			}
		}

		for(int i = 0; i < y.rows(); ++i) {
    	System.out.println(y.row(i).toString());
		}

		Vec olsWeights = new Vec(14);
		ll.ordinary_least_squares(x,y,olsWeights);


	}

	public static void test() {
		Matrix test = new Matrix();
		test.newColumns(3);
		double[] x1 = {1, 2, 3};
		double[] x2 = {3, 4, 5};
		double[] x3 = {6, 7, 8};
		test.takeRow(x1);
		test.takeRow(x2);
		test.takeRow(x3);

		for(int i = 0; i < test.rows(); i++) {
			test.row(i).set(i, -1);
		}

		for(int i = 0; i < test.rows(); i++) {
			System.out.println(test.row(i).toString());
		}
	}

	public static void testOLS() {

		Matrix testFeatures = new Matrix();
		testFeatures.newColumns(2);
		double[] x1 = {0,1};
		double[] x2 = {1,2};
		double[] x3 = {2,0};
		testFeatures.takeRow(x1);
		testFeatures.takeRow(x2);
		testFeatures.takeRow(x3);

		Matrix testLabels = new Matrix();
		testLabels.newColumns(1);
		double[] y1 = {2};
		double[] y2 = {0};
		double[] y3 = {1};
		testLabels.takeRow(y1);
		testLabels.takeRow(y2);
		testLabels.takeRow(y3);

		//Vec weights = new Vec(testFeatures.rows() + (testFeatures.rows() * testFeatures.cols()));
		Vec weights = new Vec(3);

		double[] t1 = {0,1};
		Vec t = new Vec(t1);

		LayerLinear ll = new LayerLinear(testFeatures.cols(), testLabels.cols());
		ll.ordinary_least_squares(testFeatures, testLabels, weights);
		ll.activate(weights, t);
		//Vec labels = new Matrix
	}


	public static void testLayer() {
		double[] x = {0, 1, 2};
		double[] m = {1, 5, 1, 2, 3, 2, 1, 0};
		LayerLinear ll = new LayerLinear(3, 2);
		ll.activate(new Vec(m), new Vec(x));
		System.out.println(ll.activation.toString());
	}

	public static void main(String[] args)
	{
		//testLearner(new BaselineLearner());
		//testRegression(new BaselineLearner());
		testOLS2();
		//testLayer();
		//test();

		//testLearner(new RandomForest(50));
	}
}
