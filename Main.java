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
		int length = featureData.rows();
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
		for(int k = 0; k < repititions; ++k) {
			for(int i = beginStep; i < folds; ++i) {
				int firstTrainBlockSize = beginIndex;
				int secondTrainBlockSize = featureData.rows() - endIndex - 1;

				// First Training block
				trainFeatures.copyBlock(0, 0, featureData, 0, 0, firstTrainBlockSize, 13);
				trainLabels.copyBlock(0, 0, featureData, 0, 0, firstTrainBlockSize, 1);

				// Test block
				testFeatures.copyBlock(0, 0, featureData, beginIndex, 0, testBlockSize, 13);
				testLabels.copyBlock(0, 0, featureData, beginIndex, 0, testBlockSize, 1);

				// 2nd Training block
				trainFeatures.copyBlock(firstTrainBlockSize, 0, featureData,
					firstTrainBlockSize + testBlockSize, 0, secondTrainBlockSize, 13);
				trainLabels.copyBlock(firstTrainBlockSize, 0, featureData,
					firstTrainBlockSize + testBlockSize, 0, secondTrainBlockSize, 1);

				learner.train(trainFeatures, trainLabels);

				sse = sse + learner.sum_squared_error(testFeatures, testLabels);

				// Adjust the interval slicing
				++beginStep;
				++endStep;
				beginIndex = testBlockSize * beginStep;
				endIndex = testBlockSize * endStep;
			}

			beginStep = 0;
			endStep = 1;
			beginIndex = (int)((double)length * foldRatio * (double)beginStep);
			endIndex = (int)((double)length * foldRatio * (double)endStep);

			mse = mse + (sse / (double)featureData.rows());
			sse = 0;

			for(int i = 0; i < featureData.rows(); ++i) {
				int selectedRow = random.nextInt(featureData.rows());
				int destinationRow = random.nextInt(featureData.rows());
				featureData.swapRows(selectedRow, destinationRow);
				labelData.swapRows(selectedRow, destinationRow);
			}
		}

		rmse = Math.sqrt(mse / repititions);
		System.out.println("RMSE: " + rmse);

	}

	public static void testLearner(SupervisedLearner learner)
	{
		test(learner, "hep");
		test(learner, "vow");
		test(learner, "soy");
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

		LayerLinear ll = new LayerLinear(testFeatures.cols(), testLabels.cols());
		ll.ordinary_least_squares(testFeatures, testLabels, weights);
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
		testRegression(new BaselineLearner());
		//testOLS();

		//testLearner(new RandomForest(50));
	}
}
