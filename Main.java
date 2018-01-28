// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

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

	static void testHousing(SupervisedLearner learner) {
		// Load the training data
		Matrix trainFeatures = new Matrix();
		trainFeatures.loadARFF("data/housing_features.arff");
		Matrix trainLabels = new Matrix();
		trainLabels.loadARFF("data/housing_labels.arff");

		// Train the model
		learner.train(trainFeatures, trainLabels);
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
		//testHousing(new BaselineLearner());
		testOLS();

		//testLearner(new RandomForest(50));
	}
}
