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

	static void testRegression(SupervisedLearner learner) {
		// Load the training data
		Matrix featureData = new Matrix();
		featureData.loadARFF("data/housing_features.arff");
		Matrix labelData = new Matrix();
		labelData.loadARFF("data/housing_labels.arff");

		// Train the model
		//learner.train(trainFeatures, trainLabels);

		// Cross-Validation indices
		int folds = 10;
		double foldRatio = 1.0 / folds;
		int beginStep = 0;
		int endStep = 1;
		int length = featureData.rows();
		int testBlockSize = (int)(length * foldRatio);
		int beginIndex = (int)((double)length * foldRatio * (double)beginStep);
		int endIndex = (int)((double)length * foldRatio * (double)endStep);

		// Create train matrix
		Matrix trainFeatures = new Matrix((int)(featureData.rows() - featureData.rows()*foldRatio), featureData.cols());
		Matrix trainLabels = new Matrix((int)(featureData.rows() - featureData.rows()*foldRatio), featureData.cols());

		// Create test matrix
		Matrix testFeatures = new Matrix((int)(featureData.rows()*foldRatio), featureData.cols());
		Matrix testLabels = new Matrix((int)(featureData.rows()*foldRatio), labelData.cols());


		//int destRow, int destCol, Matrix that,
		//int rowBegin, int colBegin, int rowCount, int colCount

		// System.out.println("feature data rows " + featureData.rows());
		// System.out.println("trainFeatures rows: " + trainFeatures.rows());
		// System.out.println("trainLabels rows: " + trainLabels.rows());
		// System.out.println("testFeatures rows: " + testFeatures.rows());
		// System.out.println("testLabels rows: " + testLabels.rows() + "\n");


		// Partition the data 5-fold
		for(int i = beginStep; i < folds; ++i) {
			int firstTrainBlockSize = beginIndex;
			int secondTrainBlockSize = featureData.rows() - endIndex - 1;

			// System.out.println("Train 1st block: " + 0 + " to " + firstTrainBlockSize);
			// System.out.println("test block " + beginIndex + " to " + endIndex);
			// System.out.println("Train 2nd block: " +
			// 	(firstTrainBlockSize + testBlockSize) + " to " + featureData.rows());
			// System.out.println("test block size: " + testBlockSize);
			// System.out.println("Train Size: " + (secondTrainBlockSize + firstTrainBlockSize));
			// System.out.println("beginIndex: " + beginIndex);
			// System.out.println("endIndex: " + endIndex);

			// First Training block
			trainFeatures.copyBlock(0, 0, featureData, 0, 0, firstTrainBlockSize, 13);
			// Test block
			testFeatures.copyBlock(0, 0, featureData, beginIndex, 0, testBlockSize, 13);
			testLabels.copyBlock(0, 0, featureData, beginIndex, 0, testBlockSize, 1);
			// 2nd Training block
			trainFeatures.copyBlock(firstTrainBlockSize, 0, featureData,
				firstTrainBlockSize + testBlockSize, 0, secondTrainBlockSize, 13);

			trainLabels.copyBlock(0, 0, featureData, 0, 0, firstTrainBlockSize, 1);
			trainLabels.copyBlock(firstTrainBlockSize, 0, featureData,
				firstTrainBlockSize + testBlockSize, 0, secondTrainBlockSize, 1);

			System.out.println(" BREAK ");

			learner.train(trainFeatures, trainLabels);

			double misclassifications = learner.sum_squared_error(testFeatures, testLabels);
			System.out.println("error?: " + misclassifications);


			//
			// Adjust the interval slicing
			++beginStep;
			++endStep;
			beginIndex = testBlockSize * beginStep;
			endIndex = testBlockSize * endStep;
		}

	}

	static void testHousing(SupervisedLearner learner) {
		// Load the training data
		Matrix featureData = new Matrix();
		featureData.loadARFF("data/housing_features.arff");
		Matrix labelData = new Matrix();
		labelData.loadARFF("data/housing_labels.arff");

		// Train the model
		//learner.train(trainFeatures, trainLabels);

		// Cross-Validation indices
		int folds = 10;
		double foldRatio = 1.0 / folds;
		int beginStep = 0;
		int endStep = 1;
		int length = featureData.rows();
		int testBlockSize = (int)(length * foldRatio);
		int beginIndex = (int)((double)length * foldRatio * (double)beginStep);
		int endIndex = (int)((double)length * foldRatio * (double)endStep);

		// Create train matrix
		Matrix trainFeatures = new Matrix((int)(featureData.rows() - featureData.rows()*foldRatio), featureData.cols());
		Matrix trainLabels = new Matrix((int)(featureData.rows() - featureData.rows()*foldRatio), featureData.cols());

		// Create test matrix
		Matrix testFeatures = new Matrix((int)(featureData.rows()*foldRatio), featureData.cols());
		Matrix testLabels = new Matrix((int)(featureData.rows()*foldRatio), labelData.cols());


		//int destRow, int destCol, Matrix that,
		//int rowBegin, int colBegin, int rowCount, int colCount

		System.out.println("feature data rows " + featureData.rows());
		System.out.println("trainFeatures rows: " + trainFeatures.rows());
		System.out.println("trainLabels rows: " + trainLabels.rows());
		System.out.println("testFeatures rows: " + testFeatures.rows());
		System.out.println("testLabels rows: " + testLabels.rows() + "\n");


		// Partition the data 5-fold
		for(int i = beginStep; i < folds; ++i) {
			int firstTrainBlockSize = beginIndex;
			int secondTrainBlockSize = featureData.rows() - endIndex - 1;

			System.out.println("Train 1st block: " + 0 + " to " + firstTrainBlockSize);
			System.out.println("test block " + beginIndex + " to " + endIndex);
			System.out.println("Train 2nd block: " +
				(firstTrainBlockSize + testBlockSize) + " to " + featureData.rows());
			System.out.println("test block size: " + testBlockSize);
			System.out.println("Train Size: " + (secondTrainBlockSize + firstTrainBlockSize));
			System.out.println("beginIndex: " + beginIndex);
			System.out.println("endIndex: " + endIndex);

			// First Training block
			trainFeatures.copyBlock(0, 0, featureData, 0, 0, firstTrainBlockSize, 13);
			// Test block
			testFeatures.copyBlock(0, 0, featureData, beginIndex, 0, testBlockSize, 13);
			testLabels.copyBlock(0, 0, featureData, beginIndex, 0, testBlockSize, 13);
			// 2nd Training block
			trainFeatures.copyBlock(firstTrainBlockSize, 0, featureData,
				firstTrainBlockSize + testBlockSize, 0, secondTrainBlockSize, 13);

			trainLabels.copyBlock(0, 0, featureData, 0, 0, firstTrainBlockSize, 13);
			trainLabels.copyBlock(firstTrainBlockSize, 0, featureData,
				firstTrainBlockSize + testBlockSize, 0, secondTrainBlockSize, 13);

			System.out.println(" BREAK ");

			//
			// Adjust the interval slicing
			++beginStep;
			++endStep;
			beginIndex = testBlockSize * beginStep;
			endIndex = testBlockSize * endStep;
		}

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
