// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;

abstract class SupervisedLearner
{
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels);

	/// Make a prediction
	abstract Vec predict(Vec in);

	double cross_validation(int r, int f, Matrix featureData, Matrix labelData) {
		Random random = new Random(1234);

		// Cross-Validation indices
		int repititions = r;
		int folds = f;
		double foldRatio = 1.0 / (double)folds;
		int beginStep = 0;
		int endStep = 1;
		int testBlockSize = (int)(featureData.rows() * foldRatio);
		int beginIndex = 0;
		int endIndex = 0;

		// Create train matrices
		Matrix trainFeatures = new Matrix((int)(featureData.rows() - Math.floor(featureData.rows()*foldRatio)), featureData.cols());
		Matrix trainLabels = new Matrix((int)(featureData.rows() - Math.floor(featureData.rows()*foldRatio)), labelData.cols());

		// Create test matrices
		Matrix testFeatures = new Matrix((int)(featureData.rows()*foldRatio), featureData.cols());
		Matrix testLabels = new Matrix((int)(featureData.rows()*foldRatio), labelData.cols());


		// Partition the data by folds
		double sse = 0; // Sum squared error
		double mse = 0; // Mean squared error
		double rmse = 0; // Root mean squared error


		for(int k = 0; k < repititions; ++k) {
			for(beginStep = 0; beginStep < folds; ++beginStep) {
				beginIndex = beginStep * (featureData.rows() / folds);
				endIndex = (beginStep + 1) * (featureData.rows() / folds);

				// First Training block
				trainFeatures.copyBlock(0, 0, featureData, 0, 0, beginIndex, featureData.cols());
				trainLabels.copyBlock(0, 0, labelData, 0, 0, beginIndex, labelData.cols());


				// Test block
				testFeatures.copyBlock(0, 0, featureData, beginIndex, 0, endIndex-beginIndex, featureData.cols());
				testLabels.copyBlock(0, 0, labelData, beginIndex, 0, endIndex-beginIndex, labelData.cols());


				// 2nd Training block
				trainFeatures.copyBlock(beginIndex, 0, featureData,
					beginIndex+1, 0, featureData.rows() - endIndex, featureData.cols());
				trainLabels.copyBlock(beginIndex, 0, labelData,
					beginIndex+1, 0, featureData.rows() - endIndex, labelData.cols());


				train(trainFeatures, trainLabels);
				sse = sse + sum_squared_error(testFeatures, testLabels);
			}

			System.out.println("SSE: " + sse);
			mse = mse + (sse / featureData.rows());
			sse = 0;

			for(int i = 0; i < featureData.rows(); ++i) {
				int selectedRow = random.nextInt(featureData.rows());
				int destinationRow = random.nextInt(featureData.rows());
				featureData.swapRows(selectedRow, destinationRow);
				labelData.swapRows(selectedRow, destinationRow);
			}
		}


		System.out.println("MSE: " + mse);
		rmse = Math.sqrt(mse/repititions);
		return rmse;
	}

	double sum_squared_error(Matrix features, Matrix labels) {
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mistmatching number of rows");

		double mis = 0;
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = labels.row(i);
			for(int j = 0; j < lab.size(); j++) {
				mis = mis + (lab.get(j) - pred.get(j)) * (lab.get(j) - pred.get(j));
			}
		}

		return mis;
	}

	/// Measures the misclassifications with the provided test data
	int countMisclassifications(Matrix features, Matrix labels) {
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mismatching number of rows");
		int mis = 0;
		for(int i = 0; i < features.rows(); i++) {
			Vec feat = features.row(i);
			Vec pred = predict(feat);
			Vec lab = labels.row(i);
			for(int j = 0; j < lab.size(); j++) {
				if(pred.get(j) != lab.get(j))
					mis++;
			}
		}
		return mis;
	}
}
