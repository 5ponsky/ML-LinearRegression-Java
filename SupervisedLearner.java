// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

abstract class SupervisedLearner
{
	/// Return the name of this learner
	abstract String name();

	/// Train this supervised learner
	abstract void train(Matrix features, Matrix labels);

	/// Make a prediction
	abstract Vec predict(Vec in);

	double mis;

	void cross_validation() {

	}

	double sum_squared_error(Matrix features, Matrix labels) {
		if(features.rows() != labels.rows())
			throw new IllegalArgumentException("Mistmatching number of rows");

			for(int i = 0; i < features.rows(); i++) {
				Vec feat = features.row(i);
				Vec pred = predict(feat);
				Vec lab = labels.row(i);
				for(int j = 0; j < lab.size(); j++) {
					//Vec temp = new Vec(label.get(j));
					//temp.addScaled(-1, pred.get(j));
					//System.out.println(label.get(j).toString() + " " + pred.get(j).toString());
					mis = mis + Math.pow(lab.get(j) - pred.get(j), 2);
					//System.out.println(mis);
				}
			}

			return mis;
		//return (mis / features.rows());
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
