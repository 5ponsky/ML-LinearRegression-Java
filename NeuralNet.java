import java.util.ArrayList;


public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  //protected ArrayList<Layer> layers;
  protected ArrayList<LayerLinear> layers;

  NeuralNet() {

  }

  String name() {
    return "Linear Regression";
  }

  Vec predict(Vec in) {
    // for(int i = 0; i < layers.size(); ++i) {
    //   layers.get(i).activate(weights, in);
    // }
    layers.get(0).activate(weights, in);
    System.out.println(layers.get(0).activation.toString());
    return layers.get(0).activation;

  }

  /// Train this supervised learner
  void train(Matrix features, Matrix labels) {
    for(int i = 0; i < layers.size(); ++i) {
      layers.get(i).ordinary_least_squares(features, labels, weights);
    }
  }

}
