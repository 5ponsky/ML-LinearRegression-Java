import java.util.ArrayList;


public class NeuralNet extends SupervisedLearner {
  protected Vec weights;
  //protected ArrayList<Layer> layers;
  protected ArrayList<LayerLinear> layers;

  NeuralNet() {
    layers = new ArrayList<LayerLinear>();
    layers.add(new LayerLinear(13, 13));
    weights = new Vec(182);

  }

  String name() {
    return "Linear Regression";
  }

  Vec predict(Vec in) {
    //layers.add(new LayerLinear(in.size(), 1));
    layers.get(0).activate(weights, in);
    //System.out.println("activation: " + layers.get(0).activation.toString());
    System.out.println(weights);
    return layers.get(0).activation;
  }

  /// Train this supervised learner
  void train(Matrix features, Matrix labels) {
    for(int i = 0; i < layers.size(); ++i) {
      layers.get(i).ordinary_least_squares(features, labels, weights);
    }
  }

}
