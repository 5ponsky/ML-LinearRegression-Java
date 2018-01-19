


public class LayerLinear extends Layer {


  LayerLinear() {

  }

  void activate(Vec weights, Vec x) {
    // linear equation activation: Mx + b
    // x is a vector of size inputs
    // activation is a vector of size outputs
    // M is a matrix with "outputs" rows and "inputs" columns
    // b is a vector of size outputs
    // vector 'weights' contains all values needed to fill both M and b
    // The number of elements in weights will be outputs + outputs*inputs

    // M is our parameters, or weights ( M = weights)

    //activation =

    // Vec(Vec v, int begin, int length)
    Vec b = new Vec(weights, 0, outputs);
  }


}
