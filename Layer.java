abstract class Layer
{
	protected Vec activation;

	Layer(int inputs, int outputs)
	{
		activation = new Vec(outputs);
	}

	abstract void activate(Vec weights, Vec x);
}
