abstract class Layer
{
	protected Vec activation;

	Layer(size_t inputs, size_t outputs)
	{
		activation = new Vec(outputs);
	}

	abstract void activate(Vec weights, Vec x);
}
