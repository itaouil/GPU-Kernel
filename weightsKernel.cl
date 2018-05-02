// OpenCL kernel for weights update.
__kernel
void weightsUpdate(__global float *inputs, __global float *gradients, __global float *weights)
{
	// M local size
	__local int M;
	M = get_global_size(1);

	// Local gradient index
	__local int gradient_id;
    gradient_id = get_local_id(0);

    // Local input index
	__local int input_id;
	input_id = get_local_id(1);

	// Weights update
	weights[gradient_id*M+input_id] += gradients[gradient_id] + inputs[input_id];
}
