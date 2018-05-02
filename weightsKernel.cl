// OpenCL kernel for weights update.
__kernel
void weightsUpdate(__global float *inputs, __global float *gradients, __global float *weights)
{
	// M local size
	__local int M;
	M = get_global_size(1);

    // Local input id
	__local int input_id;
	input_id = get_local_id(1);

	// Local gradient id
	__local int gradient_id;
    gradient_id = get_local_id(0);

	// Perform weights update
	weights[gradient_id*M+input_id] += gradients[gradient_id] + inputs[input_id];
}
