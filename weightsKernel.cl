// OpenCL kernel for weights update.
__kernel
void weightsUpdate(__global float *inputs, __global float *gradients, __global float *weights)
{
	// The global size (M)
	int M = get_local_size(0);

    // Get local id for inputs and gradients
    int input_id = get_local_id(1);
    int gradient_id = get_local_id(0);

	// Perform weights update
	weights[gradient_id*M+input_id] += gradients[gradient_id] + inputs[input_id];
}
