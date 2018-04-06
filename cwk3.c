//
// Starting point for the GPU coursework. Please read coursework instructions before attempting this.
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>
#include "helper_cwk.h"			// Note this is not the same as the 'helper.h' used for examples.




//
// Main.
//
int main( int argc, char **argv )
{
	//
	// Initialisation.
	//

	// Initialise OpenCL. This is the same as the examples in lectures.
	cl_device_id device;
	cl_context context = simpleOpenContext_GPU(&device);

	cl_int status;
	cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

	// Get the parameters (N = no. of nodes/gradients, M = no. of inputs). getCmdLineArgs() is in helper_cwk.h.
	int N, M;
	getCmdLineArgs( argc, argv, &N, &M );

	// Initialise host arrays. initialiseArrays() is defined in helper_cwk.h. DO NOT REMOVE or alter this routine;
	// it will be replaced with a different version as part of the assessment.
	float
		*gradients = (float*) malloc( N  *sizeof(float) ),
		*inputs    = (float*) malloc( M*sizeof(float) ),
		*weights   = (float*) malloc( N*M*sizeof(float) );
	initialiseArrays( gradients, inputs, weights, N, M );			// DO NOT REMOVE.

	// Allocate and copy inputs array
	// from host to device memory
	cl_mem device_inputs = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		M*sizeof(float),
		inputs,
		&status
	);

	// Allocate and copy gradients array
	// from host to device memory
	cl_mem device_gradients = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		N*sizeof(float),
		gradients,
		&status
	);

	// Allocate and copy weights array
	// from host to device memory
	cl_mem device_weights = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
		N*M*sizeof(float),
		weights,
		&status
	);

	// Create kernel from .cl file
	cl_kernel kernel = compileKernelFromFile("weights_kernel.cl",
											 "weightsUpdate",
											 context,
										 	 device
										 	);

	// Specify arguments to the kernel
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_inputs);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_gradients);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_weights);

	//  Start kernel
	size_t globalSize [2] = {N,M};
	// size_t workGroupSize [2] = {N,M};

	// Put the kernel onto the command queue.
	status = clEnqueueNDRangeKernel(queue,
									kernel,
									2,
									0,
									globalSize,
									NULL,
									0,
									NULL,
									NULL);
	if( status != CL_SUCCESS )
	{
		printf( "Failure enqueuing kernel: Error %d.\n", status );
		return EXIT_FAILURE;
	}

	//
	// Get the result back from the device to the host, and check.
	//
	status = clEnqueueReadBuffer(queue,
								device_weights,
								CL_TRUE,
								0,
								N*M*sizeof(float),
								weights,
								0,
								NULL,
								NULL);
	if( status != CL_SUCCESS )
	{
		printf( "Could not copy device data to host: Error %d.\n", status );
		return EXIT_FAILURE;
	}

	// printf( "Checking (will only display first few elements):\n" );
	// for( i=0; i<N; i++ )
	// {
	// 	// Only print a few values.
	// 	if( i<10 ) printf( "%g + %g = %g.\n", host_a[i], host_b[i], host_c[i] );
	// 	if( host_a[i] + host_b[i] != host_c[i] )
	// 	{
	// 		printf( "Vector addition FAILED.\n" );
	// 		break;
	// 	}
	// }

	// Output result to screen. DO NOT REMOVE THIS LINE (or alter displayWeights() in helper_cwk.h); this will be replaced
	// with a different displayWeights() for the the assessment, so any changes you might make will be lost.
	displayWeights( weights, N, M) ;								// DO NOT REMOVE.

	free( gradients );
	free( inputs    );
	free( weights   );

	clReleaseCommandQueue( queue   );
	clReleaseContext     ( context );

	return EXIT_SUCCESS;
}
