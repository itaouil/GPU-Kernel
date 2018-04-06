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
		*inputs    = (float*) malloc(   M*sizeof(float) ),
		*weights   = (float*) malloc( N*M*sizeof(float) );
	initialiseArrays( gradients, inputs, weights, N, M );			// DO NOT REMOVE.


	//
	// Implement the GPU solution to the problem.
	//


	//
	// Output the result and clear up.
	//

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
