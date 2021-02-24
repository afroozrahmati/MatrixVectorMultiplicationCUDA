/*
 * CSS-535 Lab 03: CUDA GEMV Implementation
 * Authors: Afrooz Rahmati & Tony Varela
 * 
 * Description: This is my (Tony) reimplementation of Afrooz's original code in C++ (thanks to her for getting this started!).
 * For now, let's focus on Part 0 - the naive implementation of GEMV using CUDA.
 */

// included header files 

// CUDA stuff 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h> // as a benchmark 

#include <random> // for random initialization
#include <chrono> // timing
#include <iostream> // for output 

/*
 * naive gemv kernel; each instance of this function is supposed to take one output vector element
 * this specific implementation does not rely on padding, but rather using an if (divergence).
 * M is for the rows, N is for the columns. And this assumes row major data.
 */
__global__ void naive_gemv(const float *A, const float *x, float *y, const size_t M, const size_t N) {
	const size_t total_thread_num{static_cast<size_t>(gridDim.x) * blockDim.x};
	const size_t tid{threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x};
	size_t stride{M / total_thread_num};
	if (stride == 0) { stride = 1; if (tid > M) return; } // if we're on a thread that's greater than the vector size, just get out
	// if we had a stride of 0, that means we have more threads than elements... just add a stride in there just in case.

// else, that means stride >= 1 (more elements than threads); if the current thread index is the LAST ONE, we need to consider the possible remainders. and ONLY IF we have more vector elements than threads.
	else if (tid == total_thread_num - 1) {
		stride += (M <= total_thread_num) ? 0 : M % total_thread_num;
	}
	for (auto i{tid * stride}; i < (tid*stride) + stride; i++) {
		y[i] = 0.0f;
		for (size_t j{0}; j < N; j++) y[i] += A[i * M + j] * x[j];
	}
}

// Credits to Brian Luger for the main structure of this program (just the way it is divided, I learned this from our time together on Lab 2)
int main(int argc, char **argv) {
	// TODO: create command line arguments to configure grid/block dimensions
	// This program should only take in the M and N dimensions; within the program, we figure out the execution configurations ourselves

	// cublas declarations
	cublasHandle_t cublas_handle;

	// for now, let's put the matrix/vector dimensions in here as well
	const size_t M{ 10 };
	const size_t N{ 10 };
	// yes, I know they're always going to be square, but I like separating M and N for my own understanding.
	// TODO: consider experimenting with thrust device/host vectors as well

	// seed RNG
	std::default_random_engine dre;
	dre.seed(3); // seeded for reproducibility
	const std::uniform_real_distribution<float> uniform_dist(-10, 10); // uniform distribution [-10, 10]

	// allocate host memory
	float *m{new float[M * N]};
	float *v_in{new float[N]};
	float *v_out_naive{new float[M]};
	float *v_out_cublas{new float[M]};

	// allocate device memory
	float *d_m, *d_v_in, *d_v_out_naive, *d_v_out_cublas;
	cudaMalloc(reinterpret_cast<void**>(&d_m), sizeof(float) * M * N);
	cudaMalloc(reinterpret_cast<void**>(&d_v_in), sizeof(float) * N);
	cudaMalloc(reinterpret_cast<void**>(&d_v_out_naive), sizeof(float) * M);
	cudaMalloc(reinterpret_cast<void**>(&d_v_out_cublas), sizeof(float) * M);

	// initialize host array with random data

	// for the matrix 
	for (size_t i{0}; i < M; i++) for (size_t j{0}; j < N; j++) m[i * M + j] = uniform_dist(dre);
	//std::cout << "Printing Matrix:\n";
	//for (size_t i{0}; i < M; i++) {
	//	for (size_t j{0}; j < N; j++) {
	//		std::cout << m[i * M + j] << ' ';
	//	}
	//	std::cout << '\n';
	//}
	// for the vector
	for (size_t i{0}; i < N; i++) v_in[i] = uniform_dist(dre);
	//std::cout << "Printing Input Vector:\n";
	//for (size_t i{0}; i < N; i++) std::cout << v_in[i] << ' ';

	std::cout << '\n';
	// copy m and v_in into device memory, time it as well
	auto d2h_start = std::chrono::high_resolution_clock::now();
	cudaMemcpy(d_m, m, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v_in, v_in, sizeof(float) * N, cudaMemcpyHostToDevice);
	auto d2h_end = std::chrono::high_resolution_clock::now();
	auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - d2h_start).count();

	// TODO: there are CUBLAS operations for getting/setting matrices/vectors between host/device; consider looking/timing these as well: https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf (pg.48-49)

	// let's create the grid / block configuration, but just really simply.
	dim3 grid(1); // (1, 1, 1)
	dim3 block(1);
	// (M, 1, 1); since each thread is in charge of ONE output element, and our output is a vector, we only need as many threads as vector elements !
	std::cout << "STARTING NAIVE" << std::endl;
	auto naive_exec_start = std::chrono::high_resolution_clock::now();
	naive_gemv <<<grid, block >>>(d_m, d_v_in, d_v_out_naive, M, N);
	cudaDeviceSynchronize();
	std::cout << "FINISHED NAIVE" << std::endl;
	// since the kernels are executed asynchronously, need to sync so that we can get accurate timing
	auto naive_exec_end = std::chrono::high_resolution_clock::now();
	auto naive_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(naive_exec_end - naive_exec_start).
		count();
	
	// copy d_v_out_naive back into host
	auto h2d_start = std::chrono::high_resolution_clock::now();
	cudaMemcpy(v_out_naive, d_v_out_naive, sizeof(float) * M, cudaMemcpyDeviceToHost);
	auto h2d_end = std::chrono::high_resolution_clock::now();
	auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(h2d_end - h2d_start).count();

	// get total inclusive time 
	auto gpu_transfer_total_duration = h2d_duration + d2h_duration;
	
	// try timing cublas (not timing inclusive times, although I am copying back out to host as well)
	cublasCreate(&cublas_handle);
	// cublasSetMatrix(M, N, sizeof(float), m, M, )

	const float a{1.0f};
	const float b{0.0f};
	auto cublas_exec_start = std::chrono::high_resolution_clock::now();
	cublasSgemv(cublas_handle, CUBLAS_OP_T, N, M, &a, d_m, N, d_v_in, 1, &b, d_v_out_cublas, 1);
	auto cublas_exec_end = std::chrono::high_resolution_clock::now();
	auto cublas_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
		cublas_exec_end - cublas_exec_start).count();

	// copy the cublas device vector back out to host
	cudaMemcpy(v_out_cublas, d_v_out_cublas, sizeof(float) * M, cudaMemcpyDeviceToHost);

	std::cout << "Comparing output vectors:\n";
	float rse{ 0.0f };
	for (size_t i{ 0 }; i < M; i++) rse += abs(v_out_naive[i] - v_out_cublas[i]);
	std::cout << "ERROR: " << rse << std::endl;
	//std::cout << "Naive: ";
	//for (size_t i{ 0 }; i < M; i++) std::cout << v_out_naive[i] << ' ';
	//std::cout << '\n';
	//
	//std::cout << "cuBLAS: ";
	//for (size_t i{0}; i < M; i++) std::cout << v_out_cublas[i] << ' ';
	//std::cout << '\n';

	std::cout <<
		"Total Inclusive Time, Naive Execution Time, cuBLAS Execution Time, Naive Total Time, cuBLAS Total Time\n";
	std::cout << gpu_transfer_total_duration << ", " << naive_exec_duration << ", " << cublas_exec_duration << ", " <<
		naive_exec_duration +
		gpu_transfer_total_duration << ", " << cublas_exec_duration + gpu_transfer_total_duration << '\n';

	// clean up
	cublasDestroy(cublas_handle);

	cudaFree(d_v_out_cublas);
	cudaFree(d_v_out_naive);
	cudaFree(d_v_in);
	cudaFree(d_m);

	delete[] v_out_cublas;
	delete[] v_out_naive;
	delete[] v_in;
	delete[] m;

	return 0;
}