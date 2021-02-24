/*
 * CSS-535 Lab 03: CUDA GEMV Implementation
 * Authors: Afrooz Rahmati & Tony Varela
 *
 * Description: This is my (Tony) reimplementation of Afrooz's original code in C++ (thanks to her for getting this started!).
 * Here, we focus on modifying the amount of registers used, via loop-unrolling.
 */

// included header files 

// CUDA stuff 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <random> // for random initialization
#include <chrono> // timing
#include <iostream> // for output
#include <string>

/*
 * naive gemv kernel; each instance of this function is supposed to take one output vector element
 * this specific implementation does not rely on padding, but rather using an if (divergence).
 * M is for the rows, N is for the columns. And this assumes row major data.
 */
__global__ void register_gemv(const float *A, const float *x, float *y, const size_t M, const size_t N) {
	const size_t total_thread_num{static_cast<size_t>(gridDim.x) * blockDim.x};
	const size_t tid{threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x};
	size_t stride{M / total_thread_num};

	if (stride == 0) {
		if (tid >= M) return;
		stride = 1;
	} // if we're on a thread that's greater than the vector size, just get out
	// if we had a stride of 0, that means we have more threads than elements... just add a stride in there just in case.

	// else, that means stride >= 1 (more elements than threads); if the current thread index is the LAST ONE, we need to consider the possible remainders. and ONLY IF we have more vector elements than threads.
	const size_t begin_index{tid * stride};
	size_t end_index{begin_index + stride};
	end_index += (tid == static_cast<size_t>(total_thread_num) - 1)
		             ? ((M <= total_thread_num) ? 0 : M % total_thread_num)
		             : 0;

	for (auto i{begin_index}; i < end_index; i++) {
		float sum{ 0.0f };
		/*y[i] = 0.0f;*/
		for (size_t j{0}; j < N; j += 8) {
			const float a0 = A[i * M + j] * x[j];
			const float a1 = A[i * M + j + 1] * x[j + 1];
			const float a2 = A[i * M + j + 2] * x[j + 2];
			const float a3 = A[i * M + j + 3] * x[j + 3];
			const float a4 = A[i * M + j + 4] * x[j + 4];
			const float a5 = A[i * M + j + 5] * x[j + 5];
			const float a6 = A[i * M + j + 6] * x[j + 6];
			const float a7 = A[i * M + j + 7] * x[j + 7];
			sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
		}
		y[i] = sum;
	}
}

// Credits to Brian Luger for the main structure of this program (just the way it is divided, I learned this from our time together on Lab 2)
int main(int argc, char **argv) {
	// TODO: create command line arguments to configure grid/block dimensions
	// This program should only take in the M and N dimensions; within the program, we figure out the execution configurations ourselves
	if (argc != 4) {
		std::cout << "Input: Vector_Size GridDim.x BlockDim.x. Exiting...\n";
		return -1;
	}
	int val;
	cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, 0);
	std::cout << val << std::endl;
	// for now, let's put the matrix/vector dimensions in here as well
	const size_t M{std::stoul(std::string{argv[1]})};
	const size_t N{std::stoul(std::string{argv[1]})};
	// let's create the grid / block configuration, but just really simply.
	dim3 grid{std::stoul(std::string{argv[2]})}; // (1, 1, 1)
	dim3 block{std::stoul(std::string{argv[3]})};

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
	// float *v_out_cublas{new float[M]};

	// allocate device memory
	float *d_m, *d_v_in, *d_v_out_naive;
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaMalloc(reinterpret_cast<void**>(&d_m), sizeof(float) * M * N))
		<< std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaMalloc(reinterpret_cast<void**>(&d_v_in), sizeof(float) * N))
		<< std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(
		cudaMalloc(reinterpret_cast<void**>(&d_v_out_naive), sizeof(float) * M)) << std::endl;

	// initialize host array with random data

	// for the matrix 
	for (size_t i{0}; i < M; i++) for (size_t j{0}; j < N; j++) m[i * M + j] = uniform_dist(dre);
	for (size_t i{0}; i < N; i++) v_in[i] = uniform_dist(dre);

	// copy m and v_in into device memory, time it as well
	auto d2h_start = std::chrono::high_resolution_clock::now();
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaMemcpy(d_m, m, sizeof(float) * M * N, cudaMemcpyHostToDevice))
		<< std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(
		cudaMemcpy(d_v_in, v_in, sizeof(float) * N, cudaMemcpyHostToDevice)) << std::endl;
	auto d2h_end = std::chrono::high_resolution_clock::now();
	auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - d2h_start).count();

	auto naive_exec_start = std::chrono::high_resolution_clock::now();
	register_gemv << <grid, block >> >(d_m, d_v_in, d_v_out_naive, M, N);
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
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

	std::cout <<
		"Total Inclusive Time, Naive Execution Time, Naive Total Time\n";
	std::cout << gpu_transfer_total_duration << ", " << naive_exec_duration << ", " <<
		naive_exec_duration +
		gpu_transfer_total_duration << ", " << '\n';

	// clean up
	cudaFree(d_v_out_naive);
	cudaFree(d_v_in);
	cudaFree(d_m);

	delete[] v_out_naive;
	delete[] v_in;
	delete[] m;

	return 0;
}
