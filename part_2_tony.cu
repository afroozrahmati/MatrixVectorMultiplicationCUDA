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
#include <cublas_v2.h>

#include <random> // for random initialization
#include <chrono> // timing
#include <iostream> // for output
#include <string>

/*
 * With shared memory, we want to take advantage of locality.
 * The only reasonable thing to share, imo, among these threads is the input vector x.
 * Ideally, if each kernel/thread is only in charge of one element, then my only worry is to make sure that
 * the input vector x is shared AMAP amongst the threads of a block, so they dont have to pull it from global memory...
 * Since we're basically testing only 10000 elements at this point, that'd be around 40KB of data, which should fit into the default
 * configured shared memory size.
 *
 * But, I wouldnt want just ONE kernel loading everything... I need each kernel to load a fair portion of the vector into shared memory.
 * Which basically just ends up being another same scheme as what i already had for gemv itself...
 */
__global__ void shared_gemv(const float *A, const float *x, float *y, const size_t M, const size_t N) {
	const size_t total_thread_num{ static_cast<size_t>(gridDim.x) * blockDim.x };
	const size_t tid{ threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x };
	
	extern __shared__ float x_shared[]; // extern shared because we dont know the shared memory size at compile time 

	// remember, shared memory is PER BLOCK.

	// let's get the TPB count - threads per block, which should just be blockDim.x
	const size_t tpb{ blockDim.x };
	const size_t tidx{ threadIdx.x };
	size_t x_stride{ N / tpb };
	if (x_stride == 0) {
		if (tidx >= N) return; // i think this means... if the current thread within the block is greater than N, then dont bother loading anything in.
		x_stride = 1;
	}
	const size_t x_begin_index{tidx*x_stride};
	size_t x_end_index{ x_begin_index + x_stride };
	x_end_index += (tidx == static_cast<size_t>(tpb) - 1) // if you are the last thread within the block...
		? ((N <= tpb) ? 0 : N % tpb)
		: 0;

	// printf("THIS KERNEL IS LOADING FROM %llu TO %llu from X\n", x_begin_index, x_end_index);

	// this is the ultimate goal of what I want before proceeding to the rest of the calculating 
	for (size_t i{ x_begin_index }; i < x_end_index; i++) {
		x_shared[i] = x[i];
	}

	__syncthreads(); // make sure all of the data is loaded before proceeding 
	

	size_t stride{M / total_thread_num};

	if (stride == 0) {
		if (tid >= M) return;
		stride = 1;
	}
	// else, that means stride >= 1 (more elements than threads); if the current thread index is the LAST ONE, we need to consider the possible remainders. and ONLY IF we have more vector elements than threads.
	const size_t begin_index{tid * stride};
	size_t end_index{begin_index + stride};
	end_index += (tid == static_cast<size_t>(total_thread_num) - 1)
		             ? ((M <= total_thread_num) ? 0 : M % total_thread_num)
		             : 0;
	for (size_t i{begin_index}; i < end_index; i++) {
		/*y[i] = 0.0f;*/
		float sum{0.0f};
		for (size_t j{0}; j < N; j+=8) {
			sum += A[i * M + j] * x[j];
			sum += A[i * M + j + 1] * x_shared[j + 1];
			sum += A[i * M + j + 2] * x_shared[j + 2];
			sum += A[i * M + j + 3] * x_shared[j + 3];
			sum += A[i * M + j + 4] * x_shared[j + 4];
			sum += A[i * M + j + 5] * x_shared[j + 5];
			sum += A[i * M + j + 6] * x_shared[j + 6];
			sum += A[i * M + j + 7] * x_shared[j + 7];
		}
		y[i] = sum;
	}
}



//__global__ void mat_mul_tiled(float* vec, float* mat, float* res, const int N) {
//
//    __shared__  float As[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__  float Bs[BLOCK_SIZE];
//	
//    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y; //the row index of As and Bs
//    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; //the column index of As and Bs
//    float tmp = 0;
//    int idx;
//
//
//    for (int i = 0; i < gridDim.y; ++i) //initialize the ID index
//    {
//		
//        idx = row * N + i * BLOCK_SIZE + threadIdx.x;
//		
//        //if N is not divisible by block width
//        if (idx >= N*N)
//        {
//            As[threadIdx.y][threadIdx.x] = 0; 
//        }
//        else{
//            As[threadIdx.y][threadIdx.x] = mat[idx];   
//        }
//
//        idx = col + i * BLOCK_SIZE;
//
//
//        //if N is not divisible by block width
//        if(idx >= N)  
//        {
//            Bs[threadIdx.x] = 0;
//        }
//        else{
//            Bs[threadIdx.x] = vec[idx];  
//        }
//
//		
//
//        //Matrix and vectors should be loaded completely before any further process
//        __syncthreads();
//
//        //multiply sub matrices
//        for (int k = 0; k < BLOCK_SIZE; k++) 
//        {
//            tmp += As[threadIdx.y][k] * Bs[k];
//			
//        }
//
//        __syncthreads();
//    }
//
//    //write result back to global memory
//    if(row < N && col < N)
//    {
//        res[row] = tmp;  
//    }
//}

// Credits to Brian Luger for the main structure of this program (just the way it is divided, I learned this from our time together on Lab 2)
int main(int argc, char **argv) {
	// TODO: create command line arguments to configure grid/block dimensions
	// This program should only take in the M and N dimensions; within the program, we figure out the execution configurations ourselves
	if (argc != 4) {
		std::cout << "Input: Vector_Size GridDim.x BlockDim.x. Exiting...\n";
		return -1;
	}

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

	// cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	// allocate host memory
	float *m{new float[M * N]};
	float *v_in{new float[N]};
	float *v_out_naive{new float[M]};
	float *v_out_cublas{new float[M]};

	// allocate device memory
	float *d_m, *d_v_in, *d_v_out_naive, *d_v_out_cublas;
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaMalloc(reinterpret_cast<void**>(&d_m), sizeof(float) * M * N))
		<< std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaMalloc(reinterpret_cast<void**>(&d_v_in), sizeof(float) * N))
		<< std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(
		cudaMalloc(reinterpret_cast<void**>(&d_v_out_naive), sizeof(float) * M)) << std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(
		cudaMalloc(reinterpret_cast<void**>(&d_v_out_cublas), sizeof(float) * M)) << std::endl;
	// initialize host array with random data

	// for the matrix 
	for (size_t i{ 0 }; i < M; i++) for (size_t j{ 0 }; j < N; j++) m[i * M + j] = (i + j) % 2 + 1;//uniform_dist(dre);
	for (size_t i{ 0 }; i < N; i++) v_in[i] = (i % 2) + 1;//uniform_dist(dre);

	// copy m and v_in into device memory, time it as well
	auto d2h_start = std::chrono::high_resolution_clock::now();
	std::cout << "ERROR CODE: " << cudaGetErrorString(cudaMemcpy(d_m, m, sizeof(float) * M * N, cudaMemcpyHostToDevice))
		<< std::endl;
	std::cout << "ERROR CODE: " << cudaGetErrorString(
		cudaMemcpy(d_v_in, v_in, sizeof(float) * N, cudaMemcpyHostToDevice)) << std::endl;
	auto d2h_end = std::chrono::high_resolution_clock::now();
	auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - d2h_start).count();

	auto naive_exec_start = std::chrono::high_resolution_clock::now();
	shared_gemv << <grid, block, N*sizeof(float) >> >(d_m, d_v_in, d_v_out_naive, M, N);
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

	// cublas stuff
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	float a{1.0f};
	float b{0.0f};
	auto cublas_exec_start = std::chrono::high_resolution_clock::now();
	/*naive_gemv << <grid, block >> > (d_m, d_v_in, d_v_out_naive, M, N);*/
	cublasSgemv(cublas_handle, CUBLAS_OP_T, N, M, &a, d_m, N, d_v_in, 1, &b, d_v_out_cublas, 1);
	// since the kernels are executed asynchronously, need to sync so that we can get accurate timing
	auto cublas_exec_end = std::chrono::high_resolution_clock::now();
	auto cublas_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
			cublas_exec_end - cublas_exec_start).
		count();

	// copy d_v_out_naive back into host
	cudaMemcpy(v_out_cublas, d_v_out_cublas, sizeof(float) * M, cudaMemcpyDeviceToHost);

	float rse{0.0f};
	for (size_t i{0}; i < M; i++) rse += abs(v_out_cublas[i] - v_out_naive[i]);

	std::cout << "RSE: " << rse << std::endl;

	std::cout <<
		"Total Inclusive Time, Naive Execution Time, Naive Total Time\n";
	std::cout << gpu_transfer_total_duration << ", " << naive_exec_duration << ", " <<
		naive_exec_duration +
		gpu_transfer_total_duration << '\n';

	// clean up
	cublasDestroy(cublas_handle);
	cudaFree(d_v_out_cublas);
	cudaFree(d_v_out_naive);
	cudaFree(d_v_in);
	cudaFree(d_m);

	delete[] v_out_naive;
	delete[] v_in;
	delete[] m;

	return 0;
}


///*
// * CSS-535 Lab 03: CUDA GEMV Implementation
// * Authors: Afrooz Rahmati & Tony Varela
// */
//
//
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <cublas_v2.h> // as a benchmark 
//
//
//#include <random> // for random initialization
//#include <chrono> // timing
//#include <iostream> // for output 
//using namespace std;
//using namespace std::chrono;
//
////Matrix initialize with random values
///* void initialize_matrix(float *a, const int N, const int M) {
//        int i, j;
//        for(i=0; i<N; i++)
//            for(j=0; j<M; j++)
//                    a[i*M+j] = rand() % 4 + 1;
//} */
//
////Vector initialize with random values
///* void initialize_vector(float *a, const int N) {        
//        for(int i=0; i<N; i++)
//            a[i] = rand() % 4 + 1;
//} */
//
////GEMV naive Implementation
///* __global__ void multiplication(float *vec, float *mat, float *res, const int N, const int M){
//    
//    int tid= threadIdx.x+ blockIdx.x*blockDim.x;
//    float sum=0.0f;
//    
//    if(tid<M){
//        for(int i=0; i<N; i++)
//            sum += vec[i]*mat[(i*M)+tid];
//        res[tid]=sum;
//    }
//} */
//
///* 
//functionality: GEMV Implementation---Tiled version
//
//input parameters: 
//            vec         : the input vector 
//            mat         : the input matrix
//            res         : the result vector
//            N           : Matrix and Vector Size ( number of elements )
//            BLOCK_WIDTH : The GPU device block size
//consideration : the matrix size is square */
//
//#define BLOCK_SIZE  16  //should be change
//
//__global__ void mat_mul_tiled(float *vec, float *mat, float *res, const int N ){
//
//    __shared__  float As[BLOCK_SIZE][BLOCK_SIZE];
//    __shared__  float Bs[BLOCK_SIZE];
//	
//    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y; //the row index of As and Bs
//    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x; //the column index of As and Bs
//    float tmp = 0;
//    int idx;
//
//
//    for (int i = 0; i < gridDim.y; ++i) //initialize the ID index
//    {
//		
//        idx = row * N + i * BLOCK_SIZE + threadIdx.x;
//		
//        //if N is not divisible by block width
//        if (idx >= N*N)
//        {
//            As[threadIdx.y][threadIdx.x] = 0; 
//        }
//        else{
//            As[threadIdx.y][threadIdx.x] = mat[idx];   
//        }
//
//        idx = col + i * BLOCK_SIZE;
//
//
//        //if N is not divisible by block width
//        if(idx >= N)  
//        {
//            Bs[threadIdx.x] = 0;
//        }
//        else{
//            Bs[threadIdx.x] = vec[idx];  
//        }
//
//		
//
//        //Matrix and vectors should be loaded completely before any further process
//        __syncthreads();
//
//        //multiply sub matrices
//        for (int k = 0; k < BLOCK_SIZE; k++) 
//        {
//            tmp += As[threadIdx.y][k] * Bs[k];
//			
//        }
//
//        __syncthreads();
//    }
//
//    //write result back to global memory
//    if(row < N && col < N)
//    {
//        res[row] = tmp;  
//    }
//}
//
//
//
////printing the vector 
//void print_vector(float *a, const int N, char *d) {
//    int i;
//    for(i=0; i<N; i++)
//            printf("\n%s[%d]: %f",d, i, a[i]);
//    printf("\n");
//}
//
////printing the matrix values
//void print_matrix(float *a, const int N, const int M, char *d) {
//    int i, j;
//    for(i=0; i<N; i++){
//    printf("\n%s[%d]:", d, i);
//    for (j=0; j<M; j++)
//        printf("\t%6.4f", a[i*M+j]);
//    }
//    printf("\n");
//}
//
//
//
//// Credits to Brian Luger for the main structure of this program (just the way it is divided, I learned this from our time together on Lab 2)
//int main(int argc, char **argv) {
//	// TODO: create command line arguments to configure grid/block dimensions
//	// This program should only take in the M and N dimensions; within the program, we figure out the execution configurations ourselves
//
//	// cublas declarations
//	cublasHandle_t cublas_handle;
//
//	// for now, let's put the matrix/vector dimensions in here as well
//	const size_t M{ 10000 };
//	const size_t N{ 10000 };
//	// yes, I know they're always going to be square, but I like separating M and N for my own understanding.
//	// TODO: consider experimenting with thrust device/host vectors as well
//
//	// seed RNG
//	std::default_random_engine dre;
//	dre.seed(3); // seeded for reproducibility
//	std::uniform_real_distribution<float> uniform_dist(-10, 10); // uniform distribution [-10, 10]
//
//	// allocate host memory
//	float *m{new float[M * N]};
//	float *v_in{new float[N]};
//	float *v_out_naive{new float[M]};
//	float *v_out_cublas{new float[M]};
//
//	// allocate device memory
//	float *d_m, *d_v_in, *d_v_out_naive, *d_v_out_cublas;
//	cudaMalloc(reinterpret_cast<void**>(&d_m), sizeof(float) * M * N);
//	cudaMalloc(reinterpret_cast<void**>(&d_v_in), sizeof(float) * N);
//	cudaMalloc(reinterpret_cast<void**>(&d_v_out_naive), sizeof(float) * M);
//	cudaMalloc(reinterpret_cast<void**>(&d_v_out_cublas), sizeof(float) * M);
//
//    //**************************These lines are for debugging purpose only************************
//
////   /* The elements of the first column */
////  m[0] = 1;
////  m[1] = 2;
//// 	m[2] = 3;
//// 	m[3] = 4;
////    /* The elements of the second column */
//	
//
//// 	m[N] = 1;
//// 	m[N + 1] = 1;
//// 	m[N + 2] = 2;
//// 	m[N + 3] = 1;
////    /* The elements of the third column */
//// 	m[N * 2] = 3;
//// 	m[N * 2 + 1] = 1;
//// 	m[N * 2 + 2] = 2;
//// 	m[N * 2 + 3] = 1;
////    /* The elements of the fourth column */
//// 	m[N * 3] = 5;
//// 	m[N * 3 + 1] = 4;
//// 	m[N * 3 + 2] = 7;
//// 	m[N * 3 + 3] = 3;
//
//
//
//
//
////    /* The elements of x and y */
////    v_in[0] = 1;
////    v_in[1] = 3;
////    v_in[2] = 1;
////    v_in[3] = 2;
//
/////////////////////////**************************************************
//
//	// initialize host array with random data
//
//	//for the matrix 
//	for (size_t i{0}; i < M; i++) for (size_t j{0}; j < N; j++) 
//      m[i * M + j] = uniform_dist(dre);
//
//    //print_matrix(m, N, M, "input Matrix");  
//	
//	//std::cout << "Printing Matrix:\n";
//	//for (size_t i{0}; i < M; i++) {
//	//	for (size_t j{0}; j < N; j++) {
//	//		std::cout << m[i * M + j] << ' ';
//	//	}
//	//	std::cout << '\n';
//	//}
//	// for the vector
//	for (size_t i{0}; i < N; i++) v_in[i] = uniform_dist(dre);
//	//std::cout << "Printing Input Vector:\n";
//	//for (size_t i{0}; i < N; i++) std::cout << v_in[i] << ' ';
//
//    //print_vector(v_in, N, "input vector");
//
//	std::cout << '\n';
//	// copy m and v_in into device memory, time it as well
//	auto d2h_start = std::chrono::high_resolution_clock::now();
//	cudaMemcpy(d_m, m, sizeof(float) * M * N, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_v_in, v_in, sizeof(float) * N, cudaMemcpyHostToDevice);
//	auto d2h_end = std::chrono::high_resolution_clock::now();
//	auto d2h_duration = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - d2h_start).count();
//
//	// TODO: there are CUBLAS operations for getting/setting matrices/vectors between host/device; consider looking/timing these as well: https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf (pg.48-49)
//
//	// let's create the grid / block configuration, but just really simply.
//	
//    //*****************************************************************************************
//    /////////////////specific to part 2////////////////////////////////////////////////////////////////
//    //const unsigned int BLOCK_SIZE = 16; ///we need to change it
//    unsigned int gridrows =  (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
//
//    dim3 grid ( 1 , gridrows );  
//    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
//    
//
//	std::cout << "STARTING NAIVE" << std::endl;
//	auto naive_exec_start = std::chrono::high_resolution_clock::now();
//    mat_mul_tiled<<<grid, block>>>( d_v_in, d_m, d_v_out_naive, M);
//   
//    //naive_gemv <<<grid, block >>>(d_m, d_v_in, d_v_out_naive, M, N);
//	cudaDeviceSynchronize();
//	std::cout << "FINISHED NAIVE" << std::endl;
//	// since the kernels are executed asynchronously, need to sync so that we can get accurate timing
//	auto naive_exec_end = std::chrono::high_resolution_clock::now();
//	auto naive_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(naive_exec_end - naive_exec_start).
//		count();
//	
//
//    //print_vector(d_v_out_naive, M, "out vector");
//
//
//	// copy d_v_out_naive back into host
//	auto h2d_start = std::chrono::high_resolution_clock::now();
//	cudaMemcpy(v_out_naive, d_v_out_naive, sizeof(float) * M, cudaMemcpyDeviceToHost);
//	auto h2d_end = std::chrono::high_resolution_clock::now();
//	auto h2d_duration = std::chrono::duration_cast<std::chrono::microseconds>(h2d_end - h2d_start).count();
//
//	// get total inclusive time 
//	auto gpu_transfer_total_duration = h2d_duration + d2h_duration;
//	
//	// try timing cublas (not timing inclusive times, although I am copying back out to host as well)
//	cublasCreate(&cublas_handle);
//	// cublasSetMatrix(M, N, sizeof(float), m, M, )
//
//	const float a{1.0f};
//	const float b{0.0f};
//	auto cublas_exec_start = std::chrono::high_resolution_clock::now();
//	cublasSgemv(cublas_handle, CUBLAS_OP_T, N, M, &a, d_m, N, d_v_in, 1, &b, d_v_out_cublas, 1);
//
//
//	auto cublas_exec_end = std::chrono::high_resolution_clock::now();
//	auto cublas_exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
//		cublas_exec_end - cublas_exec_start).count();
//
//	// copy the cublas device vector back out to host
//	cudaMemcpy(v_out_cublas, d_v_out_cublas, sizeof(float) * M, cudaMemcpyDeviceToHost);
//
//	std::cout << "Comparing output vectors:\n";
//	float rse{ 0.0f };
//	for (size_t i{ 0 }; i < M; i++) rse += abs(v_out_naive[i] - v_out_cublas[i]);
//	std::cout << "ERROR: " << rse << std::endl;
//
/////////
//
////print_vector(d_v_out_cublas, M, "out vector");
//
//
//	// std::cout << "Naive: ";
//	// for (size_t i{ 0 }; i < M; i++) std::cout << v_out_naive[i] << ' ';
//	// std::cout << '\n';
//	
//	// std::cout << "cuBLAS: ";
//	// for (size_t i{0}; i < M; i++) std::cout << v_out_cublas[i] << ' ';
//	// std::cout << '\n';
//
//
//
//
//	std::cout <<
//		"Total Inclusive Time, Naive Execution Time, cuBLAS Execution Time, Naive Total Time, cuBLAS Total Time\n";
//	std::cout << gpu_transfer_total_duration << ", " << naive_exec_duration << ", " << cublas_exec_duration << ", " <<
//		naive_exec_duration +
//		gpu_transfer_total_duration << ", " << cublas_exec_duration + gpu_transfer_total_duration << '\n';
//
//	// clean up
//	cublasDestroy(cublas_handle);
//
//	cudaFree(d_v_out_cublas);
//	cudaFree(d_v_out_naive);
//	cudaFree(d_v_in);
//	cudaFree(d_m);
//
//	delete[] v_out_cublas;
//	delete[] v_out_naive;
//	delete[] v_in;
//	delete[] m;
//
//	return 0;
//}
//
