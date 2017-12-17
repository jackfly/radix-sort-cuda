#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string> 
#include <sstream>
#include <math.h>
#include <time.h>

#include "sort.h"
#include "utils.h"
using namespace std;

void test_cpu_vs_gpu(unsigned int* h_in, unsigned int num_elems)
{
    unsigned int* h_out_gpu = new unsigned int[num_elems];
    
    unsigned int* d_in;
    unsigned int* d_out;
    checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * num_elems));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * num_elems));
    checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));

    radix_sort(d_out, d_in, num_elems);

    checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_in));

    
    delete[] h_out_gpu;
}

int main()
{
    struct timespec start, stop;
    for (int i = 0; i <= 18;)
    {
        i = i + 2;
        int num = pow(2,i);
        int linecount = 0;
        string filename = "../RandomNumbers/" + std::to_string(i) + ".txt";
        unsigned int* numbers = new unsigned int[num];
        //int numbers[num];
        //Create an input file stream
        fstream file(filename);
        for(int m=0; m<num; m++)
        {
            file >> numbers[linecount];
            linecount++;
        }
        //Close the file stream
        file.close();

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        test_cpu_vs_gpu(numbers, num);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double dt = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;    // in microseconds
        printf("@time of CUDA run:\t\t\t[%.3f] microseconds\n", dt);

        delete[] numbers;

    }
}
