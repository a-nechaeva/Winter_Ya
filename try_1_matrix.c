#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>


const char *kernelSource = 
"__kernel void mat_mult(__global float *A, __global float *B, __global float *C, int N) {  \n"
"   int row = get_global_id(0);  \n"
"   int col = get_global_id(1);  \n"
"   float sum = 0.0;  \n"
"   for (int k = 0; k < N; k++) {  \n"
"       sum += A[row * N + k] * B[k * N + col];  \n"
"   }  \n"
"   C[row * N + col] = sum;  \n"
"}  \n";

void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10; // инициализация случайными числами
    }
}

int main() {
    int N = 4; // размер матрицы
    size_t bytes = N * N * sizeof(float);

    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);

    init_matrix(A, N);
    init_matrix(B, N);

    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_int err_ = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    char name[] = "UNSUCCESS";
    char name1[] = "SUCCESS";
    if (err_ != CL_SUCCESS) {
        printf("%s\n", name);
    } else {
        printf("%s\n", name1);
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, bytes, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, bytes, B, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "mat_mult", NULL);

    size_t globalWorkSize[2] = {N, N};
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&N);

    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

    // Вывод результата
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(b_mem_obj);
    clReleaseMemObject(c_mem_obj);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);

    return 0;
}
