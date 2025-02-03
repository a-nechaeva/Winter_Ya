#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

const char* kernelSource = 
"__kernel void vector_add(__global const float* A, __global const float* B, __global float* C, int N) {  \n"
"    int i = get_global_id(0);                                                          \n"
"    if (i < N) {                                                                       \n"
"        C[i] = A[i] + B[i];                                                           \n"
"    }                                                                                  \n"
"}                                                                                      \n";

int main() {
    int N = 1024;  // Размер векторов
    size_t bytes = N * sizeof(float);

    // Выделение памяти для векторов
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    // Инициализация векторов A и B
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // Инициализация OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    cl_int err_ = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    char name[] = "UNSUCCESS";
    char name1[] = "SUCCESS";
    if (err_ != CL_SUCCESS) {
        printf("%s\n", name);
    } else {
        printf("%s\n", name1);
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Создание буферов
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Создание и компиляция ядра
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

    // Установка аргументов
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // Запуск ядра
    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Чтение результата
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

    // Вывод результата
    for (int i = 0; i < N; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // Освобождение ресурсов
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(A);
    free(B);
    free(C);

    return 0;
}