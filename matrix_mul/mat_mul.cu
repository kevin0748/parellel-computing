#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 16
#define MAX_DEPTH 20
#define DEBUG 0
#define VERIFY 1
float *d_A[MAX_DEPTH], *d_B[MAX_DEPTH], *d_C;
float *d_M1[MAX_DEPTH], *d_M2[MAX_DEPTH], *d_M3[MAX_DEPTH], *d_M4[MAX_DEPTH], *d_M5[MAX_DEPTH], *d_M6[MAX_DEPTH], *d_M7[MAX_DEPTH];

/**
 * @brief
 *
 * @param A
 * @param ax
 * @param ay
 * @param B
 * @param bx
 * @param by
 * @param C
 * @param cx
 * @param cy
 * @param N stride size
 * @param len subMatrix length
 * @param subtract
 * @return __global__
 */
__global__ void MatAddKernel(float *A, int ax, int ay,
                             float *B, int bx, int by,
                             float *C, int cx, int cy,
                             int N, int len, bool subtract)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // perform the addition only if the indices are within bounds
    if (row < len && col < len)
    {
        if (subtract)
        {
            C[(cy + row) * N + cx + col] = A[(ay + row) * N + ax + col] - B[(by + row) * N + bx + col];
        }
        else
        {
            C[(cy + row) * N + cx + col] = A[(ay + row) * N + ax + col] + B[(by + row) * N + bx + col];
        }
    }
}

void MatAdd(float *A, int ax, int ay,
            float *B, int bx, int by,
            float *C, int cx, int cy,
            int N, int len, bool subtract)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N + BLOCK_SIZE - 1 / dimBlock.x, N + BLOCK_SIZE - 1 / dimBlock.y);
    MatAddKernel<<<dimGrid, dimBlock>>>(A, ax, ay,
                                        B, bx, by,
                                        C, cx, cy,
                                        N, len, subtract);
}

__global__ void MatMulKernel(float *A, int ax, int ay,
                             float *B, int bx, int by,
                             float *C, int cx, int cy,
                             int N, int len)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < len && col < len)
    {
        for (int e = 0; e < len; ++e)
            Cvalue += A[(ay + row) * N + ax + e] * B[(by + e) * N + bx + col];
        C[(cy + row) * N + cx + col] = Cvalue;
    }
}

void MatMul(float *A, int ax, int ay,
            float *B, int bx, int by,
            float *C, int cx, int cy,
            int N, int len)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N + BLOCK_SIZE - 1 / dimBlock.x, N + BLOCK_SIZE - 1 / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(A, ax, ay,
                                        B, bx, by,
                                        C, cx, cy,
                                        N, len);
}

__global__ void MatCopyKernel(float *A, int ax, int ay,
                              float *B, int bx, int by,
                              int N, int len)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // perform the addition only if the indices are within bounds
    if (row < len && col < len)
    {
        B[(by + row) * N + bx + col] = A[(ay + row) * N + ax + col];
    }
}

void MatCopy(float *A, int ax, int ay,
             float *B, int bx, int by,
             int N, int len)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N + BLOCK_SIZE - 1 / dimBlock.x, N + BLOCK_SIZE - 1 / dimBlock.y);
    MatCopyKernel<<<dimGrid, dimBlock>>>(A, ax, ay,
                                         B, bx, by,
                                         N, len);
}

void MatMulSequential(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[j * N + i] += A[j * N + k] * B[k * N + i];
            }
        }
    }
}

void printMatrix(float *A, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%.2f \t", A[j * N + i]);
        }
        printf("\n");
    }
}

void printCudaMatrix(float *C, int N, int len)
{
    float *A = (float *)malloc(N * N * sizeof(float));
    cudaMemcpy(A, C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < len; j++)
        {
            printf("%.2f \t", A[j * N + i]);
        }
        printf("\n");
    }
    free(A);
}

float *initMatrix(int N, float val)
{
    float *A = (float *)malloc(N * N * sizeof(float));

    // random assign value
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = val;
        }
    }

    return A;
}

float *initMatrixRand(int N)
{
    float *A = (float *)malloc(N * N * sizeof(float));

    // random assign value
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = rand() % 100;
        }
    }

    return A;
}

void strassen(float *A, float *B, float *C, int rowSize, int N, int threshold, int d)
{
    if (rowSize <= threshold)
    {
        MatMul(A, 0, 0,
               B, 0, 0,
               C, 0, 0,
               N, rowSize);
        return;
    }

    int subSize = rowSize / 2;

    /*
        A11 = A (0,0)
        A12 = A (subSize,0)
        A21 = A (0,subSize)
        A22 = A (subSize,subSize)

        B11 = B (0,0)
        B12 = B (subSize,0)
        B21 = B (0,subSize)
        B22 = B (subSize,subSize)

        C11 = C (0,0)
        C12 = C (subSize,0)
        C21 = C (0,subSize)
        C22 = C (subSize,subSize)
    */

    // M1 = (A11+A22)(B11+B22)
    MatAdd(A, 0, 0, A, subSize, subSize, d_A[d + 1], 0, 0, N, subSize, false);
    MatAdd(B, 0, 0, B, subSize, subSize, d_B[d + 1], 0, 0, N, subSize, false);
    strassen(d_A[d + 1], d_B[d + 1], d_M1[d + 1], subSize, N, threshold, d + 1);

    // M2 = (A21+A22)B11
    MatAdd(A, 0, subSize, A, subSize, subSize, d_A[d + 1], 0, 0, N, subSize, false);
    MatCopy(B, 0, 0, d_B[d + 1], 0, 0, N, subSize);
    strassen(d_A[d + 1], d_B[d + 1], d_M2[d + 1], subSize, N, threshold, d + 1);

    // M3 = A11(B12-B22)
    MatCopy(A, 0, 0, d_A[d + 1], 0, 0, N, subSize);
    MatAdd(B, subSize, 0, B, subSize, subSize, d_B[d + 1], 0, 0, N, subSize, true);
    strassen(d_A[d + 1], d_B[d + 1], d_M3[d + 1], subSize, N, threshold, d + 1);

    // M4 = A22(B21-B11)
    MatCopy(A, subSize, subSize, d_A[d + 1], 0, 0, N, subSize);
    MatAdd(B, 0, subSize, B, 0, 0, d_B[d + 1], 0, 0, N, subSize, true);
    strassen(d_A[d + 1], d_B[d + 1], d_M4[d + 1], subSize, N, threshold, d + 1);

    // M5 = (A11+A12)B22
    MatAdd(A, 0, 0, A, subSize, 0, d_A[d + 1], 0, 0, N, subSize, false);
    MatCopy(B, subSize, subSize, d_B[d + 1], 0, 0, N, subSize);
    strassen(d_A[d + 1], d_B[d + 1], d_M5[d + 1], subSize, N, threshold, d + 1);

    // M6 = (A21-A11)(B11+B12)
    MatAdd(A, 0, subSize, A, 0, 0, d_A[d + 1], 0, 0, N, subSize, true);
    MatAdd(B, 0, 0, B, subSize, 0, d_B[d + 1], 0, 0, N, subSize, false);
    strassen(d_A[d + 1], d_B[d + 1], d_M6[d + 1], subSize, N, threshold, d + 1);

    // M7 = (A12-A22)(B21+B22)
    MatAdd(A, subSize, 0, A, subSize, subSize, d_A[d + 1], 0, 0, N, subSize, true);
    MatAdd(B, 0, subSize, B, subSize, subSize, d_B[d + 1], 0, 0, N, subSize, false);
    strassen(d_A[d + 1], d_B[d + 1], d_M7[d + 1], subSize, N, threshold, d + 1);

    // C11 = M1+M4-M5+M7
    MatAdd(d_M1[d + 1], 0, 0, d_M4[d + 1], 0, 0, C, 0, 0, N, subSize, false); // +
    MatAdd(C, 0, 0, d_M5[d + 1], 0, 0, C, 0, 0, N, subSize, true);            // -
    MatAdd(C, 0, 0, d_M7[d + 1], 0, 0, C, 0, 0, N, subSize, false);           // +

    // C12 = M3 + M5
    MatAdd(d_M3[d + 1], 0, 0, d_M5[d + 1], 0, 0, C, subSize, 0, N, subSize, false);

    // C21 = M2 + M4
    MatAdd(d_M2[d + 1], 0, 0, d_M4[d + 1], 0, 0, C, 0, subSize, N, subSize, false);

    // C22 = M1-M2+M3+M6
    MatAdd(d_M1[d + 1], 0, 0, d_M2[d + 1], 0, 0, C, subSize, subSize, N, subSize, true);    // -
    MatAdd(C, subSize, subSize, d_M3[d + 1], 0, 0, C, subSize, subSize, N, subSize, false); //+
    MatAdd(C, subSize, subSize, d_M6[d + 1], 0, 0, C, subSize, subSize, N, subSize, false); //+

    return;
}

bool verify(float *A, float *B, int N)
{
    bool equal = true;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (A[i * N + j] != B[i * N + j])
            {
                equal = false;
                break;
            }
        }
    }

    return equal;
}

void cudaInit(int N)
{
    int _N = N;
    for (int depth = 0; depth < MAX_DEPTH && _N > 0; depth++)
    {
        cudaMalloc((float **)&d_A[depth], _N * _N * sizeof(float));
        cudaMalloc((float **)&d_B[depth], _N * _N * sizeof(float));
        if (depth == 0)
        {
            cudaMalloc((float **)&d_C, _N * _N * sizeof(float));
        }
        else
        {
            cudaMalloc((float **)&d_M1[depth], _N * _N * sizeof(float));
            cudaMalloc((float **)&d_M2[depth], _N * _N * sizeof(float));
            cudaMalloc((float **)&d_M3[depth], _N * _N * sizeof(float));
            cudaMalloc((float **)&d_M4[depth], _N * _N * sizeof(float));
            cudaMalloc((float **)&d_M5[depth], _N * _N * sizeof(float));
            cudaMalloc((float **)&d_M6[depth], _N * _N * sizeof(float));
            cudaMalloc((float **)&d_M7[depth], _N * _N * sizeof(float));
        }
        // _N /= 2;
    }
}

void cudaTruncate()
{
    for (int depth = 0; depth < MAX_DEPTH; depth++)
    {
        cudaFree(d_A[depth]);
        cudaFree(d_B[depth]);

        if (depth == 0)
        {
            cudaFree(d_C);
        }
        else
        {
            cudaFree(d_M1[depth]);
            cudaFree(d_M2[depth]);
            cudaFree(d_M3[depth]);
            cudaFree(d_M4[depth]);
            cudaFree(d_M5[depth]);
            cudaFree(d_M6[depth]);
            cudaFree(d_M7[depth]);
        }
    }
}

void timer_strassen(float *A, float *B, float *C, int rowSize, int N, int threshold, int d)
{
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    strassen(A, B, C, rowSize, N, threshold, d);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Strassen Time: %8.4f ms\n", time);
}

void timer_MatMul(float *A, int ax, int ay,
                  float *B, int bx, int by,
                  float *C, int cx, int cy,
                  int N, int len)
{
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    MatMul(A, ax, ay,
           B, bx, by,
           C, cx, cy,
           N, len);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("MatMul Time: %8.4f ms\n", time);
}

void timer_MatMulSequential(float *A,
                            float *B,
                            float *C,
                            int N)
{
    struct timespec start, stop;
    float time;

    clock_gettime(CLOCK_REALTIME, &start);

    MatMulSequential(A, B, C, N);

    clock_gettime(CLOCK_REALTIME, &stop);
    time = (stop.tv_sec - start.tv_sec) + 0.000000001 * (stop.tv_nsec - start.tv_nsec);

    printf("MatMulSeq Time: %8.4f s\n", time);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: ./a.out {k} {k'}\n");
        exit(1);
    }

    int N, threshold;
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &threshold);
    // int N = 128;
    // int threshold = 8;
    printf("k = %d, k' = %d\n", N, threshold);

    float *A = initMatrixRand(N);
    float *B = initMatrixRand(N);
    float *C = initMatrix(N, 0);
    float *expected_C = initMatrix(N, 0);

    size_t size = N * N * sizeof(float);
    cudaInit(N);
    cudaMemcpy(d_A[0], A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[0], B, size, cudaMemcpyHostToDevice);

    timer_strassen(d_A[0], d_B[0], d_C, N, N, threshold, 0);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    if (VERIFY)
    {
        // MatMulSequential(A, B, expected_C, N);
        // MatMul(d_A[0], 0, 0, d_B[0], 0, 0, d_C, 0, 0, N, N);
        timer_MatMul(d_A[0], 0, 0, d_B[0], 0, 0, d_C, 0, 0, N, N);

        cudaMemcpy(expected_C, d_C, size, cudaMemcpyDeviceToHost);

        if (verify(expected_C, C, N))
        {
            printf("Success. Matries are equals\n");
        }
        else
        {
            printf("!!!Matries are not equal\n");
        }
    }

    cudaTruncate();
    free(A);
    free(B);
    free(C);
    free(expected_C);
}
