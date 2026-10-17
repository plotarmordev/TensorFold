// Times tf_nemo_conv + tf_nemo_scan from the base (a), the previous tip 480ad61f (b) and the tip fa2af585 (t) at Super's Mamba shape, by window rows and kept rows (RMAX 32).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "glue_math.cuh"
namespace va {
#include "a.cu"
}
namespace vb {
#include "b.cu"
}
namespace vt {
#include "t.cu"
}
constexpr int H = 128, DH = 64, NG = 8, DS = 128, XD = H * DH, CD = XD + 2 * NG * DS, PROJ = XD + CD + H, DTOFF = XD + CD;
typedef void (*ScanFn)(const __nv_bfloat16*, const __nv_bfloat16*, float*, float*, const float*, const float*, const float*, const int*,
                       __nv_bfloat16*, int, float, float, int, int, int, int, int, int, int, int);
template <class T> T* dev(size_t n, float scale) {
    T* p;
    cudaMalloc(&p, n * sizeof(T));
    T* h = (T*)malloc(n * sizeof(T));
    for (size_t i = 0; i < n; ++i) {
        float v = scale * ((rand() % 2001) / 1000.0f - 1.0f);
        if constexpr (sizeof(T) == 2) h[i] = __float2bfloat16(v); else h[i] = v;
    }
    cudaMemcpy(p, h, n * sizeof(T), cudaMemcpyHostToDevice);
    free(h);
    return p;
}
int main() {
    const int rmax = 32, layers = 40, nv = 3;
    const char* names = "abt";
    ScanFn fns[nv] = {va::a_scan, vb::b_scan, vt::t_scan};
    __nv_bfloat16 *P = dev<__nv_bfloat16>((size_t)64 * PROJ, 1.0f);
    __nv_bfloat16 *BASE = dev<__nv_bfloat16>((size_t)layers * 3 * CD, 1.0f), *RAW = dev<__nv_bfloat16>((size_t)layers * 2 * rmax * CD, 1.0f);
    __nv_bfloat16 *XC = dev<__nv_bfloat16>((size_t)layers * 2 * rmax * CD, 1.0f), *Y = dev<__nv_bfloat16>((size_t)64 * XD, 0.0f);
    float *CW = dev<float>(4 * CD, 0.5f), *CB = dev<float>(CD, 0.1f), *DT = dev<float>((size_t)layers * 2 * rmax * H, 0.01f);
    float *S = dev<float>((size_t)layers * H * DH * DS, 0.01f), *A = dev<float>(H, 0.0f), *DSK = dev<float>(H, 1.0f), *DTB = dev<float>(H, 0.5f);
    int* META;
    cudaMalloc(&META, 16);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    printf("rows pk  us per layer (conv + scan) for variants %s, min of 21 reps of 40 layers x 25\n", names);
    for (int pk : {1, 8, 15})
        for (int r : {1, 3, 8, 12, 16, 24, 32}) {
            int meta[4] = {100, 0, pk, 0};
            cudaMemcpy(META, meta, 16, cudaMemcpyHostToDevice);
            float best[nv];
            for (float& b : best) b = 1e9f;
            for (int rep = 0; rep < 21; ++rep)
                for (int v = pk + r > 32 ? 1 : 0; v < nv; ++v) {
                    cudaEventRecord(e0);
                    for (int it = 0; it < 25; ++it)
                        for (int l = 0; l < layers; ++l) {
                            __nv_bfloat16 *raw = RAW + (size_t)l * 2 * rmax * CD, *xc = XC + (size_t)l * 2 * rmax * CD;
                            float *dt = DT + (size_t)l * 2 * rmax * H, *s = S + (size_t)l * H * DH * DS;
                            va::a_conv<<<(CD + 255) / 256, 256>>>(P, BASE + l * 3 * CD, raw, xc, CW, CB, META, r, PROJ, XD, CD, rmax);
                            fns[v]<<<dim3(H, DH / 32), 128>>>(P, xc, dt, s, A, DSK, DTB, META, Y, r, 0.001f, 0.1f, PROJ, XD, CD, DTOFF, H, DH, NG, rmax);
                        }
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best[v] = fminf(best[v], ms * 1000.0f / (25 * layers));
                }
            printf("%2d %2d", r, pk);
            for (int v = 0; v < nv; ++v) printf("  %c %.3f", names[v], (pk + r > 32 && v == 0) ? 0.0f : best[v]);
            printf("\n");
        }
    cudaError_t err = cudaDeviceSynchronize();
    printf("status %s\n", cudaGetErrorString(err));
    return err != cudaSuccess;
}
