/* cupti-trace.c: LD_PRELOAD shim that records every kernel launch's name, start and end
 * timestamps via CUPTI's activity API, and writes them to CUPTI_TRACE_OUT on process exit.
 * Compiled in the TOP job with the staged gcc against /usr/local/cuda's CUPTI headers. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <cupti.h>

#define MAX_KERNELS (1 << 20)
static struct {
    const char *name;
    unsigned long long start, end;
    unsigned int gridX, gridY, gridZ;
    unsigned int streamId, correlationId;
} kernels[MAX_KERNELS];
static int kernel_count;
static FILE *out;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void CUPTIAPI buffer_request(unsigned char **buffer, size_t *size, size_t *max_records)
{
    *size = 1 << 24;
    *buffer = malloc(*size);
    *max_records = 0;
}

static void CUPTIAPI buffer_complete(CUcontext ctx, unsigned int streamId, unsigned char *buffer,
                                     size_t size, size_t valid_size)
{
    CUpti_Activity *record;
    CUptiResult status;
    pthread_mutex_lock(&lock);
    status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
    while (status == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && kernel_count < MAX_KERNELS) {
            CUpti_ActivityKernel8 *k = (CUpti_ActivityKernel8 *)record;
            kernels[kernel_count].name = strdup(k->name ? k->name : "");
            kernels[kernel_count].start = k->start;
            kernels[kernel_count].end = k->end;
            kernels[kernel_count].gridX = k->gridX;
            kernels[kernel_count].gridY = k->gridY;
            kernels[kernel_count].gridZ = k->gridZ;
            kernels[kernel_count].streamId = k->streamId;
            kernels[kernel_count].correlationId = k->correlationId;
            kernel_count++;
        }
        status = cuptiActivityGetNextRecord(buffer, valid_size, &record);
    }
    pthread_mutex_unlock(&lock);
    free(buffer);
}

static void flush(void)
{
    if (!out)
        return;
    fprintf(out, "name,gridX,gridY,gridZ,streamId,correlationId,start_ns,end_ns\n");
    for (int i = 0; i < kernel_count; i++)
        fprintf(out, "\"%s\",%u,%u,%u,%u,%u,%llu,%llu\n",
                kernels[i].name, kernels[i].gridX, kernels[i].gridY, kernels[i].gridZ,
                kernels[i].streamId, kernels[i].correlationId,
                kernels[i].start, kernels[i].end);
    fflush(out);
}

__attribute__((destructor)) static void teardown(void)
{
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityFlushAll(0);
    flush();
    if (out)
        fclose(out);
}

__attribute__((constructor)) static void setup(void)
{
    const char *path = getenv("CUPTI_TRACE_OUT");
    if (!path)
        path = "/tmp/cupti-trace.csv";
    out = fopen(path, "w");
    if (!out)
        return;
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityRegisterCallbacks(buffer_request, buffer_complete);
}
