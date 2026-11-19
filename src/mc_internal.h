/* MetaCuda — internal helpers shared between the driver and runtime layers. */
#ifndef METACUDA_INTERNAL_H
#define METACUDA_INTERNAL_H
#include <stddef.h>
#include <stdint.h>
#include "../include/cuda.h"
#ifdef __cplusplus
extern "C" {
#endif
int mc_log_level(void);
void mc_log(int level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
CUresult mc_sync_all(void);
const char *mc_device_name(void);
int mc_gpu_cores(void);
size_t mc_total_mem(void);
int mc_shared_mem_per_block(void);
int mc_pointer_info(const void *p, int *is_device, int *is_managed, void **host, uint64_t *dev);
CUresult mc_function_info(CUfunction f, int *max_threads, int *static_smem, int *local_bytes);
#ifdef __cplusplus
}
#endif
#endif
