/*
 * MetaCuda — dyld interposer.
 *
 * When libmetalcuda.dylib is injected with DYLD_INSERT_LIBRARIES, dyld applies
 * the __DATA,__interpose tuples below to every image in the process: any
 * attempt to dlopen() an NVIDIA driver/runtime library (libcuda.so.1,
 * libcudart.so.12, libcuda.dylib, CUDA.framework, nvcuda.dll, ...) is
 * redirected to this library, which exports the CUDA Driver and Runtime APIs.
 * This is what lets Python (ctypes / cffi) and plugin loaders that hard-code
 * Linux library names run unmodified.
 *
 * Images that link CUDA directly (two-level namespace) are served by the
 * libcuda.dylib / libcudart.dylib names installed next to this library.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mc_internal.h"

#define DYLD_INTERPOSE(_replacement, _replacee)                                                                   \
    __attribute__((used)) static struct {                                                                         \
        const void *replacement;                                                                                  \
        const void *replacee;                                                                                     \
    } _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = {(const void *)(unsigned long)&_replacement, \
                                                                              (const void *)(unsigned long)&_replacee};

static const char *self_path(void) {
    static char path[4096];
    if (!path[0]) {
        Dl_info i;
        if (dladdr((const void *)&self_path, &i) && i.dli_fname) snprintf(path, sizeof path, "%s", i.dli_fname);
    }
    return path;
}

/* Returns 1 if `path` names an NVIDIA CUDA driver / runtime library. */
int mc_is_cuda_library(const char *path) {
    if (!path) return 0;
    const char *b = strrchr(path, '/');
    b = b ? b + 1 : path;
    if (!strcmp(b, "CUDA") && strstr(path, "CUDA.framework")) return 1;
    if (!strncmp(b, "nvcuda", 6)) return 1;
    static const char *prefixes[] = {"libcuda.so", "libcuda.dylib", "libcuda.1", "libcudart.so", "libcudart.dylib", "libcudart.1",
                                     "cudart64_", "cudart32_", NULL};
    for (int i = 0; prefixes[i]; i++)
        if (!strncmp(b, prefixes[i], strlen(prefixes[i]))) return 1;
    return 0;
}

static void *mc_dlopen(const char *path, int mode) {
    if (path && mc_is_cuda_library(path) && getenv("METALCUDA_DISABLE_REDIRECT") == NULL) {
        const char *self = self_path();
        if (self[0]) {
            mc_log(1, "dlopen(\"%s\") -> %s", path, self);
            return dlopen(self, mode);
        }
    }
    return dlopen(path, mode);
}
DYLD_INTERPOSE(mc_dlopen, dlopen)

__attribute__((constructor)) static void mc_constructor(void) {
    if (mc_log_level() >= 1) {
        const char *ins = getenv("DYLD_INSERT_LIBRARIES");
        mc_log(1, "libmetalcuda loaded (%s)%s", self_path(), ins && strstr(ins, "metalcuda") ? " via DYLD_INSERT_LIBRARIES; dlopen interposition active" : "");
    }
}
