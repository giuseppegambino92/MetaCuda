// mc-ptx2msl — translate a PTX file to MSL and (optionally) compile it with Metal.
//   usage: mc-ptx2msl [-c] input.ptx [output.metal]
//     -c   also compile with MTLDevice newLibraryWithSource and report pipeline stats
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/ptx_to_metal.h"

int main(int argc, char **argv) {
    @autoreleasepool {
        int compile = 0, ai = 1;
        if (ai < argc && !strcmp(argv[ai], "-c")) { compile = 1; ai++; }
        if (ai >= argc) { fprintf(stderr, "usage: %s [-c] input.ptx [output.metal]\n", argv[0]); return 2; }
        FILE *f = fopen(argv[ai], "rb");
        if (!f) { perror(argv[ai]); return 1; }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc((size_t)n + 1);
        fread(buf, 1, (size_t)n, f);
        buf[n] = 0;
        fclose(f);
        mc_module_t m;
        if (mc_translate_ptx(buf, (size_t)n, &m)) { fprintf(stderr, "translation failed: %s\n", m.error); return 1; }
        if (m.warnings) fprintf(stderr, "warnings:\n%s", m.warnings);
        if (ai + 1 < argc) {
            FILE *o = fopen(argv[ai + 1], "wb");
            fwrite(m.msl, 1, m.msl_len, o);
            fclose(o);
        } else if (!compile) {
            fwrite(m.msl, 1, m.msl_len, stdout);
        }
        for (int i = 0; i < m.nkernels; i++) {
            mc_kernel_t *k = &m.kernels[i];
            fprintf(stderr, "kernel %-40s params=%u smem=%s(static %u, dyn@%u) local=%u convergent=%d\n", k->name, k->param_bytes,
                    k->uses_smem ? "yes" : "no", k->static_smem, k->smem_dyn_offset, k->local_bytes, k->convergent);
        }
        if (compile) {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            MTLCompileOptions *opt = [MTLCompileOptions new];
            opt.languageVersion = MTLLanguageVersion3_2;
            opt.mathMode = MTLMathModeSafe;
            NSError *err = nil;
            NSDate *t0 = [NSDate date];
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:m.msl] options:opt error:&err];
            if (!lib) { fprintf(stderr, "Metal compile failed:\n%s\n", err.localizedDescription.UTF8String); return 1; }
            for (int i = 0; i < m.nkernels; i++) {
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:m.kernels[i].msl_name]];
                id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
                if (!ps) { fprintf(stderr, "pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }
                fprintf(stderr, "  %-40s maxThreads=%lu simdWidth=%lu\n", m.kernels[i].msl_name, (unsigned long)ps.maxTotalThreadsPerThreadgroup,
                        (unsigned long)ps.threadExecutionWidth);
            }
            fprintf(stderr, "Metal compile OK in %.1f ms\n", -[t0 timeIntervalSinceNow] * 1000.0);
        }
        mc_module_free(&m);
        free(buf);
    }
    return 0;
}
