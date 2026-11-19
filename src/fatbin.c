/*
 * MetaCuda — NVIDIA fat binary parsing.
 *
 * Accepts any of the images CUDA's loaders accept and returns PTX text:
 *   - a fat binary wrapper  { magic 0x466243b1, version, const void *data, ... }
 *     (what nvcc/clang pass to __cudaRegisterFatBinary)
 *   - a fat binary container (magic 0xBA55ED50) holding PTX / ELF entries,
 *     optionally LZ4-compressed
 *   - raw PTX text
 * ELF cubins (SASS) are recognised and rejected with a clear message: SASS is
 * NVIDIA-GPU machine code and cannot be retargeted; MetaCuda needs PTX.
 */
#include "fatbin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FATBIN_WRAPPER_MAGIC 0x466243b1u
#define FATBIN_MAGIC 0xBA55ED50u
#define FATBIN_KIND_PTX 1
#define FATBIN_KIND_ELF 2
#define FATBIN_FLAG_COMPRESSED_LZ4 0x0000000000002000ull
#define FATBIN_FLAG_COMPRESSED_ZSTD 0x0000000000008000ull

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t size; /* bytes of entries following this header */
} fatbin_header_t;

typedef struct __attribute__((packed)) {
    uint16_t kind;
    uint16_t unknown1;
    uint32_t header_size;
    uint64_t size; /* payload bytes following the entry header */
    uint32_t compressed_size;
    uint32_t unknown2;
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;
    uint64_t decompressed_size;
} fatbin_entry_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    const void *data;
    const void *filename_or_fatbins;
} fatbin_wrapper_t;

/* LZ4 block format decoder. Returns decoded size or -1. */
long mc_lz4_decompress(const uint8_t *src, size_t srclen, uint8_t *dst, size_t dstcap) {
    size_t ip = 0, op = 0;
    while (ip < srclen) {
        uint8_t token = src[ip++];
        size_t lit = token >> 4;
        if (lit == 15) {
            uint8_t b;
            do {
                if (ip >= srclen) return -1;
                b = src[ip++];
                lit += b;
            } while (b == 255);
        }
        if (ip + lit > srclen || op + lit > dstcap) return -1;
        memcpy(dst + op, src + ip, lit);
        ip += lit;
        op += lit;
        if (ip >= srclen) break; /* last sequence has literals only */
        if (ip + 2 > srclen) return -1;
        size_t off = (size_t)src[ip] | ((size_t)src[ip + 1] << 8);
        ip += 2;
        if (off == 0 || off > op) return -1;
        size_t ml = token & 15;
        if (ml == 15) {
            uint8_t b;
            do {
                if (ip >= srclen) return -1;
                b = src[ip++];
                ml += b;
            } while (b == 255);
        }
        ml += 4;
        if (op + ml > dstcap) return -1;
        for (size_t i = 0; i < ml; i++, op++) dst[op] = dst[op - off];
    }
    return (long)op;
}

static int looks_like_ptx(const char *s) {
    /* skip whitespace / comments */
    for (int i = 0; i < 4096 && s[i]; i++) {
        if (!strncmp(s + i, ".version", 8) || !strncmp(s + i, ".target", 7) || !strncmp(s + i, ".visible", 8) ||
            !strncmp(s + i, ".entry", 6))
            return 1;
        unsigned char c = (unsigned char)s[i];
        if (c < 9 || (c > 13 && c < 32 && c != 27) || c > 126) return 0;
    }
    return 0;
}

static char *dup_text(const char *p, size_t n) {
    char *r = (char *)malloc(n + 1);
    memcpy(r, p, n);
    r[n] = 0;
    return r;
}

char *mc_extract_ptx(const void *image, size_t image_size, char *err, size_t errlen, int *out_arch) {
    if (out_arch) *out_arch = 0;
    if (!image) { snprintf(err, errlen, "null image"); return NULL; }
    const uint8_t *p = (const uint8_t *)image;
    uint32_t magic;
    memcpy(&magic, p, 4);

    if (magic == FATBIN_WRAPPER_MAGIC) {
        const fatbin_wrapper_t *w = (const fatbin_wrapper_t *)image;
        return mc_extract_ptx(w->data, 0, err, errlen, out_arch);
    }
    if (!memcmp(p, "\x7f" "ELF", 4)) {
        snprintf(err, errlen, "image is an ELF cubin (NVIDIA SASS machine code); MetaCuda needs PTX — "
                              "rebuild with -gencode arch=compute_XX,code=compute_XX or embed PTX");
        return NULL;
    }
    if (magic != FATBIN_MAGIC) {
        if (looks_like_ptx((const char *)p)) {
            size_t n = image_size ? image_size : strlen((const char *)p);
            /* trim at first NUL if a size was given */
            size_t l = strnlen((const char *)p, n);
            return dup_text((const char *)p, l);
        }
        snprintf(err, errlen, "unrecognised module image (magic 0x%08x)", magic);
        return NULL;
    }

    /* fat binary container(s) — possibly several back to back */
    const fatbin_entry_t *best = NULL;
    const uint8_t *best_payload = NULL;
    int nelf = 0, best_elf_arch = 0;
    const uint8_t *cur = p;
    for (int guard = 0; guard < 64; guard++) {
        fatbin_header_t h;
        memcpy(&h, cur, sizeof h);
        if (h.magic != FATBIN_MAGIC) break;
        const uint8_t *e = cur + h.header_size;
        const uint8_t *end = e + h.size;
        while (e + sizeof(fatbin_entry_t) <= end) {
            const fatbin_entry_t *en = (const fatbin_entry_t *)e;
            if (en->header_size == 0) break;
            if (en->kind == FATBIN_KIND_PTX) {
                /* prefer the highest virtual arch not newer than sm_90 */
                int a = (int)en->arch;
                int ba = best ? (int)best->arch : -1;
                int better = !best || (a <= 90 && (ba > 90 || a > ba)) || (a > 90 && ba > 90 && a < ba);
                if (better) { best = en; best_payload = e + en->header_size; }
            } else if (en->kind == FATBIN_KIND_ELF) {
                nelf++;
                if ((int)en->arch > best_elf_arch) best_elf_arch = (int)en->arch;
            }
            e += en->header_size + en->size;
        }
        cur = end;
        uint32_t next;
        memcpy(&next, cur, 4);
        if (next != FATBIN_MAGIC) break;
    }
    if (!best) {
        if (nelf)
            snprintf(err, errlen, "fat binary contains only SASS (%d ELF cubin(s), up to sm_%d) and no PTX; "
                                  "MetaCuda translates PTX — rebuild with PTX embedded (e.g. -gencode arch=compute_80,code=compute_80)",
                     nelf, best_elf_arch);
        else snprintf(err, errlen, "fat binary contains no PTX");
        return NULL;
    }
    if (out_arch) *out_arch = (int)best->arch;
    if (best->flags & FATBIN_FLAG_COMPRESSED_ZSTD) {
        snprintf(err, errlen, "fat binary PTX is ZSTD-compressed (CUDA 12.8+ -compress-mode); not supported yet — rebuild with -Xfatbin -compress=false");
        return NULL;
    }
    if (best->flags & FATBIN_FLAG_COMPRESSED_LZ4) {
        size_t out = (size_t)best->decompressed_size;
        uint8_t *buf = (uint8_t *)malloc(out + 1);
        long n = mc_lz4_decompress(best_payload, best->compressed_size ? best->compressed_size : best->size, buf, out);
        if (n < 0) { free(buf); snprintf(err, errlen, "LZ4 decompression of embedded PTX failed"); return NULL; }
        buf[n] = 0;
        return (char *)buf;
    }
    size_t l = strnlen((const char *)best_payload, (size_t)best->size);
    return dup_text((const char *)best_payload, l);
}
