#ifndef METACUDA_FATBIN_H
#define METACUDA_FATBIN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Extracts PTX from a fatbin wrapper / fatbin container / raw PTX image.
 * Returns malloc'd NUL-terminated text, or NULL with `err` filled in.
 * image_size may be 0 when unknown. */
char *mc_extract_ptx(const void *image, size_t image_size, char *err, size_t errlen, int *out_arch);
long mc_lz4_decompress(const uint8_t *src, size_t srclen, uint8_t *dst, size_t dstcap);
#ifdef __cplusplus
}
#endif
#endif
