#!/usr/bin/env python3
"""mkfatbin — package PTX into an NVIDIA fat binary container (the format nvcc's
`fatbinary` tool emits and that __cudaRegisterFatBinary / cuModuleLoadFatBinary
consume).

    mkfatbin.py [--compress] [--arch 80] input.ptx [more.ptx ...] -o out.fatbin

Layout (little endian):
    container header : u32 magic 0xBA55ED50, u16 version 1, u16 header_size 16, u64 size
    entry header     : u16 kind (1 = PTX, 2 = ELF), u16 version 0x0101, u32 header_size 64,
                       u64 payload_size, u32 compressed_size, u32 reserved,
                       u16 ptx_minor, u16 ptx_major, u32 arch, u32 name_off, u32 name_len,
                       u64 flags (0x1 64-bit, 0x2000 LZ4 compressed), u64 0, u64 uncompressed_size
    payload          : PTX text + NUL (optionally LZ4 block compressed), padded to 8 bytes

--compress uses the LZ4 *block* format, as nvcc does by default for fatbin entries.
"""
import argparse
import re
import struct
import sys

FATBIN_MAGIC = 0xBA55ED50
KIND_PTX = 1
FLAG_64BIT = 0x1
FLAG_LZ4 = 0x2000


def lz4_compress(data: bytes) -> bytes:
    """Greedy LZ4 block compressor (spec compliant: last 5 bytes are literals,
    no match starts within the last 12 bytes)."""
    n = len(data)
    out = bytearray()
    table = {}
    i = anchor = 0
    limit = n - 12

    def put_len(v):
        while v >= 255:
            out.append(255)
            v -= 255
        out.append(v)

    while i < limit:
        key = data[i:i + 4]
        cand = table.get(key)
        table[key] = i
        if cand is not None and i - cand <= 0xFFFF:
            m = 4
            while i + m < n - 5 and data[cand + m] == data[i + m]:
                m += 1
            lit = i - anchor
            ml = m - 4
            out.append((min(lit, 15) << 4) | min(ml, 15))
            if lit >= 15:
                put_len(lit - 15)
            out += data[anchor:i]
            out += struct.pack("<H", i - cand)
            if ml >= 15:
                put_len(ml - 15)
            i += m
            anchor = i
        else:
            i += 1
    lit = n - anchor
    out.append(min(lit, 15) << 4)
    if lit >= 15:
        put_len(lit - 15)
    out += data[anchor:]
    return bytes(out)


def lz4_decompress(src: bytes, size: int) -> bytes:
    dst = bytearray()
    ip = 0
    while ip < len(src):
        tok = src[ip]; ip += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                b = src[ip]; ip += 1; lit += b
                if b != 255:
                    break
        dst += src[ip:ip + lit]; ip += lit
        if ip >= len(src):
            break
        off = src[ip] | (src[ip + 1] << 8); ip += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = src[ip]; ip += 1; ml += b
                if b != 255:
                    break
        ml += 4
        for _ in range(ml):
            dst.append(dst[-off])
    assert len(dst) == size
    return bytes(dst)


def entry(ptx: bytes, arch: int, compress: bool) -> bytes:
    m = re.search(rb"\.version\s+(\d+)\.(\d+)", ptx)
    major, minor = (int(m.group(1)), int(m.group(2))) if m else (7, 0)
    raw = ptx + b"\0"
    flags = FLAG_64BIT
    payload = raw
    if compress:
        payload = lz4_compress(raw)
        assert lz4_decompress(payload, len(raw)) == raw, "LZ4 round trip failed"
        flags |= FLAG_LZ4
    padded = payload + b"\0" * ((-len(payload)) % 8)
    hdr = struct.pack("<HHIQIIHHIIIQQQ", KIND_PTX, 0x0101, 64, len(padded),
                      len(payload) if compress else 0, 0, minor, major, arch, 0, 0,
                      flags, 0, len(raw) if compress else 0)
    assert len(hdr) == 64
    return hdr + padded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--arch", type=int, default=80)
    ap.add_argument("--compress", action="store_true")
    a = ap.parse_args()
    body = b"".join(entry(open(p, "rb").read(), a.arch, a.compress) for p in a.inputs)
    blob = struct.pack("<IHHQ", FATBIN_MAGIC, 1, 16, len(body)) + body
    with open(a.output, "wb") as f:
        f.write(blob)
    raw = sum(len(open(p, "rb").read()) for p in a.inputs)
    print(f"mkfatbin: {len(a.inputs)} PTX entr{'y' if len(a.inputs) == 1 else 'ies'}, "
          f"{raw} B PTX -> {len(blob)} B fatbin{' (LZ4)' if a.compress else ''}", file=sys.stderr)


if __name__ == "__main__":
    main()
