#!/usr/bin/env python3
"""embed_msl.py NAME file.metal  ->  C header defining `static const char NAME[]` with the file's text."""
import sys

name, path = sys.argv[1], sys.argv[2]
print(f"/* generated from {path} by tools/embed_msl.py — do not edit */")
print(f"static const char {name}[] =")
for line in open(path, encoding="utf-8").read().splitlines():
    esc = line.replace("\\", "\\\\").replace('"', '\\"')
    print(f'    "{esc}\\n"')
print(";")
