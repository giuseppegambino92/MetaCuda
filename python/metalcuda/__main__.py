"""python -m metalcuda [-v] script.py [args...]   |   python -m metalcuda -c "code"   |   python -m metalcuda --info

Runs a Python program (or `--exec cmd args...` for any executable) with
libmetalcuda.dylib injected via DYLD_INSERT_LIBRARIES, so CUDA libraries it
loads — through ctypes, cffi or compiled extensions — resolve to MetaCuda.
"""
import os
import sys

import metalcuda


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    env = dict(os.environ)
    if args[0] == "-v":
        env["METALCUDA_LOG"] = env.get("METALCUDA_LOG", "1")
        args = args[1:]
    if args[0] == "--info":
        print("libmetalcuda:", metalcuda.library_path())
        print("library dir :", metalcuda.library_dir())
        print("injected    :", metalcuda.injected())
        return 0
    if args[0] == "--exec":
        metalcuda.run(args[1:], env)
    # keep the package importable in the child
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(metalcuda.__file__)))
    env["PYTHONPATH"] = os.pathsep.join([pkg_root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    metalcuda.run([sys.executable] + args, env)


if __name__ == "__main__":
    sys.exit(main())
