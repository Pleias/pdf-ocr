#!/usr/bin/env bash
# Build an offline wheelhouse for a Linux x86_64 / CPython 3.12 GPU machine,
# on any computer with internet (macOS included). Copy the output folder to the
# offline machine and follow "Offline machines" in docs/INSTALL.md.
#
#   tools/make_wheelhouse.sh [out_dir]        default: ./wheelhouse   (~4.7 GB)
#
# Plain `pip download -r requirements.txt` on a Mac is NOT enough: pip evaluates
# the environment markers against the machine it runs on, so every
# `; sys_platform == 'linux'` line (the CUDA libraries, triton) is silently
# skipped. This resolves the markers for the target first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "${1:-wheelhouse}" && cd "${1:-wheelhouse}" && pwd)"
command -v uv >/dev/null || { echo "needs uv: https://docs.astral.sh/uv/"; exit 2; }

# Two lists: what to download (direct URLs kept) and what to install. pip never
# matches a `name @ url` requirement against --find-links, so offline the install
# list must pin by version instead.
uvx --quiet --python 3.12 --with packaging python - "$HERE/requirements.txt" "$OUT" <<'PY'
import sys, os
from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
env = {"implementation_name": "cpython", "platform_python_implementation": "CPython",
       "sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64",
       "python_version": "3.12", "python_full_version": "3.12.0", "os_name": "posix",
       "platform_release": "", "platform_version": "", "implementation_version": "3.12.0",
       "extra": ""}
dl, inst = [], []
for line in open(sys.argv[1]):
    s = line.split("#")[0].strip()
    if s and not s.startswith("-"):
        r = Requirement(s)
        if r.marker is None or r.marker.evaluate(env):
            if r.url:
                _, ver, _, _ = parse_wheel_filename(os.path.basename(r.url))
                dl.append(f"{r.name} @ {r.url}")
                inst.append(f"{r.name}=={ver.public}")
            else:
                dl.append(f"{r.name}{r.specifier}"); inst.append(dl[-1])
open(os.path.join(sys.argv[2], "requirements-download.txt"), "w").write("\n".join(dl) + "\n")
open(os.path.join(sys.argv[2], "requirements-linux.txt"), "w").write("\n".join(inst) + "\n")
PY

plat=()
for t in manylinux_2_34 manylinux_2_31 manylinux_2_28 manylinux_2_27 manylinux_2_24 \
         manylinux_2_17 manylinux2014 manylinux2010 manylinux1 linux; do
    plat+=(--platform "${t}_x86_64")
done
uvx --quiet --python 3.12 pip download --no-deps --only-binary=:all: "${plat[@]}" \
    --python-version 3.12 --implementation cp --abi cp312 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$OUT/requirements-download.txt" -d "$OUT"
# A wheel fetched from a URL can carry a local version in its filename that its
# own metadata lacks (flash-attn's release wheels do: 2.8.3+cu12torch2.7… vs
# 2.8.3). pip then discards it from --find-links as "inconsistent version".
# Store it under the name its metadata states.
uvx --quiet --python 3.12 --with packaging python - "$OUT" <<'PY'
import os, sys, urllib.parse
from packaging.utils import parse_wheel_filename
out = sys.argv[1]
for line in open(os.path.join(out, "requirements-download.txt")):
    if " @ " not in line:
        continue
    fn = urllib.parse.unquote(os.path.basename(line.split(" @ ", 1)[1].strip()))
    src = os.path.join(out, fn)
    _, ver, _, _ = parse_wheel_filename(fn)
    if os.path.exists(src) and ver.local is not None:
        dst = os.path.join(out, fn.replace(fn.split("-")[1], ver.public, 1))
        os.replace(src, dst)
        print(f"renamed {fn} -> {os.path.basename(dst)}")
PY
# the build backend, so `pip install .` works offline too
uvx --quiet --python 3.12 pip download --only-binary=:all: hatchling -d "$OUT"
echo "wheelhouse: $OUT ($(ls "$OUT"/*.whl | wc -l | tr -d ' ') wheels, $(du -sh "$OUT" | cut -f1))"
