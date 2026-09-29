# Prebuilt HIP kernels for Hygon DCU (gfx9)

| File | Targets | Notes |
|---|---|---|
| `_C.abi3.so.gfx906_928_938` | `gfx906;gfx928;gfx938` | Fat binary — use this unless you have a reason not to. |
| `_C.abi3.so.gfx928` | `gfx928` | gfx928-only build, smaller. |

## Install

Copy the desired binary over the module's compiled extension:

```sh
# locate the installed package
PKG=$(python -c "import comfy_kitchen, os; print(os.path.dirname(comfy_kitchen.__file__))")

cp prebuilt/_C.abi3.so.gfx906_928_938 "$PKG/backends/hip/_C.abi3.so"
```

Restart ComfyUI afterwards so the new extension is loaded.

Built with the Hygon DTK HIP toolchain (`/opt/dtk/llvm/bin/clang++`),
`CMAKE_BUILD_TYPE=Release`. See the top-level [`README-DCU.md`](../README-DCU.md)
for build-from-source instructions.
