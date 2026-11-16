# tf-cuda-test builds again with kernels

- The Drama 3 compact router contract's row stride and router offset are `usize`, so a held id (`u8`) added to them no longer narrows the offset (2312) to `u8`; its picks are copied into an `i32` buffer rather than duplicated from an `align(1)` view.
