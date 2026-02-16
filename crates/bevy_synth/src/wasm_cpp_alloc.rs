//! wasm32-only C++ operator new/delete shims.
//!
//! Some transitive wasm object code references `_Znwm`/`_ZdlPv` (C++
//! `operator new/delete`). Defining them in-crate avoids unresolved `"env"`
//! imports in browser builds.

#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use core::alloc::Layout;
use core::mem::{align_of, size_of};
use std::alloc::{alloc, dealloc, handle_alloc_error};

const HEADER_ALIGN: usize = align_of::<usize>();
const HEADER_SIZE: usize = size_of::<usize>();

fn header_layout(size: usize) -> Layout {
    let total = size.saturating_add(HEADER_SIZE).max(HEADER_SIZE);
    // This layout matches both allocation and deallocation paths.
    Layout::from_size_align(total, HEADER_ALIGN).expect("valid C++ shim layout")
}

/// C++ `operator new(unsigned long)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Znwm(size: usize) -> *mut u8 {
    let layout = header_layout(size);
    let base = unsafe { alloc(layout) };
    if base.is_null() {
        handle_alloc_error(layout);
    }
    unsafe { (base as *mut usize).write(size) };
    unsafe { base.add(HEADER_SIZE) }
}

/// C++ `operator delete(void*)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _ZdlPv(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let base = unsafe { ptr.sub(HEADER_SIZE) };
    let size = unsafe { (base as *const usize).read() };
    let layout = header_layout(size);
    unsafe { dealloc(base, layout) };
}
