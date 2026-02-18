#![allow(clippy::missing_safety_doc)]

use core::alloc::Layout;
use core::mem::size_of;
use std::alloc::{alloc, dealloc, handle_alloc_error};

const ALIGN: usize = 16;
const HEADER_BYTES: usize = size_of::<usize>();

/// Provides C++ global new/delete symbols required by meshopt's wasm build.
#[unsafe(export_name = "_Znwm")]
pub unsafe extern "C" fn meshopt_operator_new(size: usize) -> *mut u8 {
    let total = match size.checked_add(HEADER_BYTES) {
        Some(v) => v,
        None => return core::ptr::null_mut(),
    };
    let layout = match Layout::from_size_align(total.max(1), ALIGN) {
        Ok(v) => v,
        Err(_) => return core::ptr::null_mut(),
    };
    let raw = unsafe { alloc(layout) };
    if raw.is_null() {
        handle_alloc_error(layout);
    }
    unsafe { (raw as *mut usize).write(total) };
    unsafe { raw.add(HEADER_BYTES) }
}

#[unsafe(export_name = "_ZdlPv")]
pub unsafe extern "C" fn meshopt_operator_delete(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let raw = unsafe { ptr.sub(HEADER_BYTES) };
    let total = unsafe { (raw as *const usize).read() };
    if let Ok(layout) = Layout::from_size_align(total.max(1), ALIGN) {
        unsafe { dealloc(raw, layout) };
    }
}
