// ensure_no_std/src/main.rs
#![no_std]
#![no_main]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::panic::PanicInfo;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{ArrayStorage, Const, Matrix2, OMatrix, U2, Vector2};

struct DummyAllocator;

unsafe impl GlobalAlloc for DummyAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOCATOR: DummyAllocator = DummyAllocator;

/// This function is called on panic.
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[derive(Clone)]
struct LinearProblem {
    params: Vector2<f64>,
}

impl LeastSquaresProblem<f64, Const<2>, U2> for LinearProblem {
    type ResidualStorage = ArrayStorage<f64, 2, 1>;
    type JacobianStorage = ArrayStorage<f64, 2, 2>;
    type ParameterStorage = ArrayStorage<f64, 2, 1>;

    fn set_params(&mut self, p: &Vector2<f64>) {
        self.params = *p;
    }

    fn params(&self) -> Vector2<f64> {
        self.params
    }

    fn residuals(&self) -> Option<Vector2<f64>> {
        Some(self.params)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Const<2>, U2>> {
        Some(Matrix2::identity())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let problem = LinearProblem {
        params: Vector2::new(1.0, 2.0),
    };
    let (_res, _report) = LevenbergMarquardt::new().minimize(problem);
    loop {}
}
