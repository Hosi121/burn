use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use burn_backend::{DType, TensorData, ops::ConvTransposeOptions};
use burn_flex::{FlexTensor, ops::conv_transpose::conv_transpose3d_f32};

struct Alloc;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(n: usize) {
    let size = CURRENT.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(size, Relaxed);
}
unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            allocated(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Relaxed);
        unsafe {
            System.dealloc(p, layout);
        }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        let p = unsafe { System.realloc(p, layout, new) };
        if !p.is_null() {
            CURRENT.fetch_sub(layout.size(), Relaxed);
            allocated(new);
        }
        p
    }
}
#[global_allocator]
static ALLOC: Alloc = Alloc;

struct Case {
    name: String,
    x: [usize; 5],
    co: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    groups: usize,
}
fn data(shape: [usize; 5], seed: u32, dtype: DType) -> FlexTensor {
    let mut state = seed;
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 201) as f32 / 100.0 - 1.0
        })
        .collect();
    FlexTensor::from_data(TensorData::new(values, shape).convert_dtype(dtype))
}
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for co in [2, 8, 32, 64] {
        for work in [8192, 16384, 32768, 65536, 131072, 262144, 524288] {
            cases.push(Case {
                name: format!("threshold_1d_co{co}_work{work}"),
                x: [1, 16, 1, 1, work / (co * 8)],
                co,
                kernel: [1, 1, 8],
                stride: [1, 1, 2],
                pad: [0, 0, 3],
                groups: 1,
            });
        }
    }
    for co in [8, 32] {
        for side in [4, 7, 8, 11, 14, 16, 24, 32] {
            cases.push(Case {
                name: format!("threshold_2d_co{co}_side{side}"),
                x: [1, 32, 1, side, side],
                co,
                kernel: [1, 4, 4],
                stride: [1, 2, 2],
                pad: [0, 1, 1],
                groups: 1,
            });
        }
    }
    for (side, ci) in [(7, 64), (14, 128)] {
        cases.push(Case {
            name: format!("threshold_2d_co64_side{side}"),
            x: [1, ci, 1, side, side],
            co: 64,
            kernel: [1, 4, 4],
            stride: [1, 2, 2],
            pad: [0, 1, 1],
            groups: 1,
        });
    }
    for side in [3, 4, 6, 8, 12] {
        cases.push(Case {
            name: format!("threshold_3d_side{side}"),
            x: [1, 8, side, side, side],
            co: 8,
            kernel: [3; 3],
            stride: [2; 3],
            pad: [1; 3],
            groups: 1,
        });
    }
    for co in [2, 8, 32, 64] {
        for ci in [64, 128, 512] {
            for work in [32768, 65536, 131072, 196608, 245760] {
                cases.push(Case {
                    name: format!("gemm_1d_co{co}_ci{ci}_work{work}"),
                    x: [1, ci, 1, 1, work / (co * 8)],
                    co,
                    kernel: [1, 1, 8],
                    stride: [1, 1, 2],
                    pad: [0, 0, 3],
                    groups: 1,
                });
            }
        }
    }
    for co in [8, 32, 64] {
        for ci in [128, 512] {
            for side in [7, 8, 11, 14] {
                cases.push(Case {
                    name: format!("gemm_2d_co{co}_ci{ci}_side{side}"),
                    x: [1, ci, 1, side, side],
                    co,
                    kernel: [1, 4, 4],
                    stride: [1, 2, 2],
                    pad: [0, 1, 1],
                    groups: 1,
                });
            }
        }
    }
    for groups in [2, 3, 4, 5, 7, 8, 9, 16] {
        for length in [512, 2048, 8192] {
            cases.push(Case {
                name: format!("groups_g{groups}_l{length}"),
                x: [1, 4 * groups, 1, 1, length],
                co: 8,
                kernel: [1, 1, 16],
                stride: [1, 1, 2],
                pad: [0, 0, 7],
                groups,
            });
        }
    }
    for batch in [4, 5, 8] {
        cases.push(Case {
            name: format!("groups_batch{batch}_l2048"),
            x: [batch, 4, 1, 1, 2048],
            co: 8,
            kernel: [1, 1, 16],
            stride: [1, 1, 2],
            pad: [0, 0, 7],
            groups: 1,
        });
    }
    cases
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let filter = args.get(1).map(String::as_str).unwrap_or("");
    let name = std::env::var("PROBE_DTYPE").unwrap_or_else(|_| "f32".into());
    use burn_flex::ops::conv_transpose::{conv_transpose3d_f16, conv_transpose3d_f64};
    type Op =
        fn(FlexTensor, FlexTensor, Option<FlexTensor>, &ConvTransposeOptions<3>) -> FlexTensor;
    let (dtype, op): (DType, Op) = match name.as_str() {
        "f16" => (DType::F16, conv_transpose3d_f16),
        "f64" => (DType::F64, conv_transpose3d_f64),
        "f32" => (DType::F32, conv_transpose3d_f32),
        _ => panic!("Unknown data type"),
    };
    for case in cases()
        .into_iter()
        .filter(|c| filter.split(',').any(|f| c.name.contains(f)))
    {
        let x = data(case.x, 12345, dtype);
        let [kd, kh, kw] = case.kernel;
        let w = data([case.x[1], case.co, kd, kh, kw], 67890, dtype);
        let options = ConvTransposeOptions::new(case.stride, case.pad, [0; 3], [1; 3], case.groups);
        let run = || op(x.clone(), w.clone(), None, &options);
        for _ in 0..4 {
            drop(black_box(run()));
        }
        let initial = CURRENT.load(Relaxed);
        PEAK.store(initial, Relaxed);
        let start = Instant::now();
        let output = run();
        let duration = start.elapsed().as_secs_f64();
        let output_bytes = output.bytes().len();
        let peak = PEAK.load(Relaxed).saturating_sub(initial);
        if let Some(path) = args.get(2) {
            std::fs::write(path, output.bytes()).unwrap();
        }
        drop(output);
        let count = (0.015 / duration).ceil().clamp(3.0, 100.0) as usize;
        let mut samples = Vec::new();
        for _ in 0..7 {
            let mut times = Vec::with_capacity(count);
            for _ in 0..count {
                let start = Instant::now();
                let output = black_box(run());
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                drop(output);
            }
            times.sort_by(f64::total_cmp);
            samples.push(times[times.len() / 2]);
        }
        let mut ordered = samples.clone();
        ordered.sort_by(f64::total_cmp);
        println!(
            "{{\"case\":\"{}\",\"dtype\":\"{}\",\"threads\":{},\"ms\":{},\"peak_bytes\":{},\"output_bytes\":{},\"samples\":{:?}}}",
            case.name,
            name,
            rayon::current_num_threads(),
            ordered[ordered.len() / 2],
            peak,
            output_bytes,
            samples
        );
    }
}
