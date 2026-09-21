use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use burn_backend::{TensorData, ops::ConvTransposeOptions};
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
    name: &'static str,
    x: [usize; 5],
    co: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
}

fn data(shape: [usize; 5], seed: u32) -> FlexTensor {
    let mut state = seed;
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 201) as f32 / 100.0 - 1.0
        })
        .collect();
    FlexTensor::from_data(TensorData::new(values, shape))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let ones = [1, 1, 1];
    let cases = [
        Case {
            name: "image_368_k2s2",
            x: [1, 64, 1, 368, 368],
            co: 64,
            kernel: [1, 2, 2],
            stride: [1, 2, 2],
            pad: [0; 3],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "image_256_k4s2",
            x: [1, 32, 1, 256, 256],
            co: 32,
            kernel: [1, 4, 4],
            stride: [1, 2, 2],
            pad: [0, 1, 1],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "image_128_k3s1",
            x: [1, 64, 1, 128, 128],
            co: 64,
            kernel: [1, 3, 3],
            stride: ones,
            pad: [0, 1, 1],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "volume_32_k3s2",
            x: [1, 16, 32, 32, 32],
            co: 16,
            kernel: [3; 3],
            stride: [2; 3],
            pad: [1; 3],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "volume_24_k2s2",
            x: [1, 32, 24, 24, 24],
            co: 32,
            kernel: [2; 3],
            stride: [2; 3],
            pad: [0; 3],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "audio_l8192_k16s8",
            x: [1, 32, 1, 1, 8192],
            co: 32,
            kernel: [1, 1, 16],
            stride: [1, 1, 8],
            pad: [0, 0, 4],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "audio_l2048_k8s4",
            x: [1, 64, 1, 1, 2048],
            co: 64,
            kernel: [1, 1, 8],
            stride: [1, 1, 4],
            pad: [0, 0, 2],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "groups4_batch2",
            x: [2, 32, 1, 129, 127],
            co: 32,
            kernel: [1, 4, 4],
            stride: [1, 2, 2],
            pad: [0, 1, 1],
            dilation: ones,
            groups: 4,
        },
        Case {
            name: "dilation2_tail",
            x: [1, 16, 1, 131, 137],
            co: 24,
            kernel: [1, 3, 5],
            stride: [1, 2, 3],
            pad: [0, 2, 3],
            dilation: [1, 2, 2],
            groups: 1,
        },
        Case {
            name: "small_7_k4s2",
            x: [1, 64, 1, 7, 7],
            co: 64,
            kernel: [1, 4, 4],
            stride: [1, 2, 2],
            pad: [0, 1, 1],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "small_14_k4s2",
            x: [1, 128, 1, 14, 14],
            co: 64,
            kernel: [1, 4, 4],
            stride: [1, 2, 2],
            pad: [0, 1, 1],
            dilation: ones,
            groups: 1,
        },
        Case {
            name: "depthwise_256",
            x: [1, 16, 1, 256, 256],
            co: 16,
            kernel: [1, 3, 3],
            stride: [1, 2, 2],
            pad: [0, 1, 1],
            dilation: ones,
            groups: 16,
        },
    ];
    for case in cases {
        if args.get(1).is_some_and(|s| s != "all" && s != case.name) {
            continue;
        }
        let x = data(case.x, 12345);
        let [kd, kh, kw] = case.kernel;
        let w = data([case.x[1], case.co / case.groups, kd, kh, kw], 67890);
        let options =
            ConvTransposeOptions::new(case.stride, case.pad, [0; 3], case.dilation, case.groups);
        let run = || conv_transpose3d_f32(x.clone(), w.clone(), None, &options);
        drop(black_box(run()));
        let initial = CURRENT.load(Relaxed);
        PEAK.store(initial, Relaxed);
        let start = Instant::now();
        let output = run();
        let first = start.elapsed().as_secs_f64() * 1000.0;
        let peak = PEAK.load(Relaxed).saturating_sub(initial);
        let output_bytes = output.bytes().len();
        let sum: f64 = output.storage::<f32>().iter().map(|&x| x as f64).sum();
        if let Some(path) = args.get(2) {
            std::fs::write(path, output.bytes()).unwrap();
        }
        drop(output);
        let count = (200.0 / first).ceil().clamp(3.0, 25.0) as usize;
        let mut times = Vec::with_capacity(count);
        for _ in 0..count {
            let start = Instant::now();
            let y = black_box(run());
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            drop(y);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{{\"case\":\"{}\",\"ms\":{},\"peak_bytes\":{},\"output_bytes\":{},\"sum\":{},\"samples\":{:?}}}",
            case.name,
            times[times.len() / 2],
            peak,
            output_bytes,
            sum,
            times
        );
    }
}
