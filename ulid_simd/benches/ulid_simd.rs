use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand;

#[cfg(target_arch = "aarch64")]
use ulid_simd::ulid_decode::aarch64::string_to_ulid_neon;
use ulid_simd::ulid_decode::string_to_ulid_scalar;
#[cfg(target_arch = "x86_64")]
use ulid_simd::ulid_decode::x86_64::{string_to_ulid_avx2, string_to_ulid_ssse3};
#[cfg(target_arch = "x86_64")]
use ulid_simd::ulid_encode::x86_64::{u128_to_ascii_avx2, u128_to_ascii_ssse3};
use ulid_simd::ulid_encode::{u128_to_ascii_scalar, u128_to_ascii_scalar_unsafe};
use ulid_simd::Ulid;

fn decode_string_to_ulid_scalar(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_string_to_ulid_scalar");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    let _result = string_to_ulid_scalar(ulid);
                }
            });
        },
    );
}

#[cfg(target_arch = "x86_64")]
fn decode_string_to_ulid_ssse3(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_string_to_ulid_ssse3");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = string_to_ulid_ssse3(ulid);
                    }
                }
            });
        },
    );
}

#[cfg(target_arch = "x86_64")]
fn decode_string_to_ulid_avx2(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_string_to_ulid_avx2");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = string_to_ulid_avx2(ulid);
                    }
                }
            });
        },
    );
}

#[cfg(target_arch = "aarch64")]
fn decode_string_to_ulid_neon(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_string_to_ulid_neon");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = string_to_ulid_neon(ulid);
                    }
                }
            });
        },
    );
}

fn encode_u128_to_ascii_scalar(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("encode_u128_to_ascii_scalar");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_bytes = generate_ulid_bytes(count);
            bencher.iter(|| {
                for ulid in &ulid_bytes {
                    let _result = u128_to_ascii_scalar(ulid);
                }
            });
        },
    );
}

fn encode_u128_to_ascii_scalar_unsafe(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("encode_u128_to_ascii_scalar_unsafe");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_bytes = generate_ulid_bytes(count);
            bencher.iter(|| {
                for ulid in &ulid_bytes {
                    unsafe {
                        let _result = u128_to_ascii_scalar_unsafe(ulid);
                    }
                }
            });
        },
    );
}

#[cfg(target_arch = "x86_64")]
fn encode_u128_to_ascii_ssse3(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("encode_u128_to_ascii_ssse3");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_bytes = generate_ulid_bytes(count);
            bencher.iter(|| {
                for ulid in &ulid_bytes {
                    unsafe {
                        let _result = u128_to_ascii_ssse3(ulid);
                    }
                }
            });
        },
    );
}

#[cfg(target_arch = "x86_64")]
fn encode_u128_to_ascii_avx2(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("encode_u128_to_ascii_avx2");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_bytes = generate_ulid_bytes(count);
            bencher.iter(|| {
                for ulid in &ulid_bytes {
                    unsafe {
                        let _result = u128_to_ascii_avx2(ulid);
                    }
                }
            });
        },
    );
}

fn generate_ulid_strings(count: usize) -> Vec<String> {
    let mut ulids = Vec::with_capacity(count);
    for _ in 0..count {
        let ulid: Ulid = rand::random::<u128>().into();
        ulids.push(ulid.encode());
    }

    ulids
}

fn generate_ulid_bytes(count: usize) -> Vec<u128> {
    let mut ulids = Vec::with_capacity(count);
    for _ in 0..count {
        let ulid = rand::random::<u128>();
        ulids.push(ulid);
    }

    ulids
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    ulid_decode,
    decode_string_to_ulid_scalar,
    decode_string_to_ulid_ssse3,
    decode_string_to_ulid_avx2,
);

#[cfg(target_arch = "aarch64")]
criterion_group!(
    ulid_decode,
    decode_string_to_ulid_scalar,
    decode_string_to_ulid_neon,
);

#[cfg(target_arch = "x86_64")]
criterion_group!(
    ulid_encode,
    encode_u128_to_ascii_scalar,
    encode_u128_to_ascii_scalar_unsafe,
    encode_u128_to_ascii_ssse3,
    encode_u128_to_ascii_avx2,
);

#[cfg(target_arch = "aarch64")]
criterion_group!(
    ulid_encode,
    encode_u128_to_ascii_scalar,
    encode_u128_to_ascii_scalar_unsafe,
);

criterion_main!(ulid_decode, ulid_encode);
