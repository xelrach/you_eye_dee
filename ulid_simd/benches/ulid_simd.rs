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

const COUNT: usize = 1_000;

fn encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Elements(COUNT as u64));

    group.bench_with_input(
        BenchmarkId::new("u128_to_ascii_scalar", COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_bytes = generate_ulid_bytes(count);
            bencher.iter(|| {
                for ulid in &ulid_bytes {
                    let _result = u128_to_ascii_scalar(ulid);
                }
            });
        }
    );
    group.bench_with_input(
        BenchmarkId::new("u128_to_ascii_scalar_unsafe", COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_bytes = generate_ulid_bytes(count);
            bencher.iter(|| {
                for ulid in &ulid_bytes {
                    let _result = u128_to_ascii_scalar_unsafe(ulid);
                }
            });
        }
    );
    #[cfg(all(target_arch = "x64_64", target_feature = "ssse3"))]
    {
        group.bench_with_input(
            BenchmarkId::new("u128_to_ascii_ssse3", crate::COUNT),
            &crate::COUNT,
            |bencher, &count| {
                let ulid_bytes = crate::generate_ulid_bytes(count);
                bencher.iter(|| {
                    for ulid in &ulid_bytes {
                        unsafe {
                            let _result = u128_to_ascii_ssse3(ulid);
                        }
                    }
                });
            }
        );
    }
    #[cfg(all(target_arch = "x64_64", target_feature = "axv2"))]
    {
        group.bench_with_input(
            BenchmarkId::new("u128_to_ascii_avx2", crate::COUNT),
            &crate::COUNT,
            |bencher, &count| {
                let ulid_bytes = crate::generate_ulid_bytes(count);
                bencher.iter(|| {
                    for ulid in &ulid_bytes {
                        unsafe {
                            let _result = u128_to_ascii_avx2(ulid);
                        }
                    }
                });
            }
        );
    }

    group.finish();
}

fn decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Elements(COUNT as u64));

    group.bench_with_input(
        BenchmarkId::new("string_to_ulid_scalar", COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = string_to_ulid_scalar(ulid);
                    }
                }
            });
        }
    );
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        group.bench_with_input(
            BenchmarkId::new("string_to_ulid_neon", COUNT),
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
            }
        );
    }
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

criterion_group!(ulid_foo, encode, decode);
criterion_main!(ulid_foo);
