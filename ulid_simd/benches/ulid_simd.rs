use criterion::{criterion_group, criterion_main, Criterion};
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

fn encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let rand_u128 = rand::random::<u128>();

    group.bench_function("u128_to_ascii_scalar", |bencher| {
        bencher.iter(|| {
            let _result = u128_to_ascii_scalar(&rand_u128);
        });
    });
    group.bench_function("u128_to_ascii_scalar_unsafe", |bencher| {
        bencher.iter(|| {
            let _result = u128_to_ascii_scalar_unsafe(&rand_u128);
        });
    });
    #[cfg(all(target_arch = "x64_64", target_feature = "ssse3"))]
    {
        group.bench_function("u128_to_ascii_ssse3", |bencher| {
            bencher.iter(|| unsafe {
                let _result = u128_to_ascii_ssse3(&rand_u128);
            });
        });
    }
    #[cfg(all(target_arch = "x64_64", target_feature = "axv2"))]
    {
        group.bench_function("u128_to_ascii_avx2", |bencher| {
            bencher.iter(|| unsafe {
                let _result = u128_to_ascii_avx2(&rand_u128);
            });
        });
    }

    group.finish();
}

fn decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    let ulid_string = Ulid::from(rand::random::<u128>()).encode();

    group.bench_function("string_to_ulid_scalar", |bencher| {
        bencher.iter(|| unsafe {
            let _result = string_to_ulid_scalar(&ulid_string);
        });
    });
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        group.bench_function("string_to_ulid_neon", |bencher| {
            bencher.iter(|| unsafe {
                let _result = string_to_ulid_neon(&ulid_string);
            });
        });
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
