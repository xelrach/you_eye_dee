use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand;

use you_eye_dee::ulid_decode::{
    ulid_to_u128_avx2, ulid_to_u128_scalar, ulid_to_u128_sse2, ulid_to_u128_sse41,
};
use you_eye_dee::ulid_encode::{
    u128_to_ascii_avx2, u128_to_ascii_scalar, u128_to_ascii_scalar_unsafe, u128_to_ascii_ssse3,
};
use you_eye_dee::Ulid;

fn decode_ulid_to_u128_scalar(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_ulid_to_u128_scalar");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    let _result = ulid_to_u128_scalar(ulid);
                }
            });
        },
    );
}

fn decode_ulid_to_u128_sse2(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_ulid_to_u128_sse2");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = ulid_to_u128_sse2(ulid);
                    }
                }
            });
        },
    );
}

fn decode_ulid_to_u128_sse41(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_ulid_to_u128_sse41");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = ulid_to_u128_sse41(ulid);
                    }
                }
            });
        },
    );
}

fn decode_ulid_to_u128_avx2(criterion: &mut Criterion) {
    const COUNT: usize = 10_000;
    let mut group = criterion.benchmark_group("decode_ulid_to_u128_avx2");
    group.throughput(Throughput::Elements(COUNT as u64));
    group.bench_with_input(
        BenchmarkId::from_parameter(COUNT),
        &COUNT,
        |bencher, &count| {
            let ulid_strings = generate_ulid_strings(count);
            bencher.iter(|| {
                for ulid in &ulid_strings {
                    unsafe {
                        let _result = ulid_to_u128_avx2(ulid);
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
                        // let _final_str = &result[6..32];
                    }
                }
            });
        },
    );
}

fn generate_ulid_strings(count: usize) -> Vec<String> {
    let mut ulids = Vec::with_capacity(count);
    for i in 0..count {
        let ulid: Ulid = rand::random::<u128>().into();
        ulids.push(ulid.encode());
    }

    ulids
}

fn generate_ulid_bytes(count: usize) -> Vec<u128> {
    let mut ulids = Vec::with_capacity(count);
    for i in 0..count {
        let ulid = rand::random::<u128>();
        ulids.push(ulid);
    }

    ulids
}

criterion_group!(
    ulid_decode,
    decode_ulid_to_u128_scalar,
    decode_ulid_to_u128_sse2,
    decode_ulid_to_u128_sse41,
    decode_ulid_to_u128_avx2,
);

criterion_group!(
    ulid_encode,
    encode_u128_to_ascii_scalar,
    encode_u128_to_ascii_scalar_unsafe,
    encode_u128_to_ascii_ssse3,
    encode_u128_to_ascii_avx2,
);

criterion_main!(ulid_decode, ulid_encode);
