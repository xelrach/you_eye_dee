use crate::{Ulid, ULID_LENGTH};
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_loadu_si256, _mm256_testz_si256, _mm_and_si128, _mm_cmpeq_epi64,
    _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8, _mm_or_si128,
    _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi64x, _mm_setzero_si128, _mm_slli_epi64,
    _mm_srli_epi64,
};

static CROCKFORD_BASE32_ENCODE: [char; 256] = include!("../resources/crockford_base32_encode.txt");

static mut ENCODE_ULID_FN: unsafe fn(ulid: &u128) -> String = u128_to_ascii_scalar;

pub fn ulid_to_string(input: &Ulid) -> String {
    unsafe { ENCODE_ULID_FN(&input.0) }
}

fn u128_to_ascii_scalar(ulid: &u128) -> String {
    let mut encoded = String::with_capacity(ULID_LENGTH);
    for i in 0..ULID_LENGTH {
        let mut shifted: u8 = (ulid >> (125 - i * 5)) as u8;
        shifted &= 0x1F;

        let character = CROCKFORD_BASE32_ENCODE[shifted as usize];
        encoded.push(character);
    }

    encoded
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::ulid_encode::*;

    static ULIDS: [&str; 6] = [
        "00000000000000000000000000",
        "0000000000ZZZZZZZZZZZZZZZZ",
        "0123456789ABCDEFGHJKMNPQRS",
        "7TVWXYZ0123456789ABCDEFGHJ",
        "7ZZZZZZZZZ0000000000000000",
        "7ZZZZZZZZZZZZZZZZZZZZZZZZZ",
    ];
    static U128S: [u128; 6] = [
        0x00000000000000000000000000000000,
        0x000000000000FFFFFFFFFFFFFFFFFFFF,
        0x0110C8531D0952D8D73E1194E95B5F19,
        0xFADF3BEF8022190A63A12A5B1AE7C232,
        0xFFFFFFFFFFFF00000000000000000000,
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    ];

    #[test]
    fn test_u128_to_ascii_scalar() {
        for (ulid_str, value) in zip(&ULIDS, &U128S) {
            let actual = u128_to_ascii_scalar(value);
            assert_eq!(actual, *ulid_str);
        }
    }
}
