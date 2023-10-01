use std::arch::x86_64::{
    __m128i, __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_or_si256, _mm256_permute4x64_epi64, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_setr_epi8, _mm256_shuffle_epi8,
    _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_testz_si256, _mm_and_si128,
    _mm_cmpeq_epi8, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8,
    _mm_or_si128, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi64x, _mm_setzero_si128,
    _mm_slli_epi64, _mm_srli_epi64, _mm_testz_si128,
};
use std::ptr::copy_nonoverlapping;

use crate::ULID_LENGTH;

static CROCKFORD_BASE32: [u8; 256] = include!("../resources/crockford_base32_decode.txt");
// Bits set to 1 in this mask are invalid in BASE32. Doing a bitwise and with this mask will find
// invalid inputs
static ULID_INVALID_MASK: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF8, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
    0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
];

static mut PARSE_ULID_FN: unsafe fn(&str) -> Result<u128, DecodeError> = ulid_to_u128_scalar;

#[derive(Debug)]
pub enum DecodeError {
    Unknown,
    WrongLength(usize),
    InvalidCharacter(usize),
}

pub unsafe fn initialize_simd() {
    if is_x86_feature_detected!("avx2") {
        PARSE_ULID_FN = ulid_to_u128_avx2;
    } else if is_x86_feature_detected!("sse4.1") {
        PARSE_ULID_FN = ulid_to_u128_sse41;
    } else if is_x86_feature_detected!("sse2") {
        PARSE_ULID_FN = ulid_to_u128_sse2;
    } else {
        PARSE_ULID_FN = ulid_to_u128_scalar;
    }
}

pub fn ulid_to_u128(input: &str) -> Result<u128, DecodeError> {
    if input.len() != ULID_LENGTH {
        return Err(DecodeError::WrongLength(input.len()));
    }

    unsafe { PARSE_ULID_FN(input) }
}

/**
 * Decodes a ULID string into a `u128`
 */
pub fn ulid_to_u128_scalar(input: &str) -> Result<u128, DecodeError> {
    let mut result = 0u128;
    for (i, byte) in input.as_bytes().iter().enumerate() {
        let quint = CROCKFORD_BASE32[*byte as usize];
        if quint > 31 {
            return Err(DecodeError::InvalidCharacter(i));
        }

        result <<= 5;
        result |= quint as u128;
    }

    Ok(result)
}

#[inline(always)]
fn find_invalid_char(bytes: &[u8; 32]) -> DecodeError {
    for (i, quint) in bytes.iter().enumerate() {
        if *quint > 31 {
            return DecodeError::InvalidCharacter(i);
        }
    }

    DecodeError::Unknown
}

#[inline(always)]
fn gather_bytes_scalar(input: &str) -> [u8; 32] {
    let str_bytes = input.as_bytes();

    let mut bytes: [u8; 32] = [0; 32];
    for i in 0..ULID_LENGTH {
        bytes[i + 6] = CROCKFORD_BASE32[str_bytes[i] as usize];
    }

    bytes
}

/**
 * Decodes a ULID string into a `u128` using SSE2 instructions
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
pub unsafe fn ulid_to_u128_sse2(input: &str) -> Result<u128, DecodeError> {
    // Convert from the string into an array of bytes
    let decoded_bytes = gather_bytes_scalar(input);
    let bytes_ptr = decoded_bytes.as_ptr();

    let high = _mm_loadu_si128(bytes_ptr as *const __m128i);
    let low = _mm_loadu_si128(bytes_ptr.offset(16) as *const __m128i);

    // Check that all of the characters were valid
    let mask_max_high = _mm_loadu_si128(ULID_INVALID_MASK.as_ptr() as *const __m128i);
    let invalid_high = _mm_and_si128(high, mask_max_high);
    let valid_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(invalid_high, _mm_setzero_si128()));
    if valid_mask != 0xFFFF {
        return Err(find_invalid_char(&decoded_bytes));
    }

    let mask_max_low = _mm_loadu_si128((ULID_INVALID_MASK.as_ptr()).offset(16) as *const __m128i);
    let invalid_low = _mm_and_si128(low, mask_max_low);
    let valid_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(invalid_low, _mm_setzero_si128()));
    if valid_mask != 0xFFFF {
        return Err(find_invalid_char(&decoded_bytes));
    }

    // Rearrange the bits into an array
    let mut le_bytes: [u8; 20] = [0; 20];
    shift_ulid_bytes_128(low, le_bytes.as_mut_ptr());
    shift_ulid_bytes_128(high, le_bytes.as_mut_ptr().offset(10));

    Ok(u128::from_le_bytes(le_bytes[0..16].try_into().unwrap()))
}

/**
 * Decodes ULID characters in a `__m128i` into part of a `u128`
 * # Safety
 * `result` MUST have 10 bytes of space
 */
#[inline(always)]
unsafe fn shift_ulid_bytes_128(value: __m128i, result: *mut u8) {
    // Lower eight bytes (64 bits) of the value is:
    // 000hhhhh|000ggggg|000fffff|000eeeee|000ddddd|000ccccc|000bbbbb|000aaaaa

    // Shift left by 5 to align even bytes
    // hhh000gg|ggg000ff|fff000ee|eee000dd|ddd000cc|ccc000bb|bbb000aa|aaa00000
    // mask:
    // 00000011|11100000|00000011|11100000|00000011|11100000|00000011|11100000
    let mut even = _mm_slli_epi64::<5>(value);
    let even_mask = _mm_set1_epi16(0x03_E0);
    even = _mm_and_si128(even, even_mask);

    // Shift right by 8 to align odd bytes
    // 00000000|000hhhhh|000ggggg|000fffff|000eeeee|000ddddd|000ccccc|000bbbbb
    // mask:
    // 00000000|00011111|00000000|00011111|00000000|00011111|00000000|00011111
    let mut odd = _mm_srli_epi64::<8>(value);
    let odd_mask = _mm_set1_epi16(0x00_1F);
    odd = _mm_and_si128(odd, odd_mask);

    // 000000gg|ggghhhhh|000000ee|eeefffff|000000cc|cccddddd|000000aa|aaabbbbb
    let byte_product = _mm_or_si128(even, odd);

    // Shift even words 10 to the left. Don't shift odd words. Sum the results
    // 00000000|00000001|00000100|00000000|00000000|00000001|00000100|00000000
    let multiplicand = _mm_set1_epi32(0x0001_0400);
    // 00000000|0000eeee|efffffgg|ggghhhhh|00000000|0000aaaa|abbbbbcc|cccddddd
    let word_product = _mm_madd_epi16(byte_product, multiplicand);

    // Shift left 20 to align a, b, c, and d
    // ffgggggh|hhhh0000|00000000|aaaaabbb|bbcccccd|dddd0000|00000000|00000000
    let abcd = _mm_slli_epi64::<20>(word_product);

    // Shift right 32 to align e, f, g, and h
    // 00000000|00000000|00000000|00000000|00000000|0000eeee|efffffgg|ggghhhhh
    let efgh = _mm_srli_epi64::<32>(word_product);

    // ffgggggh|hhhh0000|00000000|aaaaabbb|bbcccccd|ddddeeee|efffffgg|ggghhhhh
    let combined = _mm_or_si128(abcd, efgh);

    // mask:
    // 00000000|00000000|00000000|11111111|11111111|11111111|11111111|11111111
    let abcdefgh_mask = _mm_set1_epi64x(0x000000FFFFFFFFFF);
    // 00000000|00000000|00000000|aaaaabbb|bbcccccd|ddddeeee|efffffgg|ggghhhhh
    let combined2 = _mm_and_si128(combined, abcdefgh_mask);

    // We now have two u64s (in big-endian order) with the bytes in each in little-endian order
    let be_u64_le_bytes_ptr: *const u8 = &combined2 as *const __m128i as *const u8;

    // Copy both of the five byte segments
    copy_nonoverlapping::<u8>(be_u64_le_bytes_ptr.offset(8), result, 5);
    copy_nonoverlapping::<u8>(be_u64_le_bytes_ptr, result.offset(5), 5);
}

/**
 * Decodes a ULID string into a `u128` using SSE4.1 instructions
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
pub unsafe fn ulid_to_u128_sse41(input: &str) -> Result<u128, DecodeError> {
    // Convert from the string into an array of bytes
    let decoded_bytes = gather_bytes_scalar(input);
    let bytes_ptr = decoded_bytes.as_ptr();

    let high = _mm_loadu_si128(bytes_ptr as *const __m128i);
    let low = _mm_loadu_si128(bytes_ptr.offset(16) as *const __m128i);

    // Check that all of the characters were valid
    let mask_max_high = _mm_loadu_si128(ULID_INVALID_MASK.as_ptr() as *const __m128i);
    let invalid_high = _mm_and_si128(high, mask_max_high);
    if _mm_testz_si128(invalid_high, invalid_high) == 0 {
        return Err(find_invalid_char(&decoded_bytes));
    }

    let mask_max_low = _mm_loadu_si128((ULID_INVALID_MASK.as_ptr()).offset(16) as *const __m128i);
    let invalid_low = _mm_and_si128(low, mask_max_low);
    if _mm_testz_si128(invalid_low, invalid_low) == 0 {
        return Err(find_invalid_char(&decoded_bytes));
    }

    // Rearrange the bits into an array
    let mut le_bytes: [u8; 20] = [0; 20];
    shift_ulid_bytes_128(low, le_bytes.as_mut_ptr());
    shift_ulid_bytes_128(high, le_bytes.as_mut_ptr().offset(10));

    Ok(u128::from_le_bytes(le_bytes[0..16].try_into().unwrap()))
}

/**
 * Decodes ULID characters in a `__m128i` into part of a `u128`
 * # Safety
 * `result` MUST have 10 bytes of space
 */
#[inline(always)]
unsafe fn madd_ulid_bytes_128(value: __m128i, result: *mut u8) {
    // Lower eight bytes (64 bits) of the value is:
    // 000hhhhh|000ggggg|000fffff|000eeeee|000ddddd|000ccccc|000bbbbb|000aaaaa

    // Shift even bytes 5 to the left. Don't shift odd bytes. Sum the results
    // 00000001|00100000|00000001|00100000|00000001|00100000|00000001|00100000
    let multiplicand = _mm_set1_epi16(0x01_20);
    // 000000gg|ggghhhhh|000000ee|eeefffff|000000cc|cccddddd|000000aa|aaabbbbb
    let byte_product = _mm_maddubs_epi16(value, multiplicand);

    // Shift even words 10 to the left. Don't shift odd words. Sum the results
    // 00000000|00000001|00000100|00000000|00000000|00000001|00000100|00000000
    let multiplicand = _mm_set1_epi32(0x0001_0400);
    // 00000000|0000eeee|efffffgg|ggghhhhh|00000000|0000aaaa|abbbbbcc|cccddddd
    let word_product = _mm_madd_epi16(byte_product, multiplicand);

    // Shift left 20 to align a, b, c, and d
    // ffgggggh|hhhh0000|00000000|aaaaabbb|bbcccccd|dddd0000|00000000|00000000
    let abcd = _mm_slli_epi64::<20>(word_product);

    // Shift right 32 to align e, f, g, and h
    // 00000000|00000000|00000000|00000000|00000000|0000eeee|efffffgg|ggghhhhh
    let efgh = _mm_srli_epi64::<32>(word_product);

    // ffgggggh|hhhh0000|00000000|aaaaabbb|bbcccccd|ddddeeee|efffffgg|ggghhhhh
    let combined = _mm_or_si128(abcd, efgh);

    // mask:
    // 00000000|00000000|00000000|11111111|11111111|11111111|11111111|11111111
    let abcdefgh_mask = _mm_set1_epi64x(0x000000FFFFFFFFFF);
    // 00000000|00000000|00000000|aaaaabbb|bbcccccd|ddddeeee|efffffgg|ggghhhhh
    let combined2 = _mm_and_si128(combined, abcdefgh_mask);

    // We now have two u64s (in big-endian order) with the bytes in each in little-endian order
    let be_u64_le_bytes_ptr: *const u8 = &combined2 as *const __m128i as *const u8;

    // Copy both of the five byte segments
    copy_nonoverlapping::<u8>(be_u64_le_bytes_ptr.offset(8), result, 5);
    copy_nonoverlapping::<u8>(be_u64_le_bytes_ptr, result.offset(5), 5);
}

/**
 * Decodes a ULID string into a `u128` using AVX2 instructions
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
pub unsafe fn ulid_to_u128_avx2(input: &str) -> Result<u128, DecodeError> {
    let decoded_bytes = gather_bytes_scalar(input);
    let bytes_ptr = decoded_bytes.as_ptr();

    let values = _mm256_loadu_si256(bytes_ptr as *const __m256i);

    // Check that all of the characters were valid
    let mask_max = _mm256_loadu_si256(ULID_INVALID_MASK.as_ptr() as *const __m256i);
    let valid = _mm256_testz_si256(values, mask_max);
    if valid == 0 {
        return Err(find_invalid_char(&decoded_bytes));
    }

    // Lower eight bytes (64 bits) of the value is:
    // 000hhhhh|000ggggg|000fffff|000eeeee|000ddddd|000ccccc|000bbbbb|000aaaaa

    // Swap and shift u8s into u16s

    // 00100000|00000001|00100000|00000001|00100000|00000001|00100000|00000001
    let multiplicand = _mm256_set1_epi16(0x01_20);
    // hhhggggg|000000hh|fffeeeee|000000ff|dddccccc|000000dd|bbbaaaaa|000000bb
    let byte_product = _mm256_maddubs_epi16(values, multiplicand);

    // Swap and shift u16s into u32s

    // Treat each pair of bytes as a LE word.
    // Shift even words 10 to the left. Don't shift odd words. Sum the results into u32
    // 00000000|00000100|00000001|00000000|00000000|00000100|00000001|00000000
    let multiplicand = _mm256_set1_epi32(0x0001_0400);
    // 00000000|0000eeee|efffffgg|ggghhhhh|00000000|0000aaaa|abbbbbcc|cccddddd
    let word_product = _mm256_madd_epi16(byte_product, multiplicand);

    // Swap and shift u32s into u64s

    // Shift left 20 to align a, b, c, and d
    // ffgggggh|hhhh0000|00000000|aaaaabbb|bbcccccd|dddd0000|00000000|00000000
    let abcd = _mm256_slli_epi64::<20>(word_product);

    // Shift right 32 to align e, f, g, and h
    // 00000000|00000000|00000000|00000000|00000000|0000eeee|efffffgg|ggghhhhh
    let efgh = _mm256_srli_epi64::<32>(word_product);

    // ffgggggh|hhhh0000|00000000|aaaaabbb|bbcccccd|ddddeeee|efffffgg|ggghhhhh
    let mut combined = _mm256_or_si256(abcd, efgh);

    // mask:
    // 00000000|00000000|00000000|11111111|11111111|11111111|11111111|11111111
    let abcdefgh_mask = _mm256_set1_epi64x(0x000000FFFFFFFFFF);
    // 00000000|00000000|00000000|aaaaabbb|bbcccccd|ddddeeee|efffffgg|ggghhhhh
    combined = _mm256_and_si256(combined, abcdefgh_mask);

    // Swap the quadwords from big endian to little endian

    // ABCDE000|FGHIJ000|KLMNO000|P0000000
    let permuted = _mm256_permute4x64_epi64::<0x1B>(combined);

    #[rustfmt::skip]
    let shuffle = _mm256_setr_epi8(
        0x0, 0x1, 0x2, 0x3, 0x4, 0x8, 0x9, 0xA,
        0xB, 0xC, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, 0x0, 0x1, 0x2, 0x3, 0x4, 0x8,
    );
    // ABCDEFGHIJ000000|000O0000000KLMNP
    let shuffled = _mm256_shuffle_epi8(permuted, shuffle);

    let mut bytes: [u8; 256] = [0; 256];
    _mm256_storeu_si256(bytes.as_mut_ptr() as *mut __m256i, shuffled);

    let mut result = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
    let high = u128::from_le_bytes(bytes[16..32].try_into().unwrap());

    result |= high;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::ulid_decode::*;

    static ULIDS: [&str; 1] = [
        //"00000000000000000000000000",
        //"0000000000ZZZZZZZZZZZZZZZZ",
        //"0123456789ABCDEFGHJKMNPQRS",
        //"7TVWXYZ0123456789ABCDEFGHJ",
        //"7ZZZZZZZZZ0000000000000000",
        "7ZZZZZZZZZZZZZZZZZZZZZZZZZ",
    ];
    static U128S: [u128; 1] = [
        //0x00000000000000000000000000000000,
        //0x000000000000FFFFFFFFFFFFFFFFFFFF,
        //0x0110C8531D0952D8D73E1194E95B5F19,
        //0xFADF3BEF8022190A63A12A5B1AE7C232,
        //0xFFFFFFFFFFFF00000000000000000000,
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    ];

    #[test]
    fn test_ulid_to_u128_scalar() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            let actual = ulid_to_u128_scalar(ulid_str).unwrap();
            assert_eq!(
                actual, *expected,
                "Got: {:#X} Expected: {:#X}",
                actual, *expected
            );
        }
    }

    #[test]
    fn test_ulid_to_u128_sse2() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = ulid_to_u128_sse2(ulid_str).unwrap();
                assert_eq!(
                    actual, *expected,
                    "Got: {:#X} Expected: {:#X}",
                    actual, *expected
                );
            }
        }
    }

    #[test]
    fn test_ulid_to_u128_sse41() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = ulid_to_u128_sse41(ulid_str).unwrap();
                assert_eq!(
                    actual, *expected,
                    "Got: {:#X} Expected: {:#X}",
                    actual, *expected
                );
            }
        }
    }

    #[test]
    fn test_ulid_to_u128_avx2() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = ulid_to_u128_avx2(ulid_str).unwrap();
                assert_eq!(
                    actual, *expected,
                    "Got: {:#X} Expected: {:#X}",
                    actual, *expected
                );
            }
        }
    }
}
