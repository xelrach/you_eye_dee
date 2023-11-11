use std::arch::x86_64::{
    __m128i, __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpgt_epi8, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_or_si256, _mm256_permute4x64_epi64,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_setr_epi8,
    _mm256_shuffle_epi8, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
    _mm256_sub_epi8, _mm256_testz_si256, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi8,
    _mm_cmpgt_epi8, _mm_loadu_si128, _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8,
    _mm_or_si128, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi64x, _mm_set1_epi8, _mm_setr_epi8,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_epi64, _mm_srli_epi64, _mm_storeu_si128,
    _mm_sub_epi8,
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

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    Unknown,
    WrongLength(usize),
    InvalidCharacter(usize),
}

pub unsafe fn initialize_simd() {
    if is_x86_feature_detected!("avx2") {
        PARSE_ULID_FN = ulid_to_u128_avx2;
    } else if is_x86_feature_detected!("ssse3") {
        PARSE_ULID_FN = ulid_to_u128_ssse3;
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
        if quint > 31 || (i == 0 && quint > 7) {
            return Err(DecodeError::InvalidCharacter(i));
        }

        result <<= 5;
        result |= quint as u128;
    }

    Ok(result)
}

pub unsafe fn ulid_to_u128_scalar_unsafe(input: &str) -> Result<u128, DecodeError> {
    let mut result = 0u128;
    for (i, byte) in input.as_bytes().iter().enumerate() {
        let quint = CROCKFORD_BASE32.get_unchecked(*byte as usize);
        if *quint > 31 || (i == 0 && *quint > 7) {
            return Err(DecodeError::InvalidCharacter(i));
        }

        result <<= 5;
        result |= quint as u128;
    }

    Ok(result)
}

#[inline(always)]
unsafe fn find_invalid_char_m256i(decoded: __m256i) -> DecodeError {
    let mut bytes: [u8; 32] = [0; 32];
    _mm256_storeu_si256(bytes.as_mut_ptr() as *mut __m256i, decoded);
    for (i, quint) in bytes.iter().enumerate() {
        if *quint > 31 || (i == 6 && *quint > 7) {
            return DecodeError::InvalidCharacter(i - 6);
        }
    }

    DecodeError::Unknown
}

#[inline(always)]
unsafe fn find_invalid_char_m128i(high: __m128i, low: __m128i) -> DecodeError {
    let mut bytes: [u8; 16] = [0; 16];
    _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, high);
    for (i, quint) in bytes.iter().enumerate() {
        if *quint > 31 || (i == 6 && *quint > 7) {
            return DecodeError::InvalidCharacter(i - 6);
        }
    }

    _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, low);
    for (i, quint) in bytes.iter().enumerate() {
        if *quint > 31 {
            return DecodeError::InvalidCharacter(i + 10);
        }
    }

    DecodeError::Unknown
}

/**
 * Decodes a ULID string into a `u128` using SSE4.1 instructions
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
pub unsafe fn ulid_to_u128_ssse3(input: &str) -> Result<u128, DecodeError> {
    // Convert from the string into an array of bytes
    let mut high = shuffle_lookup_ssse3(input.as_bytes().as_ptr().offset(-6));
    // Zero the extra six bytes at the end of the array
    #[rustfmt::skip]
        let actual_bytes_mask = _mm_setr_epi8(
        0, 0, 0, 0, 0, 0,-1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );
    high = _mm_and_si128(high, actual_bytes_mask);

    let low = shuffle_lookup_ssse3(input.as_bytes().as_ptr().offset(10));

    // Check that all of the characters were valid
    let mask_max_high = _mm_loadu_si128(ULID_INVALID_MASK.as_ptr() as *const __m128i);
    let invalid_high = _mm_and_si128(high, mask_max_high);
    let valid_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(invalid_high, _mm_setzero_si128()));
    if valid_mask != 0xFFFF {
        return Err(find_invalid_char_m128i(high, low));
    }

    let mask_max_low = _mm_loadu_si128((ULID_INVALID_MASK.as_ptr()).offset(16) as *const __m128i);
    let invalid_low = _mm_and_si128(low, mask_max_low);
    let valid_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(invalid_low, _mm_setzero_si128()));
    if valid_mask != 0xFFFF {
        return Err(find_invalid_char_m128i(high, low));
    }

    // Rearrange the bits into an array
    let mut le_bytes: [u8; 20] = [0; 20];
    madd_ulid_bytes_128(low, le_bytes.as_mut_ptr());
    madd_ulid_bytes_128(high, le_bytes.as_mut_ptr().offset(10));

    Ok(u128::from_le_bytes(le_bytes[0..16].try_into().unwrap()))
}

#[inline(always)]
unsafe fn shuffle_lookup_ssse3(ulid_str_ptr: *const u8) -> __m128i {
    let encoded_bytes = _mm_loadu_si128(ulid_str_ptr as *const __m128i);

    let mut result = _mm_set1_epi8(-1);

    // Decode digits
    let mut decoded_digits = _mm_sub_epi8(encoded_bytes, _mm_set1_epi8(0x30));
    let char_is_digit = _mm_and_si128(
        _mm_cmpgt_epi8(encoded_bytes, _mm_set1_epi8(0x29)),
        _mm_cmpgt_epi8(_mm_set1_epi8(0x3A), encoded_bytes),
    );
    decoded_digits = _mm_and_si128(char_is_digit, decoded_digits);
    let not_digits = _mm_andnot_si128(char_is_digit, result);
    result = _mm_or_si128(decoded_digits, not_digits);

    let low_nibble = _mm_and_si128(encoded_bytes, _mm_set1_epi8(0x0F));
    // Decode AO and ao range
    // [A, O] lookup table
    #[rustfmt::skip]
    let lookup_ao = _mm_setr_epi8(
        -1, 0x0A, 0x0B,0x0C,0x0D,0x0E,0x0F,0x10,
        0x11,0x01,0x12,0x13,0x01,0x14,0x15,0x00,
    );
    let mut decoded_ao = _mm_shuffle_epi8(lookup_ao, low_nibble);
    // Less than 'A', greater than 'O' and less than 'a', greater than 'o'
    let char_in_ao = _mm_or_si128(
        _mm_and_si128(
            _mm_cmpgt_epi8(encoded_bytes, _mm_set1_epi8(0x40)),
            _mm_cmpgt_epi8(_mm_set1_epi8(0x50), encoded_bytes),
        ),
        _mm_and_si128(
            _mm_cmpgt_epi8(encoded_bytes, _mm_set1_epi8(0x60)),
            _mm_cmpgt_epi8(_mm_set1_epi8(0x70), encoded_bytes),
        ),
    );
    decoded_ao = _mm_and_si128(char_in_ao, decoded_ao);
    let not_in_ao = _mm_andnot_si128(char_in_ao, result);
    result = _mm_or_si128(decoded_ao, not_in_ao);

    // Decode PZ and pz range
    // Less than 'P', greater than 'Z' and less than 'p', greater than 'z'
    // [P, Z] lookup table
    #[rustfmt::skip]
    let lookup_pz = _mm_setr_epi8(
        0x16,0x17,0x18,0x19,0x1A,-1,0x1B,0x1C,
        0x1D,0x1E,0x1F,-1,-1,-1,-1,-1,
    );
    let mut decoded_pz = _mm_shuffle_epi8(lookup_pz, low_nibble);
    // Less than 'A', greater than 'O' and less than 'a', greater than 'o'
    let char_in_pz = _mm_or_si128(
        _mm_and_si128(
            _mm_cmpgt_epi8(encoded_bytes, _mm_set1_epi8(0x4F)),
            _mm_cmpgt_epi8(_mm_set1_epi8(0x5B), encoded_bytes),
        ),
        _mm_and_si128(
            _mm_cmpgt_epi8(encoded_bytes, _mm_set1_epi8(0x6F)),
            _mm_cmpgt_epi8(_mm_set1_epi8(0x7B), encoded_bytes),
        ),
    );
    decoded_pz = _mm_and_si128(char_in_pz, decoded_pz);
    let not_in_pz = _mm_andnot_si128(char_in_pz, result);
    result = _mm_or_si128(decoded_pz, not_in_pz);

    result
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
    let values = shuffle_lookup_avx2(input);

    // Check that all of the characters were valid
    let mask_max = _mm256_loadu_si256(ULID_INVALID_MASK.as_ptr() as *const __m256i);
    let valid = _mm256_testz_si256(values, mask_max);
    if valid == 0 {
        return Err(find_invalid_char_m256i(values));
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

    let mut bytes: [u8; 32] = [0; 32];
    _mm256_storeu_si256(bytes.as_mut_ptr() as *mut __m256i, shuffled);

    let mut result = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
    let high = u128::from_le_bytes(bytes[16..32].try_into().unwrap());

    result |= high;

    Ok(result)
}

#[inline(always)]
unsafe fn shuffle_lookup_avx2(ulid_str: &str) -> __m256i {
    let encoded_bytes =
        _mm256_loadu_si256(ulid_str.as_bytes().as_ptr().offset(-6) as *const __m256i);

    let mut result = _mm256_set1_epi8(-1);

    // Decode digits
    let mut decoded_digits = _mm256_sub_epi8(encoded_bytes, _mm256_set1_epi8(0x30));
    let char_is_digit = _mm256_and_si256(
        _mm256_cmpgt_epi8(encoded_bytes, _mm256_set1_epi8(0x29)),
        _mm256_cmpgt_epi8(_mm256_set1_epi8(0x3A), encoded_bytes),
    );
    decoded_digits = _mm256_and_si256(char_is_digit, decoded_digits);
    let not_digits = _mm256_andnot_si256(char_is_digit, result);
    result = _mm256_or_si256(decoded_digits, not_digits);

    let low_nibble = _mm256_and_si256(encoded_bytes, _mm256_set1_epi8(0x0F));
    // Decode AO and ao range
    // [A, O] lookup table
    #[rustfmt::skip]
    let lookup_ao = _mm256_setr_epi8(
        -1, 0x0A, 0x0B,0x0C,0x0D,0x0E,0x0F,0x10,
        0x11,0x01,0x12,0x13,0x01,0x14,0x15,0x00,
        -1, 0x0A, 0x0B,0x0C,0x0D,0x0E,0x0F,0x10,
        0x11,0x01,0x12,0x13,0x01,0x14,0x15,0x00,
    );
    let mut decoded_ao = _mm256_shuffle_epi8(lookup_ao, low_nibble);
    // Less than 'A', greater than 'O' and less than 'a', greater than 'o'
    let char_in_ao = _mm256_or_si256(
        _mm256_and_si256(
            _mm256_cmpgt_epi8(encoded_bytes, _mm256_set1_epi8(0x40)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x50), encoded_bytes),
        ),
        _mm256_and_si256(
            _mm256_cmpgt_epi8(encoded_bytes, _mm256_set1_epi8(0x60)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x70), encoded_bytes),
        ),
    );
    decoded_ao = _mm256_and_si256(char_in_ao, decoded_ao);
    let not_in_ao = _mm256_andnot_si256(char_in_ao, result);
    result = _mm256_or_si256(decoded_ao, not_in_ao);

    // Decode PZ and pz range
    // Less than 'P', greater than 'Z' and less than 'p', greater than 'z'
    // [P, Z] lookup table
    #[rustfmt::skip]
    let lookup_pz = _mm256_setr_epi8(
        0x16,0x17,0x18,0x19,0x1A,-1,0x1B,0x1C,
        0x1D,0x1E,0x1F,-1,-1,-1,-1,-1,
        0x16,0x17,0x18,0x19,0x1A,-1,0x1B,0x1C,
        0x1D,0x1E,0x1F,-1,-1,-1,-1,-1,
    );
    let mut decoded_pz = _mm256_shuffle_epi8(lookup_pz, low_nibble);
    // Less than 'A', greater than 'O' and less than 'a', greater than 'o'
    let char_in_pz = _mm256_or_si256(
        _mm256_and_si256(
            _mm256_cmpgt_epi8(encoded_bytes, _mm256_set1_epi8(0x4F)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x5B), encoded_bytes),
        ),
        _mm256_and_si256(
            _mm256_cmpgt_epi8(encoded_bytes, _mm256_set1_epi8(0x6F)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(0x7B), encoded_bytes),
        ),
    );
    decoded_pz = _mm256_and_si256(char_in_pz, decoded_pz);
    let not_in_pz = _mm256_andnot_si256(char_in_pz, result);
    result = _mm256_or_si256(decoded_pz, not_in_pz);

    // Zero the extra six bytes at the end of the array
    #[rustfmt::skip]
    let actual_bytes_mask = _mm256_setr_epi8(
         0, 0, 0, 0, 0, 0,-1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );

    _mm256_and_si256(result, actual_bytes_mask)
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::ulid_decode::*;

    static ULIDS: [&str; 9] = [
        "00000000000000000000000000",
        "0000000000ZZZZZZzzzzzzzzZZ",
        "0123456789ABCDEFGHJKMNPQRS",
        "7TVWXYZ0123456789ABCDEFGHJ",
        "0123456789abcdefghjkmnpqrs",
        "7tvwxyz0123456789abcdefghj",
        "1IiLl0Oo1IiLl0Oo1IiLl0Oo1I",
        "7ZZZZZZZZZ0000000000000000",
        "7ZZZZZZZZZZZZZZZZZZZZZZZZZ",
    ];
    static U128S: [u128; 9] = [
        0x00000000000000000000000000000000,
        0x000000000000FFFFFFFFFFFFFFFFFFFF,
        0x0110C8531D0952D8D73E1194E95B5F19,
        0xFADF3BEF8022190A63A12A5B1AE7C232,
        0x0110C8531D0952D8D73E1194E95B5F19,
        0xFADF3BEF8022190A63A12A5B1AE7C232,
        0x21084200002108420000210842000021,
        0xFFFFFFFFFFFF00000000000000000000,
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
    fn test_ulid_to_u128_ssse3() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = ulid_to_u128_ssse3(ulid_str).unwrap();
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

    #[test]
    fn test_ulid_to_u128_wrong_length() {
        let actual = ulid_to_u128("").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(0));

        let actual = ulid_to_u128("\0").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(1));

        let actual = ulid_to_u128("!!!!!!!!!!!!!!!!!!!!!!!!!").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(25));

        let actual = ulid_to_u128("???????????????????????????").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(27));
    }

    static INVALID_ULIDS: [&str; 10] = [
        "8ZZZZZZZZZZZZZZZZZZZZZZZZZ",
        "\00000000000000000000000000",
        "0000000000000000000000000/",
        "0000000000000000000000000:",
        "0000000000000000000000000@",
        "0000000000000000000000000[",
        "``````````````````````````",
        "{{{{{{{{{{{{{{{{{{{{{{{{{{",
        "0000000000000000000000000U",
        "0000000000000000000000000u",
    ];

    static INVALID_CHAR_POSITION: [usize; 10] = [0, 0, 25, 25, 25, 25, 0, 0, 25, 25];

    #[test]
    fn test_ulid_to_u128_scalar_invalid_char() {
        for (ulid_str, expected) in zip(&INVALID_ULIDS, &INVALID_CHAR_POSITION) {
            let actual = ulid_to_u128_scalar(ulid_str).unwrap_err();
            assert_eq!(
                actual,
                DecodeError::InvalidCharacter(*expected),
                "Input: {}",
                ulid_str
            );
        }
    }

    #[test]
    fn test_ulid_to_u128_ssse3_invalid_char() {
        unsafe {
            for (ulid_str, expected) in zip(&INVALID_ULIDS, &INVALID_CHAR_POSITION) {
                let actual = ulid_to_u128_ssse3(ulid_str).unwrap_err();
                assert_eq!(
                    actual,
                    DecodeError::InvalidCharacter(*expected),
                    "Input: {}",
                    ulid_str
                );
            }
        }
    }

    #[test]
    fn test_ulid_to_u128_avx2_invalid_char() {
        unsafe {
            for (ulid_str, expected) in zip(&INVALID_ULIDS, &INVALID_CHAR_POSITION) {
                let actual = ulid_to_u128_avx2(ulid_str).unwrap_err();
                assert_eq!(
                    actual,
                    DecodeError::InvalidCharacter(*expected),
                    "Input: {}",
                    ulid_str
                );
            }
        }
    }
}
