use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi8, _mm256_and_si256, _mm256_broadcastsi128_si256,
    _mm256_cmpgt_epi8, _mm256_mulhi_epu16, _mm256_mullo_epi16, _mm256_or_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi64x, _mm256_set1_epi8,
    _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_slli_si256, _mm256_srli_si256,
    _mm256_storeu_si256, _mm_add_epi8, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_mulhi_epu16, _mm_mullo_epi16, _mm_or_si128, _mm_set1_epi16, _mm_set1_epi64x, _mm_set1_epi8,
    _mm_setr_epi8, _mm_shuffle_epi8, _mm_slli_si128, _mm_srli_si128, _mm_storeu_si128,
};

use crate::ULID_LENGTH;

/**
 * Encodes a `u128` into a ULID string using SSSE3 instructions
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
#[target_feature(enable = "ssse3")]
pub unsafe fn u128_to_ascii_ssse3(ulid: &u128) -> String {
    let bytes = _mm_loadu_si128(ulid.to_le_bytes().as_ptr() as *const __m128i);

    // The bytes of the input are:
    // zzzzzyyy|yyxxxxxw|wwwwvvvv|vuuuuutt|tttsssss|rrrrrqqq|qqpppppo|oooonnnn|nmmmmmll|lllkkkkk|jjjjjiii|iihhhhhg|ggggffff|feeeeedd|dddccccc|bbbbbaaa

    // Get the 10 bytes which will become the low 16 characters
    // Copy pairs of bytes and reverse the endianness.
    // Paris will be in LE order, the array in BE order.
    // ??????ll|lllkkkkk|????nnnn|nmmmmm??|??pppppo|oooo????|rrrrrqqq|qq??????|??????tt|tttsssss|????vvvv|vuuuuu??|??xxxxxw|wwww????|zzzzzyyy|yy??????
    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(
        0x08, 0x09, 0x07, 0x08, 0x06, 0x07, 0x05, 0x06,
        0x03, 0x04, 0x02, 0x03, 0x01, 0x02, 0x00, 0x01,
    );
    let low_bytes = _mm_shuffle_epi8(bytes, shuffle);
    let low_chars = encode_bytes_ssse3(low_bytes);

    // Get the 6 bytes which will become the high 10 characters
    // Copy pairs of bytes and reverse the endianness.
    // Paris will be in LE order, the array in BE order.
    // 00000000|00000000|00000000|00000000|00000000|00000000|bbbbbaaa|00000000|??????dd|dddccccc|????ffff|feeeee??|??hhhhhg|gggg????|jjjjjiii|ii??????
    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, 0x0F, -1,
        0x0D, 0x0E, 0x0C, 0x0D, 0x0B, 0x0C, 0x0A, 0x0B,
    );
    let mid_bytes = _mm_shuffle_epi8(bytes, shuffle);
    let mid_chars = encode_bytes_ssse3(mid_bytes);
    // Shift 6 bytes to more significant
    let high_chars = _mm_srli_si128::<6>(mid_chars);

    let mut chars: Box<[u8; ULID_LENGTH]> = Box::new([0x00; ULID_LENGTH]);

    _mm_storeu_si128(chars.as_mut_ptr() as *mut __m128i, high_chars);
    _mm_storeu_si128(chars.as_mut_ptr().offset(10) as *mut __m128i, low_chars);

    String::from_raw_parts(
        Box::<[u8; ULID_LENGTH]>::into_raw(chars) as *mut u8,
        ULID_LENGTH,
        ULID_LENGTH,
    )
}

/**
 * Encodes part of a `u128` into part of a ULID string
 * # Safety
 * No restrictions on caller
 */
#[target_feature(enable = "ssse3")]
unsafe fn encode_bytes_ssse3(bytes: __m128i) -> __m128i {
    // Shift to the left: a by 8, c by 6, e by 4, g by 2
    let multiplicand = _mm_set1_epi64x(0x0100_0040_0010_0004);
    // 000aaaaa|00000000|000ccccc|00000000|000eeeee|00000000|000ggggg|00000000
    let odds = _mm_and_si128(_mm_mullo_epi16(bytes, multiplicand), _mm_set1_epi16(0x1F00));
    // Shift to the right: b by 5, d by 7, f by 9, h by 11
    let multiplicand = _mm_set1_epi64x(0x0800_0200_0080_0020);
    // 00000000|000bbbbb|00000000|000ddddd|00000000|000fffff|00000000|000hhhhh
    let evens = _mm_and_si128(_mm_mulhi_epu16(bytes, multiplicand), _mm_set1_epi16(0x001F));

    // 000aaaaa|000bbbbb|000ccccc|000ddddd|000eeeee|000fffff|000ggggg|000hhhhh
    let quints = _mm_or_si128(odds, evens);

    // Add '0' to the values
    let mut chars = _mm_add_epi8(quints, _mm_set1_epi8(0x30));

    // Add 7 to values > 9 to reach 'A'
    let delta = _mm_and_si128(_mm_cmpgt_epi8(quints, _mm_set1_epi8(9)), _mm_set1_epi8(7));
    chars = _mm_add_epi8(chars, delta);

    // Add 1 to values > 17 to skip 'I'
    let delta = _mm_and_si128(_mm_cmpgt_epi8(quints, _mm_set1_epi8(17)), _mm_set1_epi8(1));
    chars = _mm_add_epi8(chars, delta);

    // Add 1 to values > 19 to skip 'L'
    let delta = _mm_and_si128(_mm_cmpgt_epi8(quints, _mm_set1_epi8(19)), _mm_set1_epi8(1));
    chars = _mm_add_epi8(chars, delta);

    // Add 1 to values > 21 to skip 'O'
    let delta = _mm_and_si128(_mm_cmpgt_epi8(quints, _mm_set1_epi8(21)), _mm_set1_epi8(1));
    chars = _mm_add_epi8(chars, delta);

    // Add 1 to values > 26 to skip 'U'
    let delta = _mm_and_si128(_mm_cmpgt_epi8(quints, _mm_set1_epi8(26)), _mm_set1_epi8(1));
    chars = _mm_add_epi8(chars, delta);

    chars
}

/**
 * Encodes a `u128` into a ULID string using AVX2 instructions
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
#[target_feature(enable = "avx2")]
pub unsafe fn u128_to_ascii_avx2(ulid: &u128) -> String {
    let le_bytes = ulid.to_ne_bytes().as_ptr();

    let input = _mm_loadu_si128(le_bytes as *const __m128i);
    let bytes = _mm256_broadcastsi128_si256(input);

    // The bytes of the input are:
    // pppppooo|oonnnnnm|mmmmllll|lkkkkkjj|jjjiiiii hhhhhggg|ggfffffe|eeeedddd|dcccccbb|bbbaaaaa ????????|????????|????????|????????|???????? ????????

    // Copy pairs of bytes and reverse the endianness.
    // Paris will be in LE order, the array in BE order.
    // ??????bb|bbbaaaaa|????dddd|dccccc??|??fffffe|eeee????|hhhhhggg|gg?????? ?????jj|jjjiiiii|????llll|lkkkkk??|??nnnnnm|mmmm????|pppppooo|oo??????
    #[rustfmt::skip]
        let shuffle = _mm256_setr_epi8(
        -1, -1, -1, -1, -1, -1, 0x0F, -1,
        0x0D, 0x0E, 0x0C, 0x0D, 0x0B, 0x0C, 0x0A, 0x0B,
        0x08, 0x09, 0x07, 0x08, 0x06, 0x07, 0x05, 0x06,
        0x03, 0x04, 0x02, 0x03, 0x01, 0x02, 0x00, 0x01,
    );
    let duplicated_bytes = _mm256_shuffle_epi8(bytes, shuffle);

    // Shift to the left: a by 8, c by 6, e by 4, g by 2
    let multiplicand = _mm256_set1_epi64x(0x0100_0040_0010_0004);
    // 000aaaaa|00000000|000ccccc|00000000|000eeeee|00000000|000ggggg|00000000
    let odds = _mm256_and_si256(
        _mm256_mullo_epi16(duplicated_bytes, multiplicand),
        _mm256_set1_epi16(0x1F00),
    );
    // Shift to the right: b by 5, d by 7, f by 9, h by 11
    let multiplicand = _mm256_set1_epi64x(0x0800_0200_0080_0020);
    // 00000000|000bbbbb|00000000|000ddddd|00000000|000fffff|00000000|000hhhhh
    let evens = _mm256_and_si256(
        _mm256_mulhi_epu16(duplicated_bytes, multiplicand),
        _mm256_set1_epi16(0x001F),
    );

    // 000aaaaa|000bbbbb|000ccccc|000ddddd|000eeeee|000fffff|000ggggg|000hhhhh
    let quints = _mm256_or_si256(odds, evens);

    // Add '0' to the values
    let mut chars = _mm256_add_epi8(quints, _mm256_set1_epi8(0x30));

    // Add 7 to values > 9 to reach 'A'
    let delta = _mm256_and_si256(
        _mm256_cmpgt_epi8(quints, _mm256_set1_epi8(9)),
        _mm256_set1_epi8(7),
    );
    chars = _mm256_add_epi8(chars, delta);

    // Add 1 to values > 17 to skip 'I'
    let delta = _mm256_and_si256(
        _mm256_cmpgt_epi8(quints, _mm256_set1_epi8(17)),
        _mm256_set1_epi8(1),
    );
    chars = _mm256_add_epi8(chars, delta);

    // Add 1 to values > 19 to skip 'L'
    let delta = _mm256_and_si256(
        _mm256_cmpgt_epi8(quints, _mm256_set1_epi8(19)),
        _mm256_set1_epi8(1),
    );
    chars = _mm256_add_epi8(chars, delta);

    // Add 1 to values > 21 to skip 'O'
    let delta = _mm256_and_si256(
        _mm256_cmpgt_epi8(quints, _mm256_set1_epi8(21)),
        _mm256_set1_epi8(1),
    );
    chars = _mm256_add_epi8(chars, delta);

    // Add 1 to values > 26 to skip 'U'
    let delta = _mm256_and_si256(
        _mm256_cmpgt_epi8(quints, _mm256_set1_epi8(26)),
        _mm256_set1_epi8(1),
    );
    chars = _mm256_add_epi8(chars, delta);

    // Array now looks like:
    // ??????AB|CDEFGHIJ KLMNOPQR|STUVWXYZ

    // ABCDEFGH|IJ000000 QRSTUVWX|YZ000000
    let shifted_l6 = _mm256_srli_si256::<6>(chars);
    // 00??????|ABCDEFGH 00KLMNOP|QRSTUVWX
    let mut shifted_r2 = _mm256_slli_si256::<2>(chars);
    // 00KLMNOP|00KLMNOP 00KLMNOP|00KLMNOP
    shifted_r2 = _mm256_permute4x64_epi64::<0xAA>(shifted_r2);
    // 00000000|00FFFFFF 00000000|00000000
    #[rustfmt::skip]
    let mask = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    );
    // 00000000|00KLMNOP 00000000|00000000
    shifted_r2 = _mm256_and_si256(shifted_r2, mask);

    // ABCDEFGH|IJKLMNOP QRSTUVWX|YZ000000
    chars = _mm256_or_si256(shifted_l6, shifted_r2);

    let mut chars_box: Box<[u8; 32]> = Box::new([0x00; 32]);

    _mm256_storeu_si256(chars_box.as_mut_ptr() as *mut __m256i, chars);

    String::from_raw_parts(
        Box::<[u8; 32]>::into_raw(chars_box) as *mut u8,
        ULID_LENGTH,
        32,
    )
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::ulid_encode::x86_64::*;

    static ULIDS: [&str; 7] = [
        "00000000000000000000000000",
        "0000000000ZZZZZZZZZZZZZZZZ",
        "0HHHHHHHHHHHHHHHHHHHHHHHHH",
        "0123456789ABCDEFGHJKMNPQRS",
        "7TVWXYZ0123456789ABCDEFGHJ",
        "7ZZZZZZZZZ0000000000000000",
        "7ZZZZZZZZZZZZZZZZZZZZZZZZZ",
    ];
    static U128S: [u128; 7] = [
        0x00000000000000000000000000000000,
        0x000000000000FFFFFFFFFFFFFFFFFFFF,
        0x118C6318C6318C6318C6318C6318C631,
        0x0110C8531D0952D8D73E1194E95B5F19,
        0xFADF3BEF8022190A63A12A5B1AE7C232,
        0xFFFFFFFFFFFF00000000000000000000,
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    ];

    #[test]
    fn test_u128_to_ascii_ssse3() {
        for (ulid_str, value) in zip(ULIDS, U128S) {
            unsafe {
                let actual = u128_to_ascii_ssse3(&value);
                assert_eq!(actual, *ulid_str);
            }
        }
    }

    #[test]
    fn test_u128_to_ascii_avx2() {
        for (ulid_str, value) in zip(ULIDS, U128S) {
            unsafe {
                let actual = u128_to_ascii_avx2(&value);
                assert_eq!(actual, *ulid_str);
            }
        }
    }
}
