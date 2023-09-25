use crate::{Ulid, ULID_LENGTH};
use std::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_loadu_si128, _mm_mulhi_epu16, _mm_mullo_epi16, _mm_or_si128,
    _mm_set1_epi16, _mm_set1_epi64x, _mm_setr_epi8, _mm_shuffle_epi8,
};
use std::ptr::copy_nonoverlapping;

static CROCKFORD_BASE32_ENCODE: [u8; 256] = include!("../resources/crockford_base32_encode.txt");

static mut ENCODE_ULID_FN: unsafe fn(ulid: &u128) -> String = u128_to_ascii_scalar;

pub fn ulid_to_string(input: &Ulid) -> String {
    unsafe { ENCODE_ULID_FN(&input.0) }
}

pub fn u128_to_ascii_scalar(ulid: &u128) -> String {
    let mut encoded = String::with_capacity(ULID_LENGTH);
    for i in 0..ULID_LENGTH {
        let mut shifted: usize = (ulid >> (125 - i * 5)) as usize;
        shifted &= 0x1F;

        let character = CROCKFORD_BASE32_ENCODE[shifted];
        encoded.push(character.into());
    }

    encoded
}

pub unsafe fn u128_to_ascii_scalar_unsafe(ulid: &u128) -> String {
    let mut chars: Box<[u8; ULID_LENGTH]> = Box::new([0x00; ULID_LENGTH]);
    for i in 0..ULID_LENGTH {
        let mut shifted: usize = (ulid >> (125 - i * 5)) as usize;
        shifted &= 0x1F;

        let character = CROCKFORD_BASE32_ENCODE.get_unchecked(shifted);
        *chars.get_unchecked_mut(i) = *character;
    }

    String::from_raw_parts(
        Box::<[u8; 26]>::into_raw(chars) as *mut u8,
        ULID_LENGTH,
        ULID_LENGTH,
    )
}

pub unsafe fn u128_to_ascii_ssse3(ulid: &u128) -> String {
    let bytes = ulid.to_le_bytes().as_ptr();
    // Lowest 10 bytes. This will be converted into 16 characters
    let low = _mm_loadu_si128(bytes as *const __m128i);
    // Highest 6 bytes. This will be converted into 10 characters
    // (last character is not from a full byte)
    #[rustfmt::skip]
    let high_mask = _mm_setr_epi8(
        -1, -1, -1, -1, -1, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    );
    let high = _mm_and_si128(
        _mm_loadu_si128(bytes.offset(10) as *const __m128i),
        high_mask,
    );

    let mut bytes: Box<[u8; ULID_LENGTH]> = Box::new([0x00; ULID_LENGTH]);

    encode_bytes_ssse3::<16>(low, bytes.as_mut_ptr().offset(10));
    encode_bytes_ssse3::<10>(high, bytes.as_mut_ptr());

    for i in 0..ULID_LENGTH {
        let byte_ref = bytes.get_unchecked_mut(i);
        let character = CROCKFORD_BASE32_ENCODE.get_unchecked(*byte_ref as usize);
        *byte_ref = *character;
    }

    String::from_raw_parts(
        Box::<[u8; ULID_LENGTH]>::into_raw(bytes) as *mut u8,
        ULID_LENGTH,
        ULID_LENGTH,
    )
}

#[inline(always)]
unsafe fn encode_bytes_ssse3<const T: isize>(bytes: __m128i, result: *mut u8) {
    // The bytes of the input are:
    // pppppooo|oonnnnnm|mmmmllll|lkkkkkjj|jjjiiiii hhhhhggg|ggfffffe|eeeedddd|dcccccbb|bbbaaaaa|????????|????????|????????|????????|????????|????????

    // Copy pairs of bytes and reverse the endianness.
    // Paris will be in LE order, the array in BE order.
    // ??????bb|bbbaaaaa|????dddd|dccccc??|??fffffe|eeee????|hhhhhggg|gg?????? ?????jj|jjjiiiii|????llll|lkkkkk??|??nnnnnm|mmmm????|pppppooo|oo??????
    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(
        0x08, 0x09, 0x07, 0x08, 0x06, 0x07, 0x05, 0x06,
        0x03, 0x04, 0x02, 0x03, 0x01, 0x02, 0x00, 0x01,
    );
    let duplicated_bytes = _mm_shuffle_epi8(bytes, shuffle);

    // Shift to the left: a by 8, c by 6, e by 4, g by 2
    let multiplicand = _mm_set1_epi64x(0x0100_0040_0010_0004);
    // 000aaaaa|00000000|000ccccc|00000000|000eeeee|00000000|000ggggg|00000000
    let odds = _mm_and_si128(
        _mm_mullo_epi16(duplicated_bytes, multiplicand),
        _mm_set1_epi16(0x1F00),
    );
    // Shift to the right: b by 5, d by 7, f by 9, h by 11
    let multiplicand = _mm_set1_epi64x(0x0800_0200_0080_0020);
    // 00000000|000bbbbb|00000000|000ddddd|00000000|000fffff|00000000|000hhhhh
    let evens = _mm_and_si128(
        _mm_mulhi_epu16(duplicated_bytes, multiplicand),
        _mm_set1_epi16(0x001F),
    );

    // 000aaaaa|000bbbbb|000ccccc|000ddddd|000eeeee|000fffff|000ggggg|000hhhhh
    let quints = _mm_or_si128(odds, evens);
    let quints_ptr = &quints as *const __m128i as *const u8;

    // For the low bytes, we copy all 16 characters.
    // For the high bytes, we copy the last 10 characters in the array
    copy_nonoverlapping::<u8>(quints_ptr.offset(16 - T), result, T as usize);
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use crate::ulid_encode::*;

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
    fn test_u128_to_ascii_scalar() {
        for (ulid_str, value) in zip(&ULIDS, &U128S) {
            let actual = u128_to_ascii_scalar(value);
            assert_eq!(actual, *ulid_str);
        }
    }

    #[test]
    fn test_u128_to_ascii_scalar_unsafe() {
        for (ulid_str, value) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = u128_to_ascii_scalar_unsafe(value);
                assert_eq!(actual, *ulid_str);
            }
        }
    }

    #[test]
    fn test_u128_to_ascii_ssse3() {
        for (ulid_str, value) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = u128_to_ascii_ssse3(value);
                assert_eq!(actual, *ulid_str);
            }
        }
    }
}
