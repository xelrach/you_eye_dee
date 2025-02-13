/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use crate::ulid_decode::consts::*;
use crate::ulid_decode::DecodeError;
use crate::Ulid;
use aligned::{Aligned, A16};
use std::arch::aarch64::{
    uint8x16_t, vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8, vdupq_n_u8, vget_lane_u64, vld1q_s8,
    vld1q_u8, vmvnq_u8, vorrq_u8, vqtbl1q_u8, vreinterpret_u64_u8, vreinterpretq_s8_u8,
    vreinterpretq_u16_u8, vreinterpretq_u8_s8, vshlq_u8, vshrn_n_u16, vst1q_u8, vsubq_s8,
};

pub static FINAL_BYTES_SHIFT: [u8; 16] = [
    0x03, 0x04, 0x05, 0x06, 0x07, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

static STRIP_UNUSED: Aligned<A16, [u8; 16]> = Aligned([
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
]);

/**
 * Decodes a ULID string into a `Ulid`
 * # Safety
 * Code uses aarch64 intrinsics. No safety requirements for caller.
 */
#[target_feature(enable = "neon")]
pub unsafe fn string_to_ulid_neon(input: &str) -> Result<Ulid, DecodeError> {
    let high_chars = vld1q_u8(input.as_bytes().as_ptr().offset(-6));
    let low_chars = vld1q_u8(input.as_bytes().as_ptr().offset(10));

    let mut left_chars_decoded = encoding_lookup(high_chars);
    // Remove the extra 6 bytes
    left_chars_decoded = vandq_u8(left_chars_decoded, vld1q_u8(STRIP_UNUSED.as_ptr()));
    // 16 characters
    let right_chars_decoded = encoding_lookup(low_chars);

    // Check that all the characters were valid
    let invalid_left = vandq_u8(left_chars_decoded, vld1q_u8(ULID_INVALID_MASK.as_ptr()));
    if !is_all_zeros(invalid_left) {
        return Err(DecodeError::InvalidCharacter(1));
    }

    let invalid_right = vandq_u8(
        right_chars_decoded,
        vld1q_u8(ULID_INVALID_MASK.as_ptr().offset(16)),
    );
    if !is_all_zeros(invalid_right) {
        return Err(DecodeError::InvalidCharacter(1));
    }

    // Shift and rearrange the bits into an array
    let mut le_bytes: [u8; 26] = [0; 26];
    // 000000FEDCBA0000
    let high_bytes_shifted = shift_bits(left_chars_decoded);
    // 0000000000000000FEDCBA0000
    vst1q_u8(le_bytes.as_mut_ptr().offset(10), high_bytes_shifted);
    // 000000PONMLKJIHG
    let low_bytes_shifted = shift_bits(right_chars_decoded);
    // 000000PONMLKJIHGFEDCBA0000
    vst1q_u8(le_bytes.as_mut_ptr(), low_bytes_shifted);

    Ok(u128::from_le_bytes(le_bytes[6..22].try_into().unwrap()).into())
}

#[target_feature(enable = "neon")]
unsafe fn is_all_zeros(value: uint8x16_t) -> bool {
    let equal_mask = vreinterpretq_u16_u8(vceqq_u8(value, vdupq_n_u8(0x00)));
    let result = vshrn_n_u16::<4>(equal_mask);
    vget_lane_u64::<0>(vreinterpret_u64_u8(result)) == u64::MAX
}

#[target_feature(enable = "neon")]
unsafe fn encoding_lookup(encoded_bytes: uint8x16_t) -> uint8x16_t {
    let mut result = vdupq_n_u8(0xFF);

    // Decode digits
    let mut decoded_digits = vreinterpretq_u8_s8(vsubq_s8(
        vreinterpretq_s8_u8(encoded_bytes),
        vreinterpretq_s8_u8(vdupq_n_u8(CHAR_0)),
    ));
    // Character is a digit
    let char_is_digit = vandq_u8(
        vcgeq_u8(encoded_bytes, vdupq_n_u8(CHAR_0)),
        vcleq_u8(encoded_bytes, vdupq_n_u8(CHAR_9)),
    );
    decoded_digits = vandq_u8(char_is_digit, decoded_digits);

    let not_digits = vandq_u8(vmvnq_u8(char_is_digit), result);
    result = vorrq_u8(decoded_digits, not_digits);

    let low_nibble = vandq_u8(encoded_bytes, vdupq_n_u8(0x0F));
    // Decode AO and ao range
    let lookup_ao = vld1q_u8(AO_TABLE.as_ptr());
    let mut decoded_ao = vqtbl1q_u8(lookup_ao, low_nibble);
    // Character is in [AO] or [ao]
    let char_in_ao = vorrq_u8(
        vandq_u8(
            vcgeq_u8(encoded_bytes, vdupq_n_u8(CHAR_UPPER_A)),
            vcleq_u8(encoded_bytes, vdupq_n_u8(CHAR_UPPER_O)),
        ),
        vandq_u8(
            vcgeq_u8(encoded_bytes, vdupq_n_u8(CHAR_LOWER_A)),
            vcleq_u8(encoded_bytes, vdupq_n_u8(CHAR_LOWER_O)),
        ),
    );
    decoded_ao = vandq_u8(char_in_ao, decoded_ao);
    let not_in_ao = vandq_u8(vmvnq_u8(char_in_ao), result);
    result = vorrq_u8(decoded_ao, not_in_ao);

    // Decode PZ and pz range
    let lookup_pz = vld1q_u8(PZ_TABLE.as_ptr());
    let mut decoded_pz = vqtbl1q_u8(lookup_pz, low_nibble);
    // Character is in [PZ] or [pz]
    let char_in_pz = vorrq_u8(
        vandq_u8(
            vcgeq_u8(encoded_bytes, vdupq_n_u8(CHAR_UPPER_P)),
            vcleq_u8(encoded_bytes, vdupq_n_u8(CHAR_UPPER_Z)),
        ),
        vandq_u8(
            vcgeq_u8(encoded_bytes, vdupq_n_u8(CHAR_LOWER_P)),
            vcleq_u8(encoded_bytes, vdupq_n_u8(CHAR_LOWER_Z)),
        ),
    );
    decoded_pz = vandq_u8(char_in_pz, decoded_pz);
    let not_in_pz = vandq_u8(vmvnq_u8(char_in_pz), result);
    result = vorrq_u8(decoded_pz, not_in_pz);

    result
}

static SHIFT_1: Aligned<A16, [i8; 16]> = Aligned([3, 6, 1, 4, 7, 2, 5, 0, 3, 6, 1, 4, 7, 2, 5, 0]);

static SHIFT_2: Aligned<A16, [i8; 16]> =
    Aligned([0, -2, 0, -4, -1, 0, -3, 0, 0, -2, 0, -4, -1, 0, -3, 0]);

static LOOKUP_1: Aligned<A16, [u8; 16]> = Aligned([
    0xFF, 0xFF, 0xFF, 0x06, 0x04, 0x03, 0x01, 0x00, 0xFF, 0xFF, 0xFF, 0x0E, 0x0C, 0x0B, 0x09, 0x08,
]);

static LOOKUP_2: Aligned<A16, [u8; 16]> = Aligned([
    0xFF, 0xFF, 0xFF, 0x07, 0x05, 0xFF, 0x02, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F, 0x0D, 0xFF, 0x0A, 0xFF,
]);

static LOOKUP_3: Aligned<A16, [u8; 16]> = Aligned([
    0xFF, 0xFF, 0xFF, 0xFF, 0x06, 0x04, 0x03, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x0E, 0x0C, 0x0B, 0x09,
]);

static SHIFT_64: Aligned<A16, [u8; 16]> = Aligned([
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x03, 0x04, 0x05, 0x06, 0x07,
]);

/**
 * Shifts bits from the 5-bit representation into normal bytes.
 * Returns an array with six 0 bytes followed by ten bytes of data
 * # Safety
 * No restrictions on caller
 */
#[target_feature(enable = "neon")]
unsafe fn shift_bits(value: uint8x16_t) -> uint8x16_t {
    // Eight bytes of the value is:
    // 000ABCDE|000FGHIJ|000KLMNO|000PQRST|000UVWXY|000Zabcd|000efghi|000jklmn

    //     3   |    6   |    1   |    4   |    7   |    2   |    5   |    0
    // ABCDE000|IJ000000|00KLMNO0|QRST0000|Y0000000|0Zabcd00|ghi00000|000jklmn
    let shift_1 = vshlq_u8(value, vld1q_s8(SHIFT_1.as_ptr()));
    // 00000000|00000000|00000000|ghi00000|Y0000000|QRST0000|IJ000000|ABCDE000
    let lookup_1 = vqtbl1q_u8(shift_1, vld1q_u8(LOOKUP_1.as_ptr()));
    // 00000000|00000000|00000000|000jklmn|0Zabcd00|00000000|00KLMNO0|00000000
    let lookup_2 = vqtbl1q_u8(shift_1, vld1q_u8(LOOKUP_2.as_ptr()));

    //     0   |   -2   |    0   |   -4   |   -1   |    0   |   -3   |    0
    // 000ABCDE|00000FGH|000KLMNO|0000000P|0000UVWX|000Zabcd|000000ef|000jklmn
    let shift_2 = vshlq_u8(value, vld1q_s8(SHIFT_2.as_ptr()));
    // 00000000|00000000|00000000|00000000|000000ef|0000UVWX|0000000P|00000FGH
    let lookup_3 = vqtbl1q_u8(shift_2, vld1q_u8(LOOKUP_3.as_ptr()));

    // 00000000|00000000|00000000|ghijklmn|YZabcdef|QRSTUVWX|IJKLMNOP|ABCDEFGH
    let sixtyfourx2 = vorrq_u8(vorrq_u8(lookup_1, lookup_2), lookup_3);

    // Six 0 bytes followed by ten bytes of data
    vqtbl1q_u8(sixtyfourx2, vld1q_u8(SHIFT_64.as_ptr()))
}

#[cfg(test)]
mod tests {
    use crate::ulid_decode::aarch64::string_to_ulid_neon;
    use crate::ulid_decode::DecodeError;
    use crate::Ulid;
    use std::iter::zip;

    static ULIDS: [&str; 10] = [
        "00000000000000000000000000",
        "0000000000ZZZZZZzzzzzzzzZZ",
        "01081G81860W40J2GB1G6GW3RG",
        "0123456789ABCDEFGHJKMNPQRS",
        "7TVWXYZ0123456789ABCDEFGHJ",
        "0123456789abcdefghjkmnpqrs",
        "7tvwxyz0123456789abcdefghj",
        "1IiLl0Oo1IiLl0Oo1IiLl0Oo1I",
        "7ZZZZZZZZZ0000000000000000",
        "7ZZZZZZZZZZZZZZZZZZZZZZZZZ",
    ];

    static U128S: [u128; 10] = [
        0x00000000000000000000000000000000,
        0x000000000000FFFFFFFFFFFFFFFFFFFF,
        0x0102030405060708090A0B0C0D0E0F10,
        0x0110C8531D0952D8D73E1194E95B5F19,
        0xFADF3BEF8022190A63A12A5B1AE7C232,
        0x0110C8531D0952D8D73E1194E95B5F19,
        0xFADF3BEF8022190A63A12A5B1AE7C232,
        0x21084200002108420000210842000021,
        0xFFFFFFFFFFFF00000000000000000000,
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    ];

    #[test]
    fn test_string_to_ulid_neon() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            unsafe {
                let actual = string_to_ulid_neon(ulid_str).unwrap();
                assert_eq!(
                    <Ulid as Into<u128>>::into(actual),
                    *expected,
                    "Got: {:#X} Expected: {:#X}",
                    <Ulid as Into<u128>>::into(actual),
                    *expected
                );
            }
        }
    }

    static INVALID_ULIDS: [&str; 11] = [
        "8ZZZZZZZZZZZZZZZZZZZZZZZZZ",
        "7:ZZZZZZZZZZZZZZZZZZZZZZZZ",
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

    static INVALID_CHAR_POSITION: [usize; 11] = [0, 1, 0, 25, 25, 25, 25, 0, 0, 25, 25];

    #[test]
    fn test_string_to_ulid_neon_invalid_char() {
        unsafe {
            for (ulid_str, expected) in zip(&INVALID_ULIDS, &INVALID_CHAR_POSITION) {
                let actual = string_to_ulid_neon(ulid_str).unwrap_err();
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
