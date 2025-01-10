use crate::ulid_decode::consts::*;
use crate::ulid_decode::DecodeError;
use crate::Ulid;
use std::arch::aarch64::{
    uint8x16_t, vandq_u16, vandq_u32, vandq_u64, vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8,
    vdupq_n_u16, vdupq_n_u32, vdupq_n_u64, vdupq_n_u8, vget_lane_u64, vld1q_u8, vmvnq_u8,
    vorrq_u16, vorrq_u32, vorrq_u64, vorrq_u8, vqtbl1q_u8, vreinterpret_u64_u8,
    vreinterpretq_s8_u8, vreinterpretq_u16_u8, vreinterpretq_u32_u16, vreinterpretq_u64_u32,
    vreinterpretq_u8_s8, vreinterpretq_u8_u64, vshlq_n_u16, vshlq_n_u32, vshlq_n_u64, vshlq_n_u8,
    vshrn_n_u16, vst1q_u8, vsubq_s8,
};

pub static FINAL_BYTES_SHIFT: [u8; 16] = [
    0x03, 0x04, 0x05, 0x06, 0x07, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
];

/*
#[target_feature(enable = "neon")]
pub unsafe fn string_to_ulid_neon(input: &str) -> Result<Ulid, DecodeError> {
    let input = vld1_u8_x2(input.as_ptr());

    let field_a = lookup(input.val[0], error_a);
    let field_b = lookup(input.val[1], error_b);
    let field_c = lookup(input.val[2], error_c);
    let field_d = lookup(input.val[3], error_d);

    let mut result: uint8x8x3_t;
    result.val[0] = vorr_u8(vshr_n_u8(field_b, 4), vshl_n_u8(field_a, 2));
    result.val[1] = vorr_u8(vshr_n_u8(field_c, 2), vshl_n_u8(field_b, 4));
    result.val[2] = vorr_u8(field_d, vshl_n_u8(field_c, 6));
}

unsafe fn lookup_pshufb_bitmask(input: uint8x8_t) -> uint8x8_t {

const uint8x8_t higher_nibble = vshr_n_u8(input, 4);
const uint8x8_t lower_nibble = vand_u8(input, packed_byte(0x0f));

const uint8x8x2_t shiftLUT = {
0, 0, 19, 4, uint8_t(- 65), uint8_t( - 65), uint8_t( - 71), uint8_t( - 71),
0, 0, 0, 0,   0, 0, 0, 0};

const uint8x8x2_t maskLUT = {
/* 0        : 0b1010_1000*/ 0xa8,
/* 1 .. 9   : 0b1111_1000*/ 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8,
/* 10       : 0b1111_0000*/ 0xf0,
/* 11       : 0b0101_0100*/ 0x54,
/* 12 .. 14 : 0b0101_0000*/ 0x50, 0x50, 0x50,
/* 15       : 0b0101_0100*/ 0x54
};

const uint8x8x2_t bitposLUT = {
0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

const uint8x8_t sh = vtbl2_u8(shiftLUT, higher_nibble);
const uint8x8_t eq_2f = vceq_u8(input, packed_byte(0x2f));
const uint8x8_t shift = vbsl_u8(eq_2f, packed_byte(16), sh);

const uint8x8_t M = vtbl2_u8(maskLUT,   lower_nibble);
const uint8x8_t bit = vtbl2_u8(bitposLUT, higher_nibble);

error = vceq_u8(vand_u8(M, bit), packed_byte(0));

const uint8x8_t result = vadd_u8(input, shift);

return result;
}
 */

/**
 * Decodes a ULID string into a `Ulid`
 * # Safety
 * Code uses raw pointers and x86_64 intrinsics. No safety requirements for caller.
 */
#[target_feature(enable = "neon")]
pub unsafe fn string_to_ulid_neon(input: &str) -> Result<Ulid, DecodeError> {
    // Convert from the string into an array of bytes
    let mut high = encoding_lookup(input.as_bytes().as_ptr().offset(-6));
    // Zero the extra six bytes at the start of the array
    #[rustfmt::skip]
    let actual_bytes_mask = vld1q_u8(MASK_LAST_TEN_BYTES.as_ptr());
    high = vandq_u8(high, actual_bytes_mask);

    let low = encoding_lookup(input.as_bytes().as_ptr().offset(10));

    // Check that all the characters were valid
    let mask_max_high = vld1q_u8(ULID_INVALID_MASK.as_ptr());
    let invalid_high = vandq_u8(high, mask_max_high);
    if !is_all_zeros(invalid_high) {
        return Err(DecodeError::InvalidCharacter(1));
    }

    let mask_max_low = vld1q_u8((ULID_INVALID_MASK.as_ptr()).offset(16));
    let invalid_low = vandq_u8(low, mask_max_low);
    if !is_all_zeros(invalid_low) {
        return Err(DecodeError::InvalidCharacter(1));
    }

    // Shift and rearrange the bits into an array
    let mut be_bytes: [u8; 20] = [0; 20];
    let high_bytes = shift_bits(high);
    vst1q_u8(be_bytes.as_mut_ptr(), high_bytes);
    let low_bytes = shift_bits(low);
    vst1q_u8(be_bytes.as_mut_ptr().offset(6), low_bytes);

    Ok(u128::from_be_bytes(be_bytes[0..16].try_into().unwrap()).into())
}

#[target_feature(enable = "neon")]
unsafe fn is_all_zeros(value: uint8x16_t) -> bool {
    let equal_mask = vreinterpretq_u16_u8(vceqq_u8(value, vdupq_n_u8(0x00)));
    let result = vshrn_n_u16::<4>(equal_mask);
    vget_lane_u64::<0>(vreinterpret_u64_u8(result)) == u64::MAX
}

#[target_feature(enable = "neon")]
unsafe fn encoding_lookup(ulid_str_ptr: *const u8) -> uint8x16_t {
    let encoded_bytes = vld1q_u8(ulid_str_ptr);

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

/**
 * Shifts bits from the 5-bit representation into normal bytes
 * # Safety
 * `result` MUST have 10 bytes of space
 */
#[target_feature(enable = "neon")]
unsafe fn shift_bits(value: uint8x16_t) -> uint8x16_t {
    // Lower eight bytes (64 bits) of the value is:
    // 000ABCDE|000FGHIJ|000KLMNO|000PQRST|000UVWXY|000Zabcd|000efghi|000jklmn

    // ABCDE000|FGHIJ000|KLMNO000|PQRST000|UVWXY000|Zabcd000|efghi000|jklmn000
    let left_8_3 = vshlq_n_u8::<3>(value);
    // ABCDE000|00000000|KLMNO000|00000000|UVWXY000|00000000|efghi000|00000000
    let left_8_3_maked = vandq_u16(vreinterpretq_u16_u8(left_8_3), vdupq_n_u16(0xF8_00));

    // DE000FGH|IJ000000|NO000PQR|ST000000|XY000Zab|cd000000|hi000jkl|mn000000
    let left_16_6 = vshlq_n_u16::<6>(vreinterpretq_u16_u8(value));
    // 00000FGH|IJ000000|00000PQR|ST000000|00000Zab|cd000000|00000jkl|mn000000
    let left_16_6_maked = vandq_u16(left_16_6, vdupq_n_u16(0x07_C0));

    // ABCDEFGH|IJ000000|KLMNOPQR|ST000000|UVWXYZab|cd000000|efghijkl|mn000000
    let two_chars = vreinterpretq_u32_u16(vorrq_u16(left_8_3_maked, left_16_6_maked));
    // ABCDEFGH|IJ000000|00000000|00000000|UVWXYZab|cd000000|00000000|00000000
    let two_chars_masked = vandq_u32(two_chars, vdupq_n_u32(0xFF_C0_00_00));

    // GHIJ0000|00KLMNOP|QRST0000|00000000|abcd0000|00efghij|klmn0000|00000000
    let left_32_6 = vshlq_n_u32::<6>(two_chars);
    // 00000000|00KLMNOP|QRST0000|00000000|00000000|00efghij|klmn0000|00000000
    let left_32_6_masked = vandq_u32(left_32_6, vdupq_n_u32(0x00_3F_F0_00));

    // ABCDEFGH|IJKLMNOP|QRST0000|00000000|UVWXYZab|cdefghij|klmn0000|00000000
    let four_chars = vreinterpretq_u64_u32(vorrq_u32(two_chars_masked, left_32_6_masked));
    // ABCDEFGH|IJKLMNOP|QRST0000|00000000|00000000|00000000|00000000|00000000
    let four_chars_masked = vandq_u64(four_chars, vdupq_n_u64(0xFF_FF_F0_00_00_00_00_00));

    // MNOPQRST|00000000|0000UVWX|YZabcdef|ghijklmn|00000000|00000000|00000000
    let left_64_12 = vshlq_n_u64::<12>(four_chars);
    // 00000000|00000000|0000UVWX|YZabcdef|ghijklmn|00000000|00000000|00000000
    let left_64_12_masked = vandq_u64(left_64_12, vdupq_n_u64(0x00_00_0F_FF_FF_00_00_00));

    // ABCDEFGH|IJKLMNOP|QRSTUVWX|YZabcdef|ghijklmn|00000000|00000000|00000000
    let shifted_bits = vorrq_u64(four_chars_masked, left_64_12_masked);

    // The array is now to u64 (little-endian) in big-endian order
    let bytes5x2 = vreinterpretq_u8_u64(shifted_bits);

    vqtbl1q_u8(bytes5x2, vld1q_u8(FINAL_BYTES_SHIFT.as_ptr()))
}

#[cfg(test)]
mod tests {
    use crate::ulid_decode::aarch64::string_to_ulid_neon;
    use crate::ulid_decode::DecodeError;
    use crate::Ulid;
    use std::iter::zip;

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
