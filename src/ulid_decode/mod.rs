#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

use crate::ULID_LENGTH;

static CROCKFORD_BASE32: [u8; 256] = include!("../../resources/crockford_base32_decode.txt");
// Bits set to 1 in this mask are invalid in BASE32. Doing a bitwise and with this mask will find
// invalid inputs
static ULID_INVALID_MASK: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF8, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
    0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
];

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    Unknown,
    WrongLength(usize),
    InvalidCharacter(usize),
}

#[cfg(not(any(target_feature = "sssse3", target_feature = "avx2", target_feature = "neon")))]
pub fn ulid_to_u128(input: &str) -> Result<u128, DecodeError> {
    if input.len() != ULID_LENGTH {
        return Err(DecodeError::WrongLength(input.len()));
    }

    unsafe { ulid_to_u128_scalar_unsafe(input) }
}

#[cfg(all(target_feature = "sssse3", not(target_feature = "avx2")))]
pub fn ulid_to_u128(input: &str) -> Result<u128, DecodeError> {
    if input.len() != ULID_LENGTH {
        return Err(DecodeError::WrongLength(input.len()));
    }

    unsafe { x86_64::ulid_to_u128_ssse3(input) }
}

#[cfg(target_feature = "avx2")]
pub fn ulid_to_u128(input: &str) -> Result<u128, DecodeError> {
    if input.len() != ULID_LENGTH {
        return Err(DecodeError::WrongLength(input.len()));
    }

    unsafe { x86_64::ulid_to_u128_avx2(input) }
}

#[cfg(target_feature = "neon")]
pub fn ulid_to_u128(input: &str) -> Result<u128, DecodeError> {
    if input.len() != ULID_LENGTH {
        return Err(DecodeError::WrongLength(input.len()));
    }

    unsafe { aarch64::ulid_to_u128_neon(input) }
}

/**
 * Decodes a ULID string into a `u128`
 */
pub fn ulid_to_u128_scalar(input: &str) -> Result<u128, DecodeError> {
    let mut result = 0u128;
    for (i, byte) in input.as_bytes().iter().enumerate() {
        let byte = *byte;
        // Assert is always true and thus is removed from actual code.
        // The assert skips bounds checking on the array.
        assert!(byte <= 255);
        let quint = CROCKFORD_BASE32[byte as usize];
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
        result |= *quint as u128;
    }

    Ok(result)
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
}
