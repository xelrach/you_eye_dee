/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
mod consts;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

use crate::{Ulid, ULID_LENGTH};
use std::arch::is_aarch64_feature_detected;
use std::sync::OnceLock;

static CROCKFORD_BASE32: [u8; 256] = include!("../../resources/crockford_base32_decode.txt");

static DECODE_ULID_FN: OnceLock<unsafe fn(input: &str) -> Result<Ulid, DecodeError>> =
    OnceLock::new();

#[derive(Debug, PartialEq, Eq)]
pub enum DecodeError {
    Unknown,
    WrongLength(usize),
    InvalidCharacter(usize),
}

pub fn string_to_ulid(input: &str) -> Result<Ulid, DecodeError> {
    if input.len() != ULID_LENGTH {
        return Err(DecodeError::WrongLength(input.len()));
    }

    let func = DECODE_ULID_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return x86_64::string_to_ulid_avx2;
            } else if is_x86_feature_detected!("ssse3") {
                return x86_64::string_to_ulid_ssse3;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                return aarch64::string_to_ulid_neon;
            }
        }

        string_to_ulid_scalar
    });

    unsafe { func(input) }
}

/**
 * Decodes a ULID string into a `Ulid`
 */
pub fn string_to_ulid_scalar(input: &str) -> Result<Ulid, DecodeError> {
    let mut result = 0u128;
    for (i, byte) in input.as_bytes().iter().enumerate() {
        let quint = CROCKFORD_BASE32[*byte as usize];
        if quint > 31 || (i == 0 && quint > 7) {
            return Err(DecodeError::InvalidCharacter(i));
        }

        result <<= 5;
        result |= quint as u128;
    }

    Ok(result.into())
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
    fn test_string_to_ulid_scalar() {
        for (ulid_str, expected) in zip(&ULIDS, &U128S) {
            let actual = string_to_ulid_scalar(ulid_str).unwrap();
            assert_eq!(
                <Ulid as Into<u128>>::into(actual),
                *expected,
                "Got: {:#X} Expected: {:#X}",
                <Ulid as Into<u128>>::into(actual),
                *expected
            );
        }
    }

    #[test]
    fn test_string_to_ulid_wrong_length() {
        let actual = string_to_ulid("").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(0));

        let actual = string_to_ulid("\0").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(1));

        let actual = string_to_ulid("!!!!!!!!!!!!!!!!!!!!!!!!!").unwrap_err();
        assert_eq!(actual, DecodeError::WrongLength(25));

        let actual = string_to_ulid("???????????????????????????").unwrap_err();
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
    fn test_string_to_ulid_scalar_invalid_char() {
        for (ulid_str, expected) in zip(&INVALID_ULIDS, &INVALID_CHAR_POSITION) {
            let actual = string_to_ulid_scalar(ulid_str).unwrap_err();
            assert_eq!(
                actual,
                DecodeError::InvalidCharacter(*expected),
                "Input: {}",
                ulid_str
            );
        }
    }
}
