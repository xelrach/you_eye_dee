#[cfg(target_arch = "x86_64")]
pub mod x86_64;

use std::sync::OnceLock;
use crate::{Ulid, ULID_LENGTH};

static CROCKFORD_BASE32_ENCODE: [u8; 256] = include!("../../resources/crockford_base32_encode.txt");

static ENCODE_ULID_FN: OnceLock<unsafe fn(ulid: &u128) -> String> = OnceLock::new();

pub fn ulid_to_string(input: &Ulid) -> String {
    let func = ENCODE_ULID_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")] {
            if is_x86_feature_detected!("avx2") {
                return x86_64::u128_to_ascii_avx2;
            } else if is_x86_feature_detected!("ssse3") {
                return x86_64::u128_to_ascii_ssse3;
            }
        }

        u128_to_ascii_scalar_unsafe
    });

    unsafe { func(&input.0) }
}

/**
 * Encodes a `u128` into a ULID string
 */
pub fn u128_to_ascii_scalar(ulid: &u128) -> String {
    let mut encoded = String::with_capacity(ULID_LENGTH);
    for i in 0..ULID_LENGTH {
        let mut shifted: usize = (ulid >> (125 - i * 5)) as usize;
        shifted &= 0x1F;

        // Assert is always true and thus is removed from actual code.
        // The assert skips bounds checking on the array.
        assert!(shifted <= 0x1F);
        let character = CROCKFORD_BASE32_ENCODE[shifted];
        encoded.push(character.into());
    }

    encoded
}

/**
 * Encodes a `u128` into a ULID string
 * # Safety
 * Code uses raw pointers and `get_unchecked`. No safety requirements for caller.
 */
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
}
