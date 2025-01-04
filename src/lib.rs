use std::fmt::{Debug, Display, Formatter};

use crate::ulid_decode::{string_to_ulid, DecodeError};
use crate::ulid_encode::ulid_to_string;

pub mod ulid_decode;
pub mod ulid_encode;
mod ulid_generate;

const ULID_LENGTH: usize = 26;

#[repr(transparent)]
#[derive(Default, Eq, Ord, PartialEq, PartialOrd, Copy, Clone)]
pub struct Ulid(u128);

impl Ulid {
    const RANDOM_MASK: u128 = 0x000000000000FFFFFFFFFFFFFFFFFFFF;

    pub fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Ulid(u128::from_le_bytes(bytes))
    }

    fn get_time_ms(&self) -> u64 {
        (self.0 >> 10) as u64
    }

    fn get_random(&self) -> u128 {
        self.0 & Ulid::RANDOM_MASK
    }

    fn from_ms_and_random(time_ms: u64, random: u128) -> Ulid {
        Ulid((random & Ulid::RANDOM_MASK) | ((time_ms as u128) << 10))
    }

    pub fn to_le_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    pub fn to_be_bytes(self) -> [u8; 16] {
        self.0.to_be_bytes()
    }

    pub fn encode(&self) -> String {
        ulid_to_string(self)
    }

    pub fn decode(string: &str) -> Result<Self, DecodeError> {
        string_to_ulid(string)
    }
}

impl Display for Ulid {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.encode())
    }
}

impl Debug for Ulid {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:#X})", self.encode(), self.0)
    }
}

impl From<&Ulid> for u128 {
    fn from(value: &Ulid) -> Self {
        value.0
    }
}

impl From<Ulid> for u128 {
    fn from(value: Ulid) -> Self {
        value.0
    }
}

impl From<u128> for Ulid {
    fn from(value: u128) -> Self {
        Ulid(value)
    }
}

impl TryFrom<&str> for Ulid {
    type Error = DecodeError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        string_to_ulid(value)
    }
}

impl From<&Ulid> for String {
    fn from(value: &Ulid) -> Self {
        ulid_to_string(value)
    }
}
