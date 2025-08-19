/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
use crate::ulid_generate::consts::*;
use crate::ulid_generate::UlidGenerationError;
use crate::Ulid;
#[cfg(feature = "default_rng")]
use rand::prelude::StdRng;
use rand::{RngCore, SeedableRng};
use std::cell::RefCell;
use std::time::{Duration, SystemTime};

pub trait UlidGeneratorStateless {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError>;
    fn create_now(&self) -> Result<Ulid, UlidGenerationError>;
}

/**
 * Creates ULIDs using nanosecond precision instead of millisecond precision. The technique is the
 * same as in RFC-9562 Section 6.2 Method 3
 */
#[derive(Debug, Clone)]
pub struct UlidGeneratorNanosecond<T: RngCore> {
    rng_cell: RefCell<T>
}

impl<T: RngCore> UlidGeneratorNanosecond<T> {
    fn new(rng: T) -> Self {
        Self {
            rng_cell: RefCell::new(rng)
        }
    }
}

impl<T: RngCore> UlidGeneratorStateless for UlidGeneratorNanosecond<T> {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME_DURATION {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME_DURATION));
        }

        let mut ulid_bytes: [u8; ULID_LENGTH] = [0; ULID_LENGTH];

        self.rng_cell.borrow_mut().fill_bytes(&mut ulid_bytes[0..8]);

        let subsec_nanos = unix_time.subsec_nanos();
        let nanos_without_millis = subsec_nanos % NANOS_IN_MILLI;
        let nano_proportion = nanos_without_millis as f64 / (NANOS_IN_MILLI - 1) as f64;
        let nano_scaled = (u16::MAX as f64 * nano_proportion) as u16;
        ulid_bytes[8..10].copy_from_slice(&nano_scaled.to_le_bytes()[0..2]);

        let time_ms = unix_time.as_millis() as u64;
        ulid_bytes[10..16].copy_from_slice(&time_ms.to_le_bytes()[0..6]);

        let ulid = Ulid::from_le_bytes(ulid_bytes);

        Ok(ulid)
    }

    fn create_now(&self) -> Result<Ulid, UlidGenerationError> {
        let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
        match unix_time {
            Ok(unix_time) => self.create(unix_time),
            Err(time_err) => Err(UlidGenerationError::TooFarPast(time_err.duration())),
        }
    }
}

#[cfg(feature = "default_rng")]
impl Default for UlidGeneratorNanosecond<StdRng> {
    fn default() -> Self {
        Self {
            rng_cell: RefCell::new(StdRng::from_os_rng()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ulid_generate::stateless::*;

    struct FixedRng {
        byte: u8
    }

    impl RngCore for FixedRng {
        fn next_u32(&mut self) -> u32 {
            panic!();
        }

        fn next_u64(&mut self) -> u64 {
            panic!();
        }

        fn fill_bytes(&mut self, dst: &mut [u8]) {
            dst.fill(self.byte);
        }
    }

    #[derive(Debug, Clone)]
    struct Param {
        rng_byte: u8,
        unix_time: Duration,
        expected: &'static str,
    }

    static PARAMS: [Param; 4] = [
        Param { rng_byte: 0, unix_time: Duration::from_secs(0), expected: "00000000000000000000000000" },
        Param { rng_byte: 0xFF, unix_time: Duration::from_secs(0), expected: "0000000000000FZZZZZZZZZZZZ" },
        Param { rng_byte: 0, unix_time: MAX_TIME_DURATION, expected: "7ZZZZZZZZZZZZG000000000000" },
        Param { rng_byte: 0xFF, unix_time: MAX_TIME_DURATION, expected: "7ZZZZZZZZZZZZZZZZZZZZZZZZZ" },
    ];

    #[test]
    fn test_ulid_generator_nanosecond() {
        for param in PARAMS.clone() {
            let generator = UlidGeneratorNanosecond::new(FixedRng { byte: param.rng_byte });
            let result = generator.create(param.unix_time);
            match result {
                Ok(ulid) => {
                    assert_eq!(ulid.to_string(), param.expected);
                }
                Err(e) => {panic!("Unexpected error: {:?}", e);}
            }
        }
    }

    #[test]
    fn test_ulid_generator_nanosecond_too_far_future() {
        let generator = UlidGeneratorNanosecond::default();
        let result = generator.create(MAX_TIME_DURATION + Duration::from_millis(1));

        assert!(matches!(result, Err(UlidGenerationError::TooFarFuture(t)) if t == Duration::from_millis(1)));
    }

    fn require_send<T: Send>(_t: &T) {}
    #[test]
    fn test_ulid_generator_nanosecond_is_send() {
        require_send(&UlidGeneratorNanosecond::default());
    }
}
