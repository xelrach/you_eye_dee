/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::{Duration, SystemTime};
use rand::RngCore;
use crate::Ulid;
use crate::ulid_generate::UlidGenerationError;
use crate::ulid_generate::consts::*;


pub trait UlidGeneratorStateless {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError>;
    fn create_now(&self) -> Result<Ulid, UlidGenerationError>;
}

#[derive(Debug, Default, Clone)]
pub struct UlidGeneratorNanosecond {
}

impl UlidGeneratorStateless for UlidGeneratorNanosecond {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME_DURATION {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME_DURATION));
        }

        let mut ulid_bytes: [u8; ULID_LENGTH] = [0; ULID_LENGTH];

        rand::rng().fill_bytes(&mut ulid_bytes[0..8]);

        let subsec_nanos = unix_time.subsec_nanos();
        let nanos_without_millis = subsec_nanos % NANOS_IN_MILLIS;
        let nano_proportion = nanos_without_millis as f64 / NANOS_IN_MILLIS as f64;
        let nano_resized = (0xFFFF as f64 * nano_proportion) as u16;
        ulid_bytes[8..10].copy_from_slice(&nano_resized.to_le_bytes()[0..2]);

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

#[cfg(test)]
mod tests {
    use crate::ulid_generate::stateless::*;

    #[derive(Debug, Clone)]
    struct Param {
        unix_time: u64,
        expected_prefix: &'static str,
    }

    static PARAMS: [Param; 4] = [
        Param { unix_time: 0, expected_prefix: "0000000000" },
        Param { unix_time: 0, expected_prefix: "0000000000" },
        Param { unix_time: 0xFFFF_FFFFFFFF, expected_prefix: "7ZZZZZZZZZ" },
        Param { unix_time: 0xFFFF_FFFFFFFF, expected_prefix: "7ZZZZZZZZZ" },
    ];

    #[test]
    fn test_ulid_generator_nanosecond() {
        for param in PARAMS.clone() {
            let generator = UlidGeneratorNanosecond::default();
            let result = generator.create(Duration::from_millis(param.unix_time));
            match result {
                Ok(ulid) => {
                    assert_eq!(&ulid.to_string()[0..10], param.expected_prefix);
                }
                Err(e) => {panic!("Unexpected error: {:?}", e);}
            }
        }
    }

    #[test]
    fn test_ulid_generator_nanosecond_too_far_future() {
        let generator = UlidGeneratorNanosecond::default();
        let result = generator.create(Duration::from_millis(MAX_TIME_MS + 1));

        assert!(matches!(result, Err(UlidGenerationError::TooFarFuture(t)) if t == Duration::from_millis(1)));
    }
}
