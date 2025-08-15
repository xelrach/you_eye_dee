/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::{Duration, SystemTime};
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

use crate::Ulid;
use crate::ulid_generate::UlidGenerationError;
use crate::ulid_generate::consts::*;

const MAX_RANDOMNESS_VALUE: u128 = 0xFFFF_FFFFFFFF_FFFFFFFF;

/**
 * A ULID generator that fully complies with thr ULID spec: https://github.com/ulid/spec. The spec
 * requires that all ULIDs generated are monotonically increasing. When multiple ULIDs are created
 * within the same millisecond, the first has its "randomness" section generated randomly. The
 * subsequent ULIDs will have its "randomness" section incremented from the previous by one.
 */
pub trait UlidGenerator {
    fn create(&mut self, unix_time: Duration) -> Result<Ulid, UlidGenerationError>;
    fn create_now(&mut self) -> Result<Ulid, UlidGenerationError>;
}

/**
 * Complies with ULID spec by storing state which ensures that all ULIDs are monotonically
 * increasing.
 */
#[derive(Debug)]
pub struct UlidGeneratorMonotonic<T: RngCore> {
    time_ms: u64,
    random: u128,
    rng: T,
}

impl<T: RngCore> UlidGenerator for UlidGeneratorMonotonic<T> {
    fn create(&mut self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME_DURATION {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME_DURATION));
        }

        let time_ms = unix_time.as_millis() as u64;
        if self.time_ms >= time_ms {
            let random = self.random + 1;
            if random > MAX_RANDOMNESS_VALUE {
                return Err(UlidGenerationError::CannotGenerateMonotonic(unix_time));
            }
            self.time_ms = time_ms;
            self.random = random;

            Ok(Ulid::from_ms_and_random(time_ms, random))
        } else {
            let mut ulid_bytes: [u8; ULID_LENGTH] = [0; ULID_LENGTH];
            self.rng.fill_bytes(&mut ulid_bytes[0..10]);
            ulid_bytes[10..16].copy_from_slice(&time_ms.to_le_bytes()[0..6]);

            let ulid = Ulid::from_le_bytes(ulid_bytes);
            self.time_ms = ulid.get_time_ms();
            self.random = ulid.get_random();

            Ok(ulid)
        }
    }

    fn create_now(&mut self) -> Result<Ulid, UlidGenerationError> {
        let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
        match unix_time {
            Ok(unix_time) => self.create(unix_time),
            Err(time_err) => Err(UlidGenerationError::TooFarPast(time_err.duration())),
        }
    }
}

impl<T: RngCore + Clone> Clone for UlidGeneratorMonotonic<T> {
    fn clone(&self) -> Self {
        Self {
                time_ms: self.time_ms,
                random: self.random,
                rng: self.rng.clone(),
        }
    }
}

#[cfg(feature = "create")]
impl Default for UlidGeneratorMonotonic<StdRng> {
    fn default() -> Self {
        Self {
                time_ms: 0,
                random: 0,
                rng: StdRng::from_os_rng(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ulid_generate::standard::*;

    struct FixedRng{
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
        unix_time: u64,
        expected: &'static str,
    }

    static PARAMS: [Param; 4] = [
        Param { rng_byte: 0, unix_time: 0, expected: "00000000000000000000000001" },
        Param { rng_byte: 0xFF, unix_time: 0, expected: "00000000000000000000000001" },
        Param { rng_byte: 0, unix_time: MAX_TIME_MS, expected: "7ZZZZZZZZZ0000000000000000" },
        Param { rng_byte: 0xFF, unix_time: MAX_TIME_MS, expected: "7ZZZZZZZZZZZZZZZZZZZZZZZZZ" },
    ];

    #[test]
    fn test_ulid_generator_monotonic_too_far_future() {
        let mut generator = UlidGeneratorMonotonic::default();
        let result = generator.create(Duration::from_millis(MAX_TIME_MS + 1));

        assert!(matches!(result, Err(UlidGenerationError::TooFarFuture(t)) if t == Duration::from_millis(1)));
    }

    #[test]
    fn test_ulid_generator_cannot_generate_monotonic() {
        let mut generator = UlidGeneratorMonotonic { rng: FixedRng{byte: 0xFF}, time_ms: 0, random: MAX_RANDOMNESS_VALUE };
        let result = generator.create(Duration::from_millis(0));

        assert!(matches!(result, Err(UlidGenerationError::CannotGenerateMonotonic(t)) if t == Duration::from_millis(0)));
    }

    #[test]
    fn test_ulid_generator_monotonic() {
        for param in PARAMS.clone() {
            let mut generator = UlidGeneratorMonotonic { rng: FixedRng{byte: param.rng_byte}, time_ms: 0, random: 0 };
            let result = generator.create(Duration::from_millis(param.unix_time));
            match result {
                Ok(ulid) => {
                    assert_eq!(ulid.to_string(), param.expected);
                }
                Err(e) => {panic!("Unexpected error: {:?}", e);}
            }
        }
    }
}
