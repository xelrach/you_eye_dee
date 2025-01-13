/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::{Duration, SystemTime};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::Ulid;

const MAX_TIME: Duration = Duration::from_millis(0);
const MAX_RANDOM: u128 = 0;

pub enum UlidGenerationError {
    CannotGenerateMonotonic(Duration),
    TooFarFuture(Duration),
    TooFarPast(Duration),
    RandomGenerationError(rand::Error),
}

#[derive(Debug)]
struct State<T: RngCore + SeedableRng> {
    time_ms: u64,
    random: u128,
    rng: T,
}

#[derive(Debug, Default)]
pub struct UlidGeneratorMonotonic {
    state: State<ChaCha12Rng>,
}

#[derive(Debug)]
pub struct UlidGeneratorNonMonotonic {
    rng: ChaCha12Rng,
}

impl UlidGeneratorMonotonic {
    pub fn create(&mut self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME));
        }

        let time_ms = unix_time.as_millis() as u64;
        if self.state.time_ms <= time_ms {
            let random = self.state.random + 1;
            if random > MAX_RANDOM {
                return Err(UlidGenerationError::CannotGenerateMonotonic(unix_time));
            }
            self.state.time_ms = time_ms;
            self.state.random = random;

            Ok(Ulid::from_ms_and_random(time_ms, random))
        } else {
            let mut ulid_bytes: [u8; 16] = [0; 16];
            self.state
                .rng
                .try_fill_bytes(&mut ulid_bytes[0..10])
                .map_err(UlidGenerationError::RandomGenerationError)?;
            ulid_bytes[6..10].copy_from_slice(&time_ms.to_le_bytes()[0..6]);
            let ulid = Ulid::from_le_bytes(ulid_bytes);
            self.state.time_ms = ulid.get_time_ms();
            self.state.random = ulid.get_random();

            Ok(ulid)
        }
    }

    pub fn create_now(&mut self) -> Result<Ulid, UlidGenerationError> {
        let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
        match unix_time {
            Ok(unix_time) => self.create(unix_time),
            Err(time_err) => Err(UlidGenerationError::TooFarPast(time_err.duration())),
        }
    }
}

impl UlidGeneratorNonMonotonic {
    pub fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME));
        }

        let time_ms = unix_time.as_millis() as u64;

        let mut ulid_bytes: [u8; 16] = [0; 16];

        rand::thread_rng()
            .try_fill_bytes(&mut ulid_bytes[0..10])
            .map_err(UlidGenerationError::RandomGenerationError)?;
        ulid_bytes[6..10].copy_from_slice(&time_ms.to_le_bytes()[0..6]);
        let ulid = Ulid::from_le_bytes(ulid_bytes);

        Ok(ulid)
    }

    pub fn create_now(&self) -> Result<Ulid, UlidGenerationError> {
        let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
        match unix_time {
            Ok(unix_time) => self.create(unix_time),
            Err(time_err) => Err(UlidGenerationError::TooFarPast(time_err.duration())),
        }
    }
}

impl Clone for UlidGeneratorNonMonotonic {
    fn clone(&self) -> Self {
        let mut cloned_rng = self.rng.clone();
        cloned_rng.set_stream(cloned_rng.get_stream() + 1);
        Self { rng: cloned_rng }
    }
}

impl Default for UlidGeneratorNonMonotonic {
    fn default() -> Self {
        Self {
            rng: ChaCha12Rng::from_entropy(),
        }
    }
}

impl Clone for UlidGeneratorMonotonic {
    fn clone(&self) -> Self {
        let mut cloned_rng = self.state.rng.clone();
        cloned_rng.set_stream(cloned_rng.get_stream() + 1);
        Self {
            state: State {
                time_ms: self.state.time_ms,
                random: self.state.random,
                rng: cloned_rng,
            },
        }
    }
}

impl<T: RngCore + SeedableRng> Default for State<T> {
    fn default() -> Self {
        Self {
            time_ms: 0,
            random: 0,
            rng: T::from_entropy(),
        }
    }
}
