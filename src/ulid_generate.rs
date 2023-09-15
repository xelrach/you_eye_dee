use std::cell::Cell;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use crate::Ulid;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::borrow::BorrowMut;

const MAX_TIME: Duration = Duration::from_millis(0);
const MAX_RANDOM: u128 = 0;

pub enum UlidGenerationError {
    CannotGenerateMonotonic(Duration),
    TooFarFuture(Duration),
    TooFarPast(Duration),
    Random(rand::Error),
}

struct State<T: RngCore + SeedableRng> {
    time_ms: u64,
    random: u128,
    rng: T,
}

pub struct UlidGeneratorMonotonic {
    state: Mutex<State<ChaCha12Rng>>,
}

pub struct UlidGeneratorNonMonotonic {
    rng: Cell<ChaCha12Rng>,
}

trait UlidGenerator {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError>;
    fn create_now(&self) -> Result<Ulid, UlidGenerationError> {
        let unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
        match unix_time {
            Ok(unix_time) => self.create(unix_time),
            Err(time_err) => Err(UlidGenerationError::TooFarPast(time_err.duration())),
        }
    }
}

impl UlidGenerator for UlidGeneratorMonotonic {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME));
        }

        let time_ms = unix_time.as_millis() as u64;
        let mut state = self.state.lock().unwrap();
        if state.time_ms <= time_ms {
            let random = state.random + 1;
            if random > MAX_RANDOM {
                return Err(UlidGenerationError::CannotGenerateMonotonic(unix_time));
            }
            state.time_ms = time_ms;
            state.random = random;

            Ok(Ulid::from_ms_and_random(time_ms, random))
        } else {
            let mut ulid_bytes: [u8; 16] = [0; 16];
            state
                .rng
                .try_fill_bytes(&mut ulid_bytes[0..10])
                .map_err(UlidGenerationError::Random)?;
            ulid_bytes[6..10].copy_from_slice(&time_ms.to_le_bytes()[0..6]);
            let ulid = Ulid::from_le_bytes(ulid_bytes);
            state.time_ms = ulid.get_time_ms();
            state.random = ulid.get_random();

            Ok(ulid)
        }
    }
}

impl UlidGenerator for UlidGeneratorNonMonotonic {
    fn create(&self, unix_time: Duration) -> Result<Ulid, UlidGenerationError> {
        if unix_time > MAX_TIME {
            return Err(UlidGenerationError::TooFarFuture(unix_time - MAX_TIME));
        }

        let time_ms = unix_time.as_millis() as u64;

        let mut ulid_bytes: [u8; 16] = [0; 16];
        todo!()
        /*
        self.rng
            .borrow_mut()
            .into_inner()
            .try_fill_bytes(&mut ulid_bytes[0..10])
            .map_err(UlidGenerationError::Random)?;
        ulid_bytes[6..10].copy_from_slice(&time_ms.to_le_bytes()[0..6]);
        let ulid = Ulid::from_le_bytes(ulid_bytes);

        Ok(ulid)
             */
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
