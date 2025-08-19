/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::Duration;

// (2^48) - 1
const MAX_TIME_MILLIS: u64 = 0xFFFF_FFFFFFFF;
pub const NANOS_IN_MILLI: u32 = 1_000_000;
pub const MAX_TIME_DURATION: Duration = Duration::new(
    MAX_TIME_MILLIS / 1000,
    ((MAX_TIME_MILLIS % 1000) as u32 * NANOS_IN_MILLI) + 999_999);

pub const ULID_LENGTH: usize = 16;
