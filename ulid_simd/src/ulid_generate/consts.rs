/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::Duration;

pub const MAX_TIME_MS: u64 = 0xFFFF_FFFFFFFF;
pub const MAX_TIME_DURATION: Duration = Duration::from_millis(MAX_TIME_MS);
pub const NANOS_IN_MILLIS: u32 = 1_000_000;
pub const ULID_LENGTH: usize = 16;
