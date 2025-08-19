/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::time::Duration;

#[derive(Debug)]
pub enum UlidGenerationError {
    CannotGenerateMonotonic(Duration),
    TooFarFuture(Duration),
    TooFarPast(Duration),
}

pub mod standard;
pub mod stateless;
mod consts;