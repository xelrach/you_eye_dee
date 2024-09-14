use core::arch::aarch64::{uint8x8_t, uint8x8x3_t, uint8x8x4_t, vld4_u8};

pub unsafe fn ulid_to_u128_neon(input: &str) -> Result<u128, DecodeError> {
    let input = vld4_u8(input.as_ptr());

    let field_a = lookup(input.val[0], error_a);
    let field_b = lookup(input.val[1], error_b);
    let field_c = lookup(input.val[2], error_c);
    let field_d = lookup(input.val[3], error_d);

    let mut result: uint8x8x3_t;
    result.val[0] = vorr_u8(vshr_n_u8(field_b, 4), vshl_n_u8(field_a, 2));
    result.val[1] = vorr_u8(vshr_n_u8(field_c, 2), vshl_n_u8(field_b, 4));
    result.val[2] = vorr_u8(field_d, vshl_n_u8(field_c, 6));
}

#[inline(always)]
unsafe fn lookup_pshufb_bitmask(input: uint8x8_t) -> uint8x8_t {

const uint8x8_t higher_nibble = vshr_n_u8(input, 4);
const uint8x8_t lower_nibble = vand_u8(input, packed_byte(0x0f));

const uint8x8x2_t shiftLUT = {
0, 0, 19, 4, uint8_t(- 65), uint8_t( - 65), uint8_t( - 71), uint8_t( - 71),
0, 0, 0, 0,   0, 0, 0, 0};

const uint8x8x2_t maskLUT = {
/* 0        : 0b1010_1000*/ 0xa8,
/* 1 .. 9   : 0b1111_1000*/ 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8, 0xf8,
/* 10       : 0b1111_0000*/ 0xf0,
/* 11       : 0b0101_0100*/ 0x54,
/* 12 .. 14 : 0b0101_0000*/ 0x50, 0x50, 0x50,
/* 15       : 0b0101_0100*/ 0x54
};

const uint8x8x2_t bitposLUT = {
0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

const uint8x8_t sh = vtbl2_u8(shiftLUT, higher_nibble);
const uint8x8_t eq_2f = vceq_u8(input, packed_byte(0x2f));
const uint8x8_t shift = vbsl_u8(eq_2f, packed_byte(16), sh);

const uint8x8_t M = vtbl2_u8(maskLUT,   lower_nibble);
const uint8x8_t bit = vtbl2_u8(bitposLUT, higher_nibble);

error = vceq_u8(vand_u8(M, bit), packed_byte(0));

const uint8x8_t result = vadd_u8(input, shift);

return result;
}
