// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
use super::*;
use std::{vec, vec::Vec};

fn get_key() -> [u8; 32] {
    std::array::from_fn(|i| i as u8)
}

// Collect the next `n` u64s from a Prng so streams can be compared.
fn take(prng: &mut Prng, n: usize) -> Vec<u64> {
    (0..n).map(|_| prng.next_u64()).collect()
}

#[test]
fn test_overlapping_parts_produce_different_integers() {
    // Both part lists concatenate to the bytes "abc"; only the length
    // prefix on each part keeps the boundary unambiguous. Without it
    // these two streams would be identical, so this asserts the prefix
    // actually disambiguates ("a","bc") from ("ab","c").
    let mut a = Prng::from_key_with_parts(&get_key(), &["a".as_bytes(), "bc".as_bytes()]);
    let mut b = Prng::from_key_with_parts(&get_key(), &["ab".as_bytes(), "c".as_bytes()]);
    assert_ne!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 1, u8 packing: both lists concatenate to "abc"; only the
// per-part length prefix keeps the boundary unambiguous.
#[test]
fn u8_packed_part_boundaries_split_the_stream() {
    let mut a = Prng::from_key_with_parts_u8::<32>(&get_key(), &[b"a", b"bc"]);
    let mut b = Prng::from_key_with_parts_u8::<32>(&get_key(), &[b"ab", b"c"]);
    assert_ne!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 1, u16 packing
#[test]
fn u16_packed_part_boundaries_split_the_stream() {
    let mut a = Prng::from_key_with_parts_u16::<1024>(&get_key(), &[b"a", b"bc"]);
    let mut b = Prng::from_key_with_parts_u16::<1024>(&get_key(), &[b"ab", b"c"]);
    assert_ne!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 2, u8 tiers: the buffer size is invisible to the stream --
// what makes re-tiering safe and the 32/128 split purely a perf choice
#[test]
fn u8_buffer_size_never_enters_the_stream() {
    let mut a = Prng::from_key_with_parts_u8::<32>(&get_key(), &[b"a", b"bc"]);
    let mut b = Prng::from_key_with_parts_u8::<2048>(&get_key(), &[b"a", b"bc"]);
    assert_eq!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 2, u16 tiers
#[test]
fn u16_buffer_size_never_enters_the_stream() {
    let mut a = Prng::from_key_with_parts_u16::<1024>(&get_key(), &[b"a", b"bc"]);
    let mut b = Prng::from_key_with_parts_u16::<2048>(&get_key(), &[b"a", b"bc"]);
    assert_eq!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 3: the prefix width is the stream family -- the same parts
// packed u8 and u16 must diverge, which is why dispatch crossing the
// 128 boundary re-keys and staying within it cannot
#[test]
fn u8_and_u16_prefix_streams_differ() {
    let mut a = Prng::from_key_with_parts_u8::<128>(&get_key(), &[b"a", b"bc"]);
    let mut b = Prng::from_key_with_parts_u16::<128>(&get_key(), &[b"a", b"bc"]);
    assert_ne!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 4: a list too long for the buffer panics (u8 packing)
#[test]
#[should_panic(expected = "parts overflow the packed buffer")]
fn u8_packing_overflowing_the_buffer_panics() {
    Prng64::from_key_with_parts_u8::<32>(&get_key(), &[&[0u8; 40]]);
}

// Invariant 4: a part too long for its u8 prefix panics even when the
// buffer itself would hold it
#[test]
#[should_panic(expected = "part too long for a u8 prefix")]
fn u8_packing_part_past_prefix_range_panics() {
    Prng64::from_key_with_parts_u8::<512>(&get_key(), &[&[0u8; 256]]);
}

// Invariant 4: a list too long for the buffer panics (u16 packing)
#[test]
#[should_panic(expected = "parts overflow the packed buffer")]
fn u16_packing_overflowing_the_buffer_panics() {
    Prng64::from_key_with_parts_u16::<32>(&get_key(), &[&[0u8; 40]]);
}

// Invariant 5, u8 side: a list that packs to exactly 128 dispatches
// into the u8 family
#[test]
fn dispatch_stays_u8_at_the_128_boundary() {
    let part = [7u8; 127]; // 127 bytes + 1 prefix = 128 packed
    let mut a = Prng::from_key_with_parts_dispatch(&get_key(), &[&part]);
    let mut b = Prng::from_key_with_parts_u8::<128>(&get_key(), &[&part]);
    assert_eq!(take(&mut a, 8), take(&mut b, 8));
}

// Invariant 5, u16 side: one byte past the boundary shifts family
#[test]
fn dispatch_shifts_to_u16_past_the_128_boundary() {
    let part = [7u8; 128]; // 128 bytes + 1 prefix = 129 packed as u8
    let mut a = Prng::from_key_with_parts_dispatch(&get_key(), &[&part]);
    let mut b = Prng::from_key_with_parts_u16::<1024>(&get_key(), &[&part]);
    assert_eq!(take(&mut a, 8), take(&mut b, 8));
}

#[test]
#[should_panic(expected = "upper bound must be > 0")]
fn below_zero_bound_panics() {
    Prng64::from_key(&get_key()).below(0u32);
}

#[test]
#[should_panic(expected = "upper bound must be > 0")]
fn below_negative_bound_panics() {
    Prng64::from_key(&get_key()).below(-7i32);
}

#[test]
fn next_f32_stays_in_its_open_closed_unit_interval() {
    let mut prng = Prng64::from_key(&get_key());
    for _ in 0..1_024 {
        let value = prng.next_f32();
        assert!(value > 0.0 && value <= 1.0);
    }
}

#[test]
fn float_scale_constants_are_exact_powers_of_two() {
    assert_eq!(F32_SCALE.to_bits(), 0x3380_0000);
    assert_eq!(F64_SCALE.to_bits(), 0x3ca0_0000_0000_0000);
}

#[test]
fn float_output_bits_are_pinned() {
    let key = get_key();
    let mut f32_prng = Prng64::from_key(&key);
    let mut f64_prng = Prng64::from_key(&key);

    assert_eq!(
        [
            f32_prng.next_f32().to_bits(),
            f32_prng.next_f32().to_bits(),
            f32_prng.next_f32().to_bits(),
            f32_prng.next_f32().to_bits(),
        ],
        [0x3dc9_5a50, 0x3f4d_715e, 0x3e9b_afd4, 0x3d19_8c10]
    );
    assert_eq!(
        [
            f64_prng.next_f64().to_bits(),
            f64_prng.next_f64().to_bits(),
            f64_prng.next_f64().to_bits(),
            f64_prng.next_f64().to_bits(),
        ],
        [
            0x3fe9_ae2b_b323_256a,
            0x3fa3_3181_d89b_afe0,
            0x3fe8_52b7_8003_65ef,
            0x3fe1_e8fa_da93_abf7,
        ]
    );
}

#[test]
#[should_panic(expected = "low must be < high")]
fn range_low_equal_high_panics() {
    Prng64::from_key(&get_key()).range(5u32, 5u32);
}

fn take_mixed_widths<const SIZE: usize>(prng: &mut Prng<SIZE>, rounds: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for _ in 0..rounds {
        bytes.extend_from_slice(&prng.next::<u8>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<u16>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<u32>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<u64>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<u128>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<usize>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<i8>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<i16>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<i32>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<i64>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<i128>().to_le_bytes());
        bytes.extend_from_slice(&prng.next::<isize>().to_le_bytes());
    }
    bytes
}

#[test]
fn miri_smoke_buffering_crosses_refill_boundaries() {
    let key = get_key();
    let mut prng = Prng::<16>::from_key(&key);
    let actual = take_mixed_widths(&mut prng, 1);
    let mut expected = vec![0u8; actual.len()];
    Hasher::new_keyed(&key).finalize_xof().fill(&mut expected);

    assert_eq!(actual, expected);
}

// Buffering must be a transparent typed view of one contiguous XOF stream.
#[test]
fn buffering_is_transparent_to_the_xof_stream() {
    const ROUNDS: usize = 16;

    let key = get_key();
    let mut smallest = Prng::<16>::from_key(&key);
    let smallest_bytes = take_mixed_widths(&mut smallest, ROUNDS);

    let mut expected = vec![0u8; smallest_bytes.len()];
    Hasher::new_keyed(&key).finalize_xof().fill(&mut expected);

    assert_eq!(smallest_bytes, expected);
    assert_eq!(
        take_mixed_widths(&mut Prng64::from_key(&key), ROUNDS),
        expected
    );
    assert_eq!(
        take_mixed_widths(&mut Prng1024::from_key(&key), ROUNDS),
        expected
    );
}

// Every integer implementation must keep `below(n)` inside [0, n).
#[test]
fn below_stays_in_its_half_open_domain_for_every_integer_type() {
    macro_rules! check {
        ($ty:ty) => {{
            let mut prng = Prng64::from_key(&get_key());
            assert_eq!(prng.below(1 as $ty), 0 as $ty);

            let bound = 101 as $ty;
            for _ in 0..128 {
                let value = prng.below(bound);
                assert!((0 as $ty..bound).contains(&value));
            }
        }};
    }

    check!(u8);
    check!(u16);
    check!(u32);
    check!(u64);
    check!(u128);
    check!(usize);
    check!(i8);
    check!(i16);
    check!(i32);
    check!(i64);
    check!(i128);
    check!(isize);
}

// Range sampling must survive near-maximum and wrapped signed spans.
#[test]
fn range_handles_extreme_integer_spans() {
    macro_rules! check_unsigned {
        ($ty:ty) => {{
            let mut prng = Prng64::from_key(&get_key());
            let high = <$ty>::MAX;
            let low = high - 100;
            for _ in 0..64 {
                let value = prng.range(low, high);
                assert!((low..high).contains(&value));
            }
            assert_eq!(prng.range(high - 1, high), high - 1);
        }};
    }

    macro_rules! check_signed {
        ($ty:ty) => {{
            let mut prng = Prng64::from_key(&get_key());
            let cross_zero_low: $ty = -100;
            let cross_zero_high: $ty = 101;
            for _ in 0..64 {
                let full_span = prng.range(<$ty>::MIN, <$ty>::MAX);
                assert!((<$ty>::MIN..<$ty>::MAX).contains(&full_span));

                let cross_zero = prng.range(cross_zero_low, cross_zero_high);
                assert!((cross_zero_low..cross_zero_high).contains(&cross_zero));
            }
            assert_eq!(prng.range(-1, 0), -1);
        }};
    }

    check_unsigned!(u8);
    check_unsigned!(u16);
    check_unsigned!(u32);
    check_unsigned!(u64);
    check_unsigned!(u128);
    check_unsigned!(usize);
    check_signed!(i8);
    check_signed!(i16);
    check_signed!(i32);
    check_signed!(i64);
    check_signed!(i128);
    check_signed!(isize);
}

// Dispatch keeps u16 prefixes through 2048 packed bytes, then uses u64.
#[test]
fn dispatch_switches_prefix_family_past_2048() {
    let exact = [7u8; 2046];
    let mut exact_dispatch = Prng::from_key_with_parts_dispatch(&get_key(), &[&exact]);
    let mut exact_u16 = Prng::from_key_with_parts_u16::<2048>(&get_key(), &[&exact]);
    assert_eq!(take(&mut exact_dispatch, 8), take(&mut exact_u16, 8));

    let past = [7u8; 2047];
    let mut past_dispatch = Prng::from_key_with_parts_dispatch(&get_key(), &[&past]);
    let mut past_u64 = Prng::from_key_with_parts(&get_key(), &[&past]);
    assert_eq!(take(&mut past_dispatch, 8), take(&mut past_u64, 8));
}
