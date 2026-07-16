// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
use blake3::{Hasher, OutputReader};

#[repr(align(16))]
#[derive(Clone, Copy)]
struct Aligned<const SIZE: usize>([u8; SIZE]);

#[derive(Clone)]
pub struct Prng<const SIZE: usize = 64> {
    reader: OutputReader,
    cursor: usize,
    buffer: Aligned<SIZE>,
}

/// A one-block (64-byte) XOF buffer and the generic fallback.
pub type Prng64 = Prng<64>;

/// A sixteen-block (1024-byte) XOF buffer, matching BLAKE3's AVX-512 XOF batch.
pub type Prng1024 = Prng<1024>;

mod sealed {
    pub trait Sealed {}
}

/// Fixed-width integers `Prng` can draw and bound. Sealed: the methods are
/// plumbing for `Prng::next`, `Prng::below`, and `Prng::range`, not for callers.
pub trait PrngInt: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn next<const SIZE: usize>(prng: &mut Prng<SIZE>) -> Self;
    #[doc(hidden)]
    fn below<const SIZE: usize>(prng: &mut Prng<SIZE>, n: Self) -> Self;
    #[doc(hidden)]
    fn range<const SIZE: usize>(prng: &mut Prng<SIZE>, low: Self, high: Self) -> Self;
}

macro_rules! impl_unsigned_prng_int {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}

            impl PrngInt for $ty {
                #[inline]
                fn next<const SIZE: usize>(prng: &mut Prng<SIZE>) -> Self {
                    Self::from_le_bytes(prng.take_bytes())
                }

                #[inline]
                fn below<const SIZE: usize>(prng: &mut Prng<SIZE>, n: Self) -> Self {
                    assert!(n > 0, "upper bound must be > 0");
                    let threshold = n.wrapping_neg() % n;
                    loop {
                        let x = Self::next(prng);
                        if x >= threshold {
                            return x % n;
                        }
                    }
                }

                #[inline]
                fn range<const SIZE: usize>(
                    prng: &mut Prng<SIZE>,
                    low: Self,
                    high: Self,
                ) -> Self {
                    assert!(low < high, "low must be < high");
                    low + Self::below(prng, high - low)
                }
            }
        )+
    };
}

macro_rules! impl_signed_prng_int {
    ($($signed:ty => $unsigned:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $signed {}

            impl PrngInt for $signed {
                #[inline]
                fn next<const SIZE: usize>(prng: &mut Prng<SIZE>) -> Self {
                    <$unsigned as PrngInt>::next(prng) as $signed
                }

                #[inline]
                fn below<const SIZE: usize>(prng: &mut Prng<SIZE>, n: Self) -> Self {
                    assert!(n > 0, "upper bound must be > 0");
                    <$unsigned as PrngInt>::below(prng, n as $unsigned) as $signed
                }

                #[inline]
                fn range<const SIZE: usize>(
                    prng: &mut Prng<SIZE>,
                    low: Self,
                    high: Self,
                ) -> Self {
                    assert!(low < high, "low must be < high");
                    let span = high.wrapping_sub(low) as $unsigned;
                    let offset = <$unsigned as PrngInt>::below(prng, span);
                    low.wrapping_add(offset as $signed)
                }
            }
        )+
    };
}

macro_rules! impl_next_aliases {
    ($($method:ident => $ty:ty),+ $(,)?) => {
        $(
            #[inline]
            pub fn $method(&mut self) -> $ty {
                self.next::<$ty>()
            }
        )+
    };
}

impl_unsigned_prng_int!(u8, u16, u32, u64, u128, usize);
impl_signed_prng_int!(
    i8 => u8,
    i16 => u16,
    i32 => u32,
    i64 => u64,
    i128 => u128,
    isize => usize,
);

impl<const SIZE: usize> Prng<SIZE> {
    #[inline]
    fn refill(&mut self) {
        self.reader.fill(&mut self.buffer.0);
        self.cursor = 0;
    }

    /// Take the next `N` bytes of the XOF stream. The common path is one
    /// unchecked fixed-width read; the boundary path preserves any bytes left
    /// in the current buffer before refilling, so mixed-width draws remain a
    /// contiguous view of the underlying stream.
    #[inline]
    fn take_bytes<const N: usize>(&mut self) -> [u8; N] {
        const {
            assert!(N > 0, "draw width must be greater than zero");
            assert!(N <= SIZE, "draw width must fit in the buffer");
        }

        if self.cursor == SIZE {
            self.refill();
        }

        let remaining = SIZE - self.cursor;
        if N <= remaining {
            // SAFETY: `cursor <= SIZE - N`, so the complete `N`-byte array is
            // inside `buffer`. Arrays of bytes have alignment one.
            let bytes = unsafe {
                self.buffer
                    .0
                    .as_ptr()
                    .add(self.cursor)
                    .cast::<[u8; N]>()
                    .read()
            };
            self.cursor += N;
            return bytes;
        }

        let mut bytes = [0u8; N];
        bytes[..remaining].copy_from_slice(&self.buffer.0[self.cursor..]);
        self.refill();

        let needed = N - remaining;
        bytes[remaining..].copy_from_slice(&self.buffer.0[..needed]);
        self.cursor = needed;
        bytes
    }

    #[inline]
    fn from_reader(reader: OutputReader) -> Self {
        const {
            assert!(SIZE >= 16, "buffer must be at least 16 bytes");
            assert!(
                SIZE.is_multiple_of(16),
                "buffer must be a multiple of 16 bytes"
            );
        }

        Self {
            reader,
            cursor: SIZE,
            buffer: Aligned([0u8; SIZE]),
        }
    }

    pub fn from_key(key: &[u8; 32]) -> Self {
        Self::from_reader(Hasher::new_keyed(key).finalize_xof())
    }

    pub fn from_key_with_parts(key: &[u8; 32], parts: &[&[u8]]) -> Self {
        let mut hasher = Hasher::new_keyed(key);
        for p in parts {
            let prefix = (p.len() as u64).to_le_bytes();
            hasher.update(&prefix);
            hasher.update(p);
        }
        Self::from_reader(hasher.finalize_xof())
    }

    pub fn from_key_with_parts_u8<const N: usize>(key: &[u8; 32], parts: &[&[u8]]) -> Self {
        let mut buf = [0u8; N];
        let mut len = 0usize;

        for p in parts {
            assert!(p.len() <= u8::MAX as usize, "part too long for a u8 prefix");
            assert!(len + 1 + p.len() <= N, "parts overflow the packed buffer");
            buf[len] = p.len() as u8;
            buf[len + 1..len + 1 + p.len()].copy_from_slice(p);
            len += 1 + p.len();
        }

        Self::from_reader(Hasher::new_keyed(key).update(&buf[..len]).finalize_xof())
    }

    pub fn from_key_with_parts_u16<const N: usize>(key: &[u8; 32], parts: &[&[u8]]) -> Self {
        let mut buf = [0u8; N];
        let mut len = 0usize;

        for p in parts {
            assert!(
                p.len() <= u16::MAX as usize,
                "part too long for a u16 prefix"
            );
            assert!(len + 2 + p.len() <= N, "parts overflow the packed buffer");
            buf[len..len + 2].copy_from_slice(&(p.len() as u16).to_le_bytes());
            buf[len + 2..len + 2 + p.len()].copy_from_slice(p);
            len += 2 + p.len();
        }

        Self::from_reader(Hasher::new_keyed(key).update(&buf[..len]).finalize_xof())
    }

    // The ladder is 32/128 with u8 prefixes, then 1024/2048 with u16
    // fallback to slow algo, TODO: log those so we can bench and add more rungs
    pub fn from_key_with_parts_dispatch(key: &[u8; 32], parts: &[&[u8]]) -> Self {
        let content: usize = parts.iter().map(|p| p.len()).sum();

        // Fitting a u8 tier implies every part fits its u8 prefix (a part
        // is at most the packed total minus its own prefix byte, < 255),
        // so the tier comparisons alone decide.
        let packed_u8 = content + parts.len();
        if packed_u8 <= 32 {
            return Self::from_key_with_parts_u8::<32>(key, parts);
        }
        if packed_u8 <= 128 {
            return Self::from_key_with_parts_u8::<128>(key, parts);
        }

        let packed_u16 = content + parts.len() * 2;
        if packed_u16 <= 1024 {
            return Self::from_key_with_parts_u16::<1024>(key, parts);
        }
        if packed_u16 <= 2048 {
            return Self::from_key_with_parts_u16::<2048>(key, parts);
        }

        Self::from_key_with_parts(key, parts)
    }

    pub fn next<T: PrngInt>(&mut self) -> T {
        T::next(self)
    }

    impl_next_aliases! {
        next_u8 => u8,
        next_u16 => u16,
        next_u32 => u32,
        next_u64 => u64,
        next_u128 => u128,
        next_usize => usize,
        next_i8 => i8,
        next_i16 => i16,
        next_i32 => i32,
        next_i64 => i64,
        next_i128 => i128,
        next_isize => isize,
    }

    pub fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    // A uniform f64 in (0, 1]: the top 53 bits of a u64 draw form the mantissa,
    // and +1 before scaling by 2^-53 nudges the usual half-open [0, 1) up one
    // ulp, so 0 never occurs and 1.0 is reachable.
    #[allow(unused)]
    pub fn next_f64(&mut self) -> f64 {
        let res = ((self.next_u64() >> 11) + 1) as f64 * 2f64.powi(-53);
        assert!(res > 0.0 && res <= 1.0, "next_f64 escaped (0, 1]");
        res
    }

    pub fn range<T: PrngInt>(&mut self, low: T, high: T) -> T {
        T::range(self, low, high)
    }

    pub fn index(&mut self, length: usize) -> usize {
        assert!(length > 0, "length must be > 0");
        self.below(length as u64) as usize
    }

    // unbiased integer in [0, n)
    pub fn below<T: PrngInt>(&mut self, n: T) -> T {
        T::below(self, n)
    }
}

#[cfg(test)]
mod test {
    use super::*;

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
}
