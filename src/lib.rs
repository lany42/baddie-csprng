// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
#![no_std]
//! # baddie-csprng
//!
//! A deterministic, keyed CSPRNG backed by the BLAKE3 extendable-output
//! function (XOF).
//!
//! [`Prng`] presents one continuous XOF stream through byte, integer, Boolean,
//! floating-point, range, and index sampling APIs. [`Prng64`] uses a compact
//! 64-byte refill buffer, while [`Prng1024`] batches 1,024 bytes per refill.
//!
//! The same key and constructor inputs always produce the same stream. Keep the
//! key secret and unpredictable when outputs must not be reproducible by an
//! attacker. Cloning a generator also clones its current stream position.
//!
//! ## Quick Start
//!
//! ```rust
//! use baddie_csprng::Prng64;
//!
//! // Use a secret, unpredictable key in production.
//! let key = [7_u8; 32];
//! let mut prng = Prng64::from_key(&key);
//!
//! let _id = prng.next_u64();
//! let die_roll = prng.range(1_u8, 7);
//! let bytes = prng.rand_bytes();
//!
//! assert!((1..7).contains(&die_roll));
//! assert_eq!(bytes.len(), 64);
//! ```
//!
//! ## Framed Inputs
//!
//! [`Prng::from_key_with_parts_dispatch`] derives a stream from multiple byte
//! strings without losing their boundaries. Each part receives a length prefix,
//! so inputs such as `("a", "bc")` and `("ab", "c")` remain distinct.
//!
//! ```rust
//! use baddie_csprng::Prng64;
//!
//! let key = [7_u8; 32];
//! let mut prng =
//!     Prng64::from_key_with_parts_dispatch(&key, &[b"account", b"42"]);
//! let nonce = prng.rand_bytes();
//!
//! assert_eq!(nonce.len(), 64);
//! ```
#[cfg(test)]
extern crate std;

use blake3::{Hasher, OutputReader};

const F32_SCALE: f32 = f32::from_bits(0x3380_0000); // 2^-24
const F64_SCALE: f64 = f64::from_bits(0x3ca0_0000_0000_0000); // 2^-53

#[repr(align(16))]
#[derive(Clone, Copy)]
struct Aligned<const SIZE: usize>([u8; SIZE]);

/// A deterministic, keyed pseudorandom generator backed by BLAKE3's XOF.
///
/// `SIZE` controls the internal refill buffer and must be at least 16 and a
/// multiple of 16. It does not change the generated stream. Cloning a generator
/// duplicates its state, so both copies produce identical subsequent output
/// until they are consumed differently.
#[derive(Clone)]
pub struct Prng<const SIZE: usize = 64> {
    reader: OutputReader,
    cursor: usize,
    buffer: Aligned<SIZE>,
}

/// A [`Prng`] with a one-block, 64-byte XOF refill buffer.
pub type Prng64 = Prng<64>;

/// A [`Prng`] with a 1,024-byte refill buffer matching BLAKE3's AVX-512 XOF batch.
/// Worth benchmarking if your CPU support AVX-512
pub type Prng1024 = Prng<1024>;

mod sealed {
    pub trait Sealed {}
}

/// A fixed-width integer supported by [`Prng`]'s sampling methods.
///
/// Implementations are provided for every Rust signed and unsigned integer
/// type. This trait is sealed and cannot be implemented by external crates;
/// its methods are plumbing for [`Prng::next`], [`Prng::below`], and
/// [`Prng::range`].
pub trait PrngInt: sealed::Sealed + Copy {
    /// Draws the next value of this integer type from `prng`.
    #[doc(hidden)]
    fn next<const SIZE: usize>(prng: &mut Prng<SIZE>) -> Self;

    /// Draws an unbiased value from `0..n` using `prng`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not positive.
    #[doc(hidden)]
    fn below<const SIZE: usize>(prng: &mut Prng<SIZE>, n: Self) -> Self;

    /// Draws an unbiased value from `low..high` using `prng`.
    ///
    /// # Panics
    ///
    /// Panics if `low` is greater than or equal to `high`.
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
            /// Draws the next integer of this method's return type from the XOF stream.
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

    /// Constructs a generator whose stream is derived directly from `key`.
    pub fn from_key(key: &[u8; 32]) -> Self {
        Self::from_reader(Hasher::new_keyed(key).finalize_xof())
    }

    /// Constructs a generator after prefixing each part with its 64-bit length.
    ///
    /// Lengths use little-endian encoding, preserving part boundaries and empty
    /// parts in the derived stream.
    pub fn from_key_with_parts(key: &[u8; 32], parts: &[&[u8]]) -> Self {
        let mut hasher = Hasher::new_keyed(key);
        for p in parts {
            let prefix = (p.len() as u64).to_le_bytes();
            hasher.update(&prefix);
            hasher.update(p);
        }
        Self::from_reader(hasher.finalize_xof())
    }

    /// Constructs a generator using one-byte part lengths and an `N`-byte buffer.
    ///
    /// `N` limits the total packed input size but is not incorporated into the
    /// generated stream.
    ///
    /// # Panics
    ///
    /// Panics if a part is longer than [`u8::MAX`] bytes or if the length
    /// prefixes and parts require more than `N` bytes.
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

    /// Constructs a generator using two-byte part lengths and an `N`-byte buffer.
    ///
    /// Lengths use little-endian encoding. `N` limits the total packed input size
    /// but is not incorporated into the generated stream.
    ///
    /// # Panics
    ///
    /// Panics if a part is longer than [`u16::MAX`] bytes or if the length
    /// prefixes and parts require more than `N` bytes.
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
    /// Constructs a generator using a length-prefix format selected for `parts`.
    ///
    /// Uses one-byte prefixes through 128 packed bytes, two-byte little-endian
    /// prefixes through 2,048 packed bytes, and 64-bit little-endian prefixes
    /// otherwise. Crossing a prefix-width boundary selects a different
    /// deterministic stream encoding.
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

    /// Manually flushes and refills the internal buffer.
    ///
    /// This is called automatically internally, but may be
    /// useful to ensure [`Prng::rand_bytes`] always aligns
    /// on the sampling hot path. The flushed bytes are lost.
    #[inline]
    pub fn flush(&mut self) {
        self.refill();
    }

    /// Draws the next `SIZE` bytes from the XOF stream.
    pub fn rand_bytes(&mut self) -> [u8; SIZE] {
        self.take_bytes()
    }

    /// Draws the next value of integer type `T` from the XOF stream.
    ///
    /// Integer bytes are interpreted in little-endian order.
    #[allow(
        clippy::should_implement_trait,
        reason = "`Prng::next<T>` intentionally claims `next` as its generic sampling API"
    )]
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

    /// Draws a uniformly distributed Boolean, consuming one stream byte.
    pub fn next_bool(&mut self) -> bool {
        self.next_u8() & 1 == 1
    }

    /// Draws a uniformly distributed `f32` value from `(0.0, 1.0]`.
    ///
    /// The result is one of the 2²⁴ positive multiples of 2⁻²⁴ in this interval.
    pub fn next_f32(&mut self) -> f32 {
        let res = ((self.next_u32() >> 8) + 1) as f32 * F32_SCALE;
        assert!(res > 0.0 && res <= 1.0, "next_f32 escaped (0, 1]");
        res
    }

    // A uniform f64 in (0, 1]: the top 53 bits of a u64 draw form the mantissa,
    // and +1 before scaling by 2^-53 nudges the usual half-open [0, 1) up one
    // ulp, so 0 never occurs and 1.0 is reachable.
    /// Draws a uniformly distributed `f64` value from `(0.0, 1.0]`.
    ///
    /// The result is one of the 2⁵³ positive multiples of 2⁻⁵³ in this interval.
    pub fn next_f64(&mut self) -> f64 {
        let res = ((self.next_u64() >> 11) + 1) as f64 * F64_SCALE;
        assert!(res > 0.0 && res <= 1.0, "next_f64 escaped (0, 1]");
        res
    }

    /// Draws an unbiased integer from the half-open interval `low..high`.
    ///
    /// # Panics
    ///
    /// Panics if `low` is greater than or equal to `high`.
    pub fn range<T: PrngInt>(&mut self, low: T, high: T) -> T {
        T::range(self, low, high)
    }

    /// Draws an unbiased index from `0..length`.
    ///
    /// # Panics
    ///
    /// Panics if `length` is zero.
    pub fn index(&mut self, length: usize) -> usize {
        assert!(length > 0, "length must be > 0");
        self.below(length as u64) as usize
    }

    // unbiased integer in [0, n)
    /// Draws an unbiased integer from the half-open interval `0..n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not positive.
    pub fn below<T: PrngInt>(&mut self, n: T) -> T {
        T::below(self, n)
    }
}

#[cfg(test)]
mod test;
