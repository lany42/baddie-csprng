# baddie-csprng

A keyed CSPRNG backed by the BLAKE3 cryptographic hash function.

## Quickstart

```rust
use baddie_csprng::Prng64;

let key = [42u8; 32]; // Replace with a securely generated 32-byte key.
let mut rng = Prng64::from_key(&key);

let first = rng.next_u64();
let second = rng.next_u64();

// Statistically, equal with probability 1 / 2^64.
assert_ne!(first, second);
```

The same key produces the same stream. This crate does not obtain entropy from
the operating system, so callers are responsible for securely generating and
managing keys.

## License

Licensed under the [GNU Affero General Public License v3.0 only](LICENSE).
