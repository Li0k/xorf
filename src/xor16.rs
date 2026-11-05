//! Implements Xor16 filters as described in [Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters].
//!
//! [Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters]: https://arxiv.org/abs/1912.08258

use crate::{xor_contains_impl, xor_from_impl, Filter};
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// Xor filter using 16-bit fingerprints.
///
/// An `Xor16` filter uses <20 bits per entry of the set is it constructed from, and has a false
/// positive rate of <0.002%. As with other probabilistic filters, a higher number of entries decreases
/// the bits per entry but increases the false positive rate.
///
/// An `Xor16` is constructed from a set of 64-bit unsigned integers and is immutable.
///
/// ```
/// # extern crate alloc;
/// use xorf::{Filter, Xor16};
/// # use alloc::vec::Vec;
/// # use rand::Rng;
///
/// # let mut rng = rand::thread_rng();
/// const SAMPLE_SIZE: usize = 1_000_000;
/// let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();
/// let filter = Xor16::from(&keys);
///
/// // no false negatives
/// for key in keys {
///     assert!(filter.contains(&key));
/// }
///
/// // bits per entry
/// let bpe = (filter.len() as f64) * 16.0 / (SAMPLE_SIZE as f64);
/// assert!(bpe < 20., "Bits per entry is {}", bpe);
///
/// // false positive rate
/// let false_positives: usize = (0..SAMPLE_SIZE)
///     .map(|_| rng.gen())
///     .filter(|n| filter.contains(n))
///     .count();
/// let fp_rate: f64 = (false_positives * 100) as f64 / SAMPLE_SIZE as f64;
/// assert!(fp_rate < 0.0025, "False positive rate is {}", fp_rate);
/// ```
///
/// Serializing and deserializing `Xor16` filters can be enabled with the [`serde`] feature  (or [`bincode`] for bincode).
///
/// [`serde`]: http://serde.rs
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
#[derive(Debug, Clone)]
pub struct Xor16 {
    /// The seed for the filter
    pub seed: u64,
    /// The number of blocks in the filter
    pub block_length: usize,
    /// The fingerprints for the filter
    pub fingerprints: Box<[u16]>,
}

impl Filter<u64> for Xor16 {
    /// Returns `true` if the filter contains the specified key. Has a false positive rate of <0.002%.
    fn contains(&self, key: &u64) -> bool {
        xor_contains_impl!(*key, self, fingerprint u16)
    }

    fn len(&self) -> usize {
        self.fingerprints.len()
    }
}

/// A borrowed [`Xor16`] backed by caller-owned fingerprints.
///
/// The seed, block length and fingerprints must come from the same filter.
/// Queries use the same implementation as the owning [`Xor16`].
#[derive(Debug, Clone)]
pub struct Xor16Ref<'a> {
    /// The seed for the filter.
    pub seed: u64,
    /// The length of each of the filter's three fingerprint blocks.
    pub block_length: usize,
    /// The fingerprints for the filter.
    pub fingerprints: &'a [u16],
}

impl Filter<u64> for Xor16Ref<'_> {
    #[inline]
    fn contains(&self, key: &u64) -> bool {
        xor_contains_impl!(*key, self, fingerprint u16)
    }

    fn len(&self) -> usize {
        self.fingerprints.len()
    }
}

/// A borrowed [`Xor16`] backed by little-endian fingerprint bytes.
///
/// The byte pairs require no alignment. The seed, block length and fingerprints
/// must come from the same filter.
#[derive(Debug, Clone, Copy)]
pub struct Xor16BytesRef<'a> {
    /// The seed for the filter.
    pub seed: u64,
    /// The length of each of the filter's three fingerprint blocks.
    pub block_length: usize,
    /// Little-endian fingerprints, indexed in elements rather than bytes.
    pub fingerprints: &'a [[u8; 2]],
}

impl<'a> Xor16BytesRef<'a> {
    /// Borrows a filter encoded as a little-endian `u64` seed, `u32` block length,
    /// and little-endian `u16` fingerprints, without copying or requiring alignment.
    ///
    /// Returns `None` for an incomplete header or fingerprint. This checks the byte layout,
    /// not whether the fingerprints describe a valid filter. Only query trusted encodings.
    #[inline]
    pub fn from_encoded_le(data: &'a [u8]) -> Option<Self> {
        let header = data.get(..12)?;
        let (fingerprints, remainder) = data[12..].as_chunks::<2>();
        if !remainder.is_empty() {
            return None;
        }
        Some(Self {
            seed: u64::from_le_bytes(header[..8].try_into().unwrap()),
            block_length: u32::from_le_bytes(header[8..].try_into().unwrap()) as usize,
            fingerprints,
        })
    }
}

impl From<Xor16BytesRef<'_>> for Xor16 {
    fn from(filter: Xor16BytesRef<'_>) -> Self {
        Self {
            seed: filter.seed,
            block_length: filter.block_length,
            fingerprints: filter
                .fingerprints
                .iter()
                .copied()
                .map(u16::from_le_bytes)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

impl Filter<u64> for Xor16BytesRef<'_> {
    #[inline]
    fn contains(&self, key: &u64) -> bool {
        xor_contains_impl!(*key, self, fingerprint u16, |index| {
            u16::from_le_bytes(self.fingerprints[index])
        })
    }

    fn len(&self) -> usize {
        self.fingerprints.len()
    }
}

impl Xor16 {
    /// Returns the byte length of the encoding written by [`Self::encode_le_into`].
    pub fn encoded_le_len(&self) -> usize {
        12 + self.fingerprints.len() * 2
    }

    /// Appends the portable encoding accepted by [`Xor16BytesRef::from_encoded_le`].
    ///
    /// # Panics
    /// Panics if the block length does not fit in a `u32`.
    pub fn encode_le_into(&self, out: &mut Vec<u8>) {
        let block_length = u32::try_from(self.block_length).expect("block length exceeds u32");
        out.reserve(self.encoded_le_len());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&block_length.to_le_bytes());
        for fingerprint in self.fingerprints.iter() {
            out.extend_from_slice(&fingerprint.to_le_bytes());
        }
    }

    /// Construct the filter from a key iterator. Can be used directly
    /// if you don't have a contiguous array of u64 keys.
    ///
    /// Note: the iterator will be iterated over multiple times while building
    /// the filter. If using a hash function to map the key, it may be cheaper
    /// just to create a scratch array of hashed keys that you pass in.
    pub fn from_iterator<T>(keys: T) -> Self
    where
        T: ExactSizeIterator<Item = u64> + Clone,
    {
        xor_from_impl!(keys fingerprint u16)
    }
}

impl From<&[u64]> for Xor16 {
    fn from(keys: &[u64]) -> Self {
        Self::from_iterator(keys.iter().copied())
    }
}

impl From<&Vec<u64>> for Xor16 {
    fn from(v: &Vec<u64>) -> Self {
        Self::from_iterator(v.iter().copied())
    }
}

impl From<Vec<u64>> for Xor16 {
    fn from(v: Vec<u64>) -> Self {
        Self::from_iterator(v.iter().copied())
    }
}

#[cfg(test)]
mod test {
    extern crate std;

    use crate::{Filter, Xor16, Xor16BytesRef};

    use alloc::vec::Vec;
    use rand::Rng;

    #[test]
    fn test_byte_reader() {
        let keys: Vec<u64> = (0..128).map(|key| key * 7919).collect();
        let owned = Xor16::from(&keys);
        let mut bytes = alloc::vec![0];
        owned.encode_le_into(&mut bytes);
        assert_eq!(bytes.len() - 1, owned.encoded_le_len());
        assert_eq!(&bytes[1..9], &owned.seed.to_le_bytes());
        assert_eq!(&bytes[9..13], &(owned.block_length as u32).to_le_bytes());
        let expected: Vec<_> = owned
            .fingerprints
            .iter()
            .flat_map(|fp| fp.to_le_bytes())
            .collect();
        assert_eq!(&bytes[13..], expected);
        let reader = Xor16BytesRef::from_encoded_le(&bytes[1..]).unwrap();
        assert_eq!(reader.len(), owned.len());
        let restored = Xor16::from(reader);
        for key in keys.iter().copied().chain(0..4096) {
            assert_eq!(reader.contains(&key), owned.contains(&key));
            assert_eq!(restored.contains(&key), owned.contains(&key));
        }
        assert!(Xor16BytesRef::from_encoded_le(&bytes[1..12]).is_none());
        assert!(Xor16BytesRef::from_encoded_le(&bytes[1..bytes.len() - 1]).is_none());

        // The third probe must not read beyond the supplied child slice.
        let bytes = [[0, 0]; 3];
        let truncated = Xor16BytesRef {
            seed: 0,
            block_length: 1,
            fingerprints: &bytes[..2],
        };
        assert!(std::panic::catch_unwind(|| truncated.contains(&42)).is_err());
    }

    #[test]
    fn test_initialization() {
        const SAMPLE_SIZE: usize = 1_000_000;
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();

        let filter = Xor16::from(&keys);

        for key in keys {
            assert!(filter.contains(&key));
        }
    }

    #[test]
    fn test_bits_per_entry() {
        const SAMPLE_SIZE: usize = 1_000_000;
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();

        let filter = Xor16::from(&keys);
        let bpe = (filter.len() as f64) * 16.0 / (SAMPLE_SIZE as f64);

        assert!(bpe < 20., "Bits per entry is {}", bpe);
    }

    #[test]
    fn test_false_positives() {
        const SAMPLE_SIZE: usize = 1_000_000;
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();

        let filter = Xor16::from(&keys);

        let false_positives: usize = (0..SAMPLE_SIZE)
            .map(|_| rng.gen())
            .filter(|n| filter.contains(n))
            .count();
        let fp_rate: f64 = (false_positives * 100) as f64 / SAMPLE_SIZE as f64;
        assert!(fp_rate < 0.0025, "False positive rate is {}", fp_rate);
    }
}
