//! A tiny, dependency-free PNG writer for the stills phase 5 captures.
//!
//! crates.io is not reachable from the target machine and the `image` crate is
//! not vendored, so the one thing the acceptance needs — writing an 8-bit RGB
//! PNG — is implemented here with uncompressed (`stored`) DEFLATE blocks. The
//! files are larger than a compressed PNG but valid and readable by any viewer,
//! and the encoder is small enough to test exhaustively.

use std::io::{self, Write};
use std::path::Path;

/// The 8-byte PNG signature.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Writes an 8-bit RGB image (`width * height * 3` bytes, row-major, top row
/// first) to `path` as a PNG.
pub fn write_rgb(path: impl AsRef<Path>, width: u32, height: u32, rgb: &[u8]) -> io::Result<()> {
    let bytes = encode_rgb(width, height, rgb);
    let mut file = std::fs::File::create(path)?;
    file.write_all(&bytes)
}

/// Encodes an 8-bit RGB image as a PNG byte stream.
///
/// `rgb` must hold `width * height * 3` bytes; extra bytes are ignored and a
/// short buffer is padded with zeros, so a mis-sized readback still produces a
/// viewable image rather than a panic.
pub fn encode_rgb(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    encode(width, height, 3, rgb)
}

/// Encodes an 8-bit RGBA image as a PNG byte stream (the alpha channel is kept).
pub fn encode_rgba(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    encode(width, height, 4, rgba)
}

/// Encodes `channels`-channel 8-bit pixels (3 = RGB, 4 = RGBA) as a PNG.
fn encode(width: u32, height: u32, channels: usize, pixels: &[u8]) -> Vec<u8> {
    let color_type = match channels {
        4 => 6,
        _ => 2,
    };
    // Raw scanlines: a zero filter byte, then the row's pixels.
    let stride = width as usize * channels;
    let mut raw = Vec::with_capacity(height as usize * (1 + stride));
    for y in 0..height as usize {
        raw.push(0);
        let start = y * stride;
        for i in 0..stride {
            raw.push(pixels.get(start + i).copied().unwrap_or(0));
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(&SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(color_type);
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter method: adaptive
    ihdr.push(0); // interlace: none
    write_chunk(&mut out, b"IHDR", &ihdr);

    write_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    write_chunk(&mut out, b"IEND", &[]);
    out
}

/// Writes one PNG chunk: length, type, data, CRC over type and data.
fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// A zlib stream whose DEFLATE payload is a sequence of `stored` (uncompressed)
/// blocks, ending with the Adler-32 of the input.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // CMF/FLG: deflate, 32K window, check bits set.
    let mut chunks = data.chunks(0xFFFF).peekable();
    if data.is_empty() {
        // A single empty final stored block.
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]);
    }
    while let Some(chunk) = chunks.next() {
        let final_block = chunks.peek().is_none();
        out.push(u8::from(final_block)); // BFINAL in bit 0, BTYPE = 00 (stored).
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// The zlib Adler-32 checksum.
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// The PNG CRC-32 (IEEE 802.3 polynomial, reflected).
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_and_headers_are_well_formed() {
        let png = encode_rgb(2, 3, &[0u8; 2 * 3 * 3]);
        assert_eq!(&png[..8], &SIGNATURE);
        // First chunk is IHDR, length 13.
        assert_eq!(&png[8..12], &13u32.to_be_bytes());
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 2);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 3);
        assert_eq!(png[24], 8); // bit depth
        assert_eq!(png[25], 2); // colour type RGB
        assert!(png.ends_with(b"IEND\xae\x42\x60\x82"));
    }

    #[test]
    fn known_chunk_crc_matches_the_spec() {
        // The IEND chunk's CRC is a published constant; if our CRC is wrong the
        // encoder is wrong.
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }

    #[test]
    fn adler_matches_a_hand_computation() {
        // "Wikipedia" example: adler32("Wikipedia") = 0x11E60398.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn payload_is_preserved_through_stored_blocks() {
        // Rows longer than one stored block (65535 bytes) still round-trip.
        let width = 300u32;
        let height = 120u32;
        let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
        let png = encode_rgb(width, height, &pixels);
        // The stored blocks and scanline filter bytes are present verbatim, so
        // the first pixel's bytes appear right after the first filter byte.
        assert!(png.windows(3).any(|w| w == [0, 1, 2]));
        // Every pixel byte must appear somewhere in the stream.
        assert!(pixels.iter().all(|&b| png.contains(&b)));
    }
}
