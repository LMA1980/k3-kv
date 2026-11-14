//! Vector-aligned quantization and zero-allocation pack/unpack helpers for `F32`, `F16`, and `Q8_0` KV planes.
//!
//! All KV slabs and per-layer K/V planes align to [`VECTOR_PLANE_ALIGNMENT`] (`128` bytes, matching
//! `SpacemiT` K3 `A100` `VLEN = 1024`, `vlenb = 128`).

use jero_core::{KV_Q8_0_BLOCK_BYTES, KV_Q8_0_BLOCK_ELEMENTS, KvDtype};

use crate::holder::KvError;

/// Canonical 128-byte vector-plane alignment (`vlenb = 128` on `A100` `VLEN = 1024`).
pub const VECTOR_PLANE_ALIGNMENT: usize = 128;

/// Converts an IEEE-754 `f32` scalar to IEEE-754 binary16 (`u16` bits) with round-to-nearest-even.
#[must_use]
pub const fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "8-bit masked exponent is in 0..=255"
    )]
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        if mant == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mant >> 13) as u16) & 0x03ff;
        return sign | 0x7e00 | if payload == 0 { 1 } else { payload };
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }

    if unbiased >= -14 {
        let mut half_exp = unbiased + 15;
        let round_bias = 0x0fff + ((mant >> 13) & 1);
        let rounded = mant + round_bias;
        let half_mant = if (rounded & 0x0080_0000) != 0 {
            half_exp += 1;
            0
        } else {
            (rounded >> 13) & 0x03ff
        };
        if half_exp >= 31 {
            return sign | 0x7c00;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "half_exp is bounded to 1..=30 and half_mant fits in 10 bits"
        )]
        return sign | ((half_exp as u16) << 10) | (half_mant as u16);
    }

    // Subnormal or underflow in binary16.
    if unbiased < -25 || exp == 0 {
        return sign;
    }

    let sig = 0x0080_0000 | mant;
    #[expect(
        clippy::cast_sign_loss,
        reason = "unbiased is bounded in -25..=-15, so shift is in 14..=24"
    )]
    let shift = (-14 - unbiased + 13) as u32;
    let half_step = (1_u32 << (shift - 1)) - 1;
    let round_bit = (sig >> shift) & 1;
    let sub_mant = (sig + half_step + round_bit) >> shift;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "sub_mant is bounded in 0..=0x0400 (seamless normal promotion at 0x0400)"
    )]
    return sign | (sub_mant as u16);
}

/// Converts IEEE-754 binary16 (`u16` bits) to an exact `f32` scalar.
#[must_use]
pub const fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut f32_exp = 113_u32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            f32_exp -= 1;
        }
        let norm_mant = (m & 0x03ff) << 13;
        return f32::from_bits(sign | (f32_exp << 23) | norm_mant);
    }

    if exp == 31 {
        if mant == 0 {
            return f32::from_bits(sign | 0x7f80_0000);
        }
        return f32::from_bits(sign | 0x7fc0_0000 | (mant << 13));
    }

    let f32_exp = exp + 112;
    f32::from_bits(sign | (f32_exp << 23) | (mant << 13))
}

/// Packs a single `32`-element `f32` slice into a `34`-byte `Q8_0` block (`f16` scale + `[i8; 32]`).
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if `src.len() != 32`, `dst.len() != 34`, or `src`
/// contains non-finite values.
pub fn quantize_q8_0_block(src: &[f32], dst: &mut [u8]) -> Result<(), KvError> {
    if src.len() != KV_Q8_0_BLOCK_ELEMENTS || dst.len() != KV_Q8_0_BLOCK_BYTES {
        return Err(KvError::InvalidDimensions(format!(
            "Q8_0 block pack expects ({KV_Q8_0_BLOCK_ELEMENTS} f32, {KV_Q8_0_BLOCK_BYTES} bytes), got ({}, {})",
            src.len(),
            dst.len()
        )));
    }

    let mut max_abs = 0.0_f32;
    for &val in src {
        if !val.is_finite() {
            return Err(KvError::InvalidDimensions(
                "Q8_0 KV pack rejected non-finite f32 value".to_owned(),
            ));
        }
        max_abs = max_abs.max(val.abs());
    }

    if max_abs == 0.0 {
        dst.fill(0);
        return Ok(());
    }

    let scale_bits = f32_to_f16(max_abs / 127.0);
    let effective_scale = f16_to_f32(scale_bits);
    if effective_scale == 0.0 || !effective_scale.is_finite() {
        dst.fill(0);
        return Ok(());
    }

    dst[..2].copy_from_slice(&scale_bits.to_le_bytes());
    let inv_scale = 1.0_f32 / effective_scale;
    for (idx, &val) in src.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "value is clamped to [-127.0, 127.0] before integer conversion"
        )]
        let q_i8 = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
        dst[2 + idx] = q_i8.to_ne_bytes()[0];
    }

    Ok(())
}

/// Unpacks a `34`-byte `Q8_0` block (`f16` scale + `[i8; 32]`) into a `32`-element `f32` slice.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if `src.len() != 34` or `dst.len() != 32`.
pub fn dequantize_q8_0_block(src: &[u8], dst: &mut [f32]) -> Result<(), KvError> {
    if src.len() != KV_Q8_0_BLOCK_BYTES || dst.len() != KV_Q8_0_BLOCK_ELEMENTS {
        return Err(KvError::InvalidDimensions(format!(
            "Q8_0 block unpack expects ({KV_Q8_0_BLOCK_BYTES} bytes, {KV_Q8_0_BLOCK_ELEMENTS} f32), got ({}, {})",
            src.len(),
            dst.len()
        )));
    }

    let scale_bits = u16::from_le_bytes([src[0], src[1]]);
    let scale = f16_to_f32(scale_bits);
    for (idx, &raw_byte) in src[2..].iter().enumerate() {
        let q = i8::from_ne_bytes([raw_byte]);
        dst[idx] = scale * f32::from(q);
    }
    Ok(())
}

/// Packs one K or V position vector (`src.len() == kv_dim`) into `dst` according to `dtype`.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] on dimension/buffer-length mismatch, unaligned `Q8_0`
/// `kv_dim`, or non-finite inputs under `Q8_0`.
pub fn pack_kv_row(dtype: KvDtype, src: &[f32], dst: &mut [u8]) -> Result<(), KvError> {
    let expected_bytes = dtype
        .row_bytes(src.len())
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    if dst.len() != expected_bytes {
        return Err(KvError::InvalidDimensions(format!(
            "pack_kv_row({dtype:?}) expected dst len {expected_bytes}, got {}",
            dst.len()
        )));
    }

    match dtype {
        KvDtype::F32 => {
            for (val, chunk) in src.iter().zip(dst.chunks_exact_mut(4)) {
                chunk.copy_from_slice(&val.to_le_bytes());
            }
            Ok(())
        }
        KvDtype::F16 => {
            for (&val, chunk) in src.iter().zip(dst.chunks_exact_mut(2)) {
                chunk.copy_from_slice(&f32_to_f16(val).to_le_bytes());
            }
            Ok(())
        }
        KvDtype::Q8_0 => {
            for (src_blk, dst_blk) in src
                .chunks_exact(KV_Q8_0_BLOCK_ELEMENTS)
                .zip(dst.chunks_exact_mut(KV_Q8_0_BLOCK_BYTES))
            {
                quantize_q8_0_block(src_blk, dst_blk)?;
            }
            Ok(())
        }
    }
}

/// Unpacks one K or V position vector (`dst.len() == kv_dim`) from `src` according to `dtype`.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] on dimension/buffer-length mismatch or unaligned `Q8_0` `kv_dim`.
pub fn unpack_kv_row(dtype: KvDtype, src: &[u8], dst: &mut [f32]) -> Result<(), KvError> {
    let expected_bytes = dtype
        .row_bytes(dst.len())
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    if src.len() != expected_bytes {
        return Err(KvError::InvalidDimensions(format!(
            "unpack_kv_row({dtype:?}) expected src len {expected_bytes}, got {}",
            src.len()
        )));
    }

    match dtype {
        KvDtype::F32 => {
            for (chunk, out) in src.chunks_exact(4).zip(dst.iter_mut()) {
                *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        KvDtype::F16 => {
            for (chunk, out) in src.chunks_exact(2).zip(dst.iter_mut()) {
                *out = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            Ok(())
        }
        KvDtype::Q8_0 => {
            for (src_blk, dst_blk) in src
                .chunks_exact(KV_Q8_0_BLOCK_BYTES)
                .zip(dst.chunks_exact_mut(KV_Q8_0_BLOCK_ELEMENTS))
            {
                dequantize_q8_0_block(src_blk, dst_blk)?;
            }
            Ok(())
        }
    }
}

/// Packs `seq_len = src_batch.len() / kv_dim` consecutive rows into `dst_plane` starting at
/// `start_pos`, returning `start_pos + seq_len`.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if `kv_dim == 0`, `src_batch.len()` is not a multiple of
/// `kv_dim`, `Q8_0` `kv_dim` is not a multiple of `32`, or `dst_plane` lacks capacity for
/// `(start_pos + seq_len) * row_bytes`.
pub fn pack_kv_batch_plane(
    dtype: KvDtype,
    kv_dim: usize,
    start_pos: usize,
    src_batch: &[f32],
    dst_plane: &mut [u8],
) -> Result<usize, KvError> {
    let row_bytes = dtype
        .row_bytes(kv_dim)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    if src_batch.len() % kv_dim != 0 {
        return Err(KvError::InvalidDimensions(format!(
            "src_batch len ({}) is not a multiple of kv_dim ({kv_dim})",
            src_batch.len()
        )));
    }
    let seq_len = src_batch.len() / kv_dim;
    let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
        KvError::InvalidDimensions("pack_kv_batch_plane end_pos overflowed usize".to_owned())
    })?;
    let byte_start = start_pos
        .checked_mul(row_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("byte_start overflowed usize".to_owned()))?;
    let byte_end = end_pos
        .checked_mul(row_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("byte_end overflowed usize".to_owned()))?;
    if byte_end > dst_plane.len() {
        return Err(KvError::InvalidDimensions(format!(
            "packed span [{byte_start}..{byte_end}) exceeds dst_plane len ({})",
            dst_plane.len()
        )));
    }

    for (rel, (src_row, dst_row)) in src_batch
        .chunks_exact(kv_dim)
        .zip(dst_plane[byte_start..byte_end].chunks_exact_mut(row_bytes))
        .enumerate()
    {
        pack_kv_row(dtype, src_row, dst_row).map_err(|err| {
            KvError::InvalidDimensions(format!(
                "pack_kv_batch_plane row pos={} failed: {err}",
                start_pos + rel
            ))
        })?;
    }
    Ok(end_pos)
}

/// Zero-allocation unpack of `[start_pos..end_pos)` rows from `src_plane` into `dst_out`,
/// returning total `f32` elements written (`(end_pos - start_pos) * kv_dim`).
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if bounds or `dst_out` length are insufficient.
pub fn unpack_kv_range_plane(
    dtype: KvDtype,
    kv_dim: usize,
    start_pos: usize,
    end_pos: usize,
    src_plane: &[u8],
    dst_out: &mut [f32],
) -> Result<usize, KvError> {
    if start_pos > end_pos {
        return Err(KvError::InvalidDimensions(format!(
            "start_pos ({start_pos}) > end_pos ({end_pos})"
        )));
    }
    let row_bytes = dtype
        .row_bytes(kv_dim)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let seq_len = end_pos - start_pos;
    let total_elems = seq_len.checked_mul(kv_dim).ok_or_else(|| {
        KvError::InvalidDimensions("unpack_kv_range_plane elem count overflowed".to_owned())
    })?;
    if dst_out.len() < total_elems {
        return Err(KvError::InvalidDimensions(format!(
            "dst_out len ({}) smaller than required ({total_elems}) f32 elements",
            dst_out.len()
        )));
    }
    let byte_start = start_pos
        .checked_mul(row_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("byte_start overflowed usize".to_owned()))?;
    let byte_end = end_pos
        .checked_mul(row_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("byte_end overflowed usize".to_owned()))?;
    if byte_end > src_plane.len() {
        return Err(KvError::InvalidDimensions(format!(
            "unpack span [{byte_start}..{byte_end}) exceeds src_plane len ({})",
            src_plane.len()
        )));
    }

    if total_elems == 0 {
        return Ok(0);
    }

    for (src_row, dst_row) in src_plane[byte_start..byte_end]
        .chunks_exact(row_bytes)
        .zip(dst_out[..total_elems].chunks_exact_mut(kv_dim))
    {
        unpack_kv_row(dtype, src_row, dst_row)?;
    }

    Ok(total_elems)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let xf = f64::from(x);
            let yf = f64::from(y);
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        if na == 0.0 && nb == 0.0 {
            return 1.0;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom == 0.0 {
            return 0.0;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test diagnostic helper bounds cosine similarity in [-1.0, 1.0]"
        )]
        let sim = (dot / denom) as f32;
        sim
    }

    #[test]
    fn test_f16_special_and_subnormal_bit_vectors() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65536.0), 0x7c00); // overflow -> +Inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());

        // Smallest positive f16 subnormal 2^-24 == 0x0001 (exact f32 bits avoid Miri powi ULP jitter).
        let min_sub = f32::from_bits(0x3380_0000);
        assert_eq!(f32_to_f16(min_sub), 0x0001);
        assert_eq!(f16_to_f32(0x0001).to_bits(), min_sub.to_bits());

        // Exact halfway 2^-25 rounds to even (0), while 2^-25 + 1 ULP rounds up to 0x0001.
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0000)), 0x0000);
        assert_eq!(f32_to_f16(f32::from_bits(0x3300_0001)), 0x0001);

        for bits in [
            0x0000_u16, 0x0001, 0x0200, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x4500, 0x7bff,
        ] {
            let f = f16_to_f32(bits);
            assert_eq!(f32_to_f16(f), bits);
        }
    }

    #[test]
    fn test_f16_and_q8_0_row_pack_unpack_accuracy_miri() {
        let mut src = [0.0_f32; 32];
        for (i, slot) in src.iter_mut().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            #[expect(clippy::cast_precision_loss, reason = "i is in 0..32")]
            let mag = (i as f32 + 1.0) * 0.045;
            *slot = sign * mag;
        }

        let mut f16_buf = [0_u8; 64];
        let mut f16_out = [0.0_f32; 32];
        pack_kv_row(KvDtype::F16, &src, &mut f16_buf).expect("pack F16");
        unpack_kv_row(KvDtype::F16, &f16_buf, &mut f16_out).expect("unpack F16");

        let f16_max_abs = src
            .iter()
            .zip(f16_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let f16_cos_drift = 1.0 - cosine_similarity(&src, &f16_out);
        assert!(f16_max_abs < 1e-3, "f16 max_abs {f16_max_abs}");
        assert!(f16_cos_drift < 1e-3, "f16 cos_drift {f16_cos_drift}");

        let mut q8_buf = [0_u8; 34];
        let mut q8_out = [0.0_f32; 32];
        pack_kv_row(KvDtype::Q8_0, &src, &mut q8_buf).expect("pack Q8_0");
        unpack_kv_row(KvDtype::Q8_0, &q8_buf, &mut q8_out).expect("unpack Q8_0");

        let q8_max_abs = src
            .iter()
            .zip(q8_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let q8_cos_drift = 1.0 - cosine_similarity(&src, &q8_out);
        assert!(q8_max_abs < 1.5e-2, "q8 max_abs {q8_max_abs}");
        assert!(q8_cos_drift < 1e-2, "q8 cos_drift {q8_cos_drift}");

        // Reject zero, mismatched, unaligned Q8_0, and non-finite inputs cleanly.
        assert!(pack_kv_row(KvDtype::Q8_0, &src[..16], &mut q8_buf[..17]).is_err());
        let mut bad_src = src;
        bad_src[3] = f32::NAN;
        assert!(pack_kv_row(KvDtype::Q8_0, &bad_src, &mut q8_buf).is_err());
    }
}
