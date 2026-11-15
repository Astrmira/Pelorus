//! SQ8 预编码方向与 FP32 共享坐标方向。数据库范数只在构建时计算。
use std::cell::RefCell;

enum DirectionData {
    Fp32(Vec<f32>),
    Sq8(Vec<i8>),
}
pub(crate) struct DirectionStore {
    /// None 表示复用调用方的原始 FP32 矩阵。
    sq8: Option<Vec<i8>>,
    inv_norms: Vec<f64>,
    dim: usize,
}
struct DirectionQuery {
    data: DirectionData,
    inv_norm: f64,
}
thread_local! {
    static QUERY: RefCell<Option<DirectionQuery>> = const { RefCell::new(None) };
}

fn encode_sq8(vector: &[f32]) -> (Vec<i8>, f64) {
    let max = vector
        .iter()
        .fold(0.0f64, |a, &v| a.max(f64::from(v).abs()));
    let scale = if max > 0.0 { 127.0 / max } else { 0.0 };
    let data: Vec<i8> = vector
        .iter()
        .map(|&v| (f64::from(v) * scale).round() as i8)
        .collect();
    let norm = data
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    (data, if norm > 0.0 { 1.0 / norm } else { 0.0 })
}

impl DirectionStore {
    #[inline]
    pub(crate) fn prefetch(&self, node: usize, raw: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let (pointer, bytes) = match &self.sq8 {
                Some(v) => (v.as_ptr().add(node * self.dim), self.dim),
                None => (raw.as_ptr().cast::<i8>(), raw.len() * 4),
            };
            _mm_prefetch(pointer, _MM_HINT_T0);
            if bytes > 64 {
                _mm_prefetch(pointer.add(64), _MM_HINT_T0);
            }
            if bytes > 512 {
                _mm_prefetch(pointer.add(512), _MM_HINT_T0);
            }
            if bytes > 1536 {
                _mm_prefetch(pointer.add(1536), _MM_HINT_T0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (node, raw);
        }
    }

    pub(crate) fn is_sq8(&self) -> bool {
        self.sq8.is_some()
    }

    pub(crate) fn build(vectors: &[f32], dim: usize, sq8: bool) -> Self {
        let mut inv_norms = Vec::with_capacity(vectors.len() / dim);
        if sq8 {
            let mut data = Vec::with_capacity(vectors.len());
            for v in vectors.chunks_exact(dim) {
                let (code, inv_norm) = encode_sq8(v);
                data.extend(code);
                inv_norms.push(inv_norm);
            }
            Self {
                sq8: Some(data),
                inv_norms,
                dim,
            }
        } else {
            for v in vectors.chunks_exact(dim) {
                let norm = crate::metric::norm_squared_f64(v).sqrt();
                inv_norms.push(if norm > 0.0 { 1.0 / norm } else { 0.0 });
            }
            Self {
                sq8: None,
                inv_norms,
                dim,
            }
        }
    }

    pub(crate) fn prepare_query(&self, query: &[f32]) {
        let q = if self.sq8.is_some() {
            let (data, inv_norm) = encode_sq8(query);
            DirectionQuery {
                data: DirectionData::Sq8(data),
                inv_norm,
            }
        } else {
            let norm = crate::metric::norm_squared_f64(query).sqrt();
            DirectionQuery {
                data: DirectionData::Fp32(query.to_vec()),
                inv_norm: if norm > 0.0 { 1.0 / norm } else { 0.0 },
            }
        };
        QUERY.with(|slot| *slot.borrow_mut() = Some(q));
    }

    pub(crate) fn pair_cosine(&self, a: usize, b: usize, raw_a: &[f32], raw_b: &[f32]) -> f64 {
        let dot = match &self.sq8 {
            Some(v) => dot_i8(
                &v[a * self.dim..(a + 1) * self.dim],
                &v[b * self.dim..(b + 1) * self.dim],
            ),
            None => dot_f32(raw_a, raw_b),
        };
        (dot * self.inv_norms[a] * self.inv_norms[b]).clamp(-1.0, 1.0)
    }

    pub(crate) fn query_cosine(&self, node: usize, raw: &[f32]) -> f64 {
        QUERY.with(|slot| {
            let q = slot.borrow();
            let q = q.as_ref().expect("方向查询必须先编码");
            let dot = match (&self.sq8, &q.data) {
                (None, DirectionData::Fp32(q)) => dot_f32(raw, q),
                (Some(v), DirectionData::Sq8(q)) => {
                    dot_i8(&v[node * self.dim..(node + 1) * self.dim], q)
                }
                _ => unreachable!("方向编码类型必须一致"),
            };
            (dot * self.inv_norms[node] * q.inv_norm).clamp(-1.0, 1.0)
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.inv_norms.len() * 8 + self.sq8.as_ref().map_or(0, Vec::len)
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        if is_x86_feature_detected!("fma") {
            return unsafe { dot_f32_fma(a, b) };
        }
        return unsafe { dot_f32_avx2(a, b) };
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_fma(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 16 <= a.len() {
            acc0 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr().add(i))),
                _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(i))),
                acc0,
            );
            acc1 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr().add(i + 4))),
                _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(i + 4))),
                acc1,
            );
            acc2 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr().add(i + 8))),
                _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(i + 8))),
                acc2,
            );
            acc3 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr().add(i + 12))),
                _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(i + 12))),
                acc3,
            );
            i += 16;
        }
        let acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        let mut lanes = [0.0; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
        lanes.iter().sum::<f64>()
            + a[i..]
                .iter()
                .zip(&b[i..])
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>()
    }
}
fn dot_i8(a: &[i8], b: &[i8]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if a.len() <= 65536 && is_x86_feature_detected!("avx2") {
        return unsafe { dot_i8_avx2(a, b) };
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum::<i64>() as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_pd();
        let mut i = 0;
        while i + 4 <= a.len() {
            let x = _mm256_cvtps_pd(_mm_loadu_ps(a.as_ptr().add(i)));
            let y = _mm256_cvtps_pd(_mm_loadu_ps(b.as_ptr().add(i)));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(x, y));
            i += 4;
        }
        let mut lanes = [0.0; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
        lanes.iter().sum::<f64>()
            + a[i..]
                .iter()
                .zip(&b[i..])
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> f64 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 16 <= a.len() {
            let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i).cast()));
            let y = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i).cast()));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
            i += 16;
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc);
        (lanes.iter().map(|&x| i64::from(x)).sum::<i64>()
            + a[i..]
                .iter()
                .zip(&b[i..])
                .map(|(&x, &y)| i64::from(x) * i64::from(y))
                .sum::<i64>()) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn 预编码方向与标量参考在尾块和大维度一致() {
        for dim in [1, 3, 63, 768, 65536, 65537] {
            let a: Vec<f32> = (0..dim)
                .map(|i| ((i * 17 % 257) as f32 - 128.0) * 0.003)
                .collect();
            let b: Vec<f32> = (0..dim)
                .map(|i| ((i * 31 % 251) as f32 - 125.0) * 0.002)
                .collect();
            for sq8 in [false, true] {
                let vectors: Vec<_> = a.iter().chain(&b).copied().collect();
                let store = DirectionStore::build(&vectors, dim, sq8);
                store.prepare_query(&a);
                assert!(
                    (store.pair_cosine(0, 1, &a, &b) - store.query_cosine(1, &b)).abs() < 1e-12
                );
                assert_eq!(store.bytes(), if sq8 { dim * 2 + 16 } else { 16 });
                let reference = match &store.sq8 {
                    None => a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| f64::from(x) * f64::from(y))
                        .sum::<f64>(),
                    Some(v) => v[..dim]
                        .iter()
                        .zip(&v[dim..])
                        .map(|(&x, &y)| i64::from(x) * i64::from(y))
                        .sum::<i64>() as f64,
                } * store.inv_norms[0]
                    * store.inv_norms[1];
                assert!(
                    (store.pair_cosine(0, 1, &a, &b) - reference.clamp(-1.0, 1.0)).abs() < 1e-12
                );
            }
        }
    }
}
