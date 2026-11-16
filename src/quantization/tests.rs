//! 量化、尾块与跨指令集一致性回归。

use super::*;

#[cfg(target_arch = "x86_64")]
fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn test_bq2_signature_and_store_encoding() {
    // 验证二值+强幅度编码精确比特布局与空存储语义（均值(|v|) = 7/7 = 1.0，严格 > 阈值触发强位）
    let sig = Bq2Signature::from_vector(&[0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]);
    assert_eq!(sig.pos, vec![0b0010_1010]);
    assert_eq!(sig.strong, vec![0b0001_1000]);
    assert_eq!(
        Bq2Signature::from_vector(&[0.0f32; 0]).pos,
        Vec::<u64>::new()
    );

    let mut store = Bq2Store::new(4);
    assert!(store.is_empty());
    let v1 = [1.0f32, 1.0, -1.0, -1.0];
    let v2 = [1.0f32, -1.0, 1.0, -1.0];
    store.push_from_vector(&v1);
    store.push_from_vector(&v2);
    assert_eq!(store.len(), 2);
    assert!(!store.is_empty());

    let sig1 = Bq2Signature::from_vector(&v1);
    let sig2 = Bq2Signature::from_vector(&v2);
    let expected = bq2_distance_raw_scalar(
        sig1.pos.as_ptr(),
        sig1.strong.as_ptr(),
        sig2.pos.as_ptr(),
        sig2.strong.as_ptr(),
        4,
    );
    assert_eq!(expected, 16);
    assert_eq!(sig1.distance(&sig2, 4), expected);
    assert_eq!(store.distance(0, 1, 4), expected);
    assert_eq!(store.distance(0, 0, 4), 12);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn dim384专用完整距离与通用标量位级一致() {
    if !is_x86_feature_detected!("popcnt") {
        return;
    }
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..2_048 {
        let mut pos_a = [0u64; 6];
        let mut strong_a = [0u64; 6];
        let mut pos_b = [0u64; 6];
        let mut strong_b = [0u64; 6];
        for index in 0..6 {
            pos_a[index] = next_u64(&mut state);
            strong_a[index] = next_u64(&mut state);
            pos_b[index] = next_u64(&mut state);
            strong_b[index] = next_u64(&mut state);
        }
        let expected = bq2_distance_raw_scalar(
            pos_a.as_ptr(),
            strong_a.as_ptr(),
            pos_b.as_ptr(),
            strong_b.as_ptr(),
            384,
        );
        let actual = unsafe {
            bq2_distance_raw_popcnt_384(
                pos_a.as_ptr(),
                strong_a.as_ptr(),
                pos_b.as_ptr(),
                strong_b.as_ptr(),
            )
        };
        assert_eq!(actual, expected);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn dim384专用廉价距离与通用循环位级一致() {
    if !is_x86_feature_detected!("popcnt") {
        return;
    }
    let mut state = 0xD1B5_4A32_D192_ED03;
    for _ in 0..2_048 {
        let mut pos_a = [0u64; 6];
        let mut strong_a = [0u64; 6];
        let mut pos_b = [0u64; 6];
        let mut strong_b = [0u64; 6];
        for index in 0..6 {
            pos_a[index] = next_u64(&mut state);
            strong_a[index] = next_u64(&mut state);
            pos_b[index] = next_u64(&mut state);
            strong_b[index] = next_u64(&mut state);
        }
        let expected = (0..6)
            .map(|index| {
                (pos_a[index] ^ pos_b[index]).count_ones()
                    + (strong_a[index] ^ strong_b[index]).count_ones()
            })
            .sum::<u32>();
        let actual = unsafe {
            bq2_distance_cheap_popcnt_384(
                pos_a.as_ptr(),
                strong_a.as_ptr(),
                pos_b.as_ptr(),
                strong_b.as_ptr(),
            )
        };
        assert_eq!(actual, expected);
    }
}

#[test]
fn dim384紧凑存储最后节点距离安全() {
    let vectors = (0..7)
        .map(|row| {
            (0..384)
                .map(|column| (((row * 389 + column * 17) % 101) as f32 - 50.0) / 50.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut store = Bq2Store::new(384);
    for vector in &vectors {
        store.push_from_vector(vector);
    }
    let query = Bq2Signature::from_vector(&vectors[0]);
    let pos_flat = store.collect_pos_data();
    let strong_flat = store.collect_strong_data();
    let expected = bq2_distance_raw_scalar(
        pos_flat[36..].as_ptr(),
        strong_flat[36..].as_ptr(),
        query.pos.as_ptr(),
        query.strong.as_ptr(),
        384,
    );
    assert_eq!(store.distance_to_sig(6, &query, 384), expected);
    let expected_cheap = (0..6)
        .map(|index| {
            (pos_flat[36 + index] ^ query.pos[index]).count_ones()
                + (strong_flat[36 + index] ^ query.strong[index]).count_ones()
        })
        .sum::<u32>();
    assert_eq!(store.distance_to_sig_cheap(6, &query, 384), expected_cheap);
}

#[test]
fn 非384维保持通用距离语义() {
    for dim in [383, 385, 512, 768, 1024, 1536] {
        let left = (0..dim)
            .map(|index| (index as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let right = (0..dim)
            .map(|index| (index as f32 * 0.31).cos())
            .collect::<Vec<_>>();
        let left = Bq2Signature::from_vector(&left);
        let right = Bq2Signature::from_vector(&right);
        let expected = bq2_distance_raw_scalar(
            left.pos.as_ptr(),
            left.strong.as_ptr(),
            right.pos.as_ptr(),
            right.strong.as_ptr(),
            dim,
        );
        assert_eq!(
            bq2_distance_raw(
                left.pos.as_ptr(),
                left.strong.as_ptr(),
                right.pos.as_ptr(),
                right.strong.as_ptr(),
                dim,
            ),
            expected
        );
    }
}

#[test]
fn 非完整尾块在所有奇偶chunk组合下保持标量语义() {
    for dim in [1, 2, 63, 65, 127, 129, 191, 193, 255, 257, 319, 383, 385] {
        let left = (0..dim)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let right = (0..dim)
            .map(|index| (index as f32 * 0.29).cos())
            .collect::<Vec<_>>();
        let left = Bq2Signature::from_vector(&left);
        let right = Bq2Signature::from_vector(&right);
        let expected = bq2_distance_raw_scalar(
            left.pos.as_ptr(),
            left.strong.as_ptr(),
            right.pos.as_ptr(),
            right.strong.as_ptr(),
            dim,
        );
        let actual = bq2_distance_raw(
            left.pos.as_ptr(),
            left.strong.as_ptr(),
            right.pos.as_ptr(),
            right.strong.as_ptr(),
            dim,
        );
        assert_eq!(actual, expected, "维度 {dim} 的尾块掩码语义不一致");
    }
}

#[cfg(feature = "research")]
#[test]
fn 合表签名后端保持读取序列化和首次写入一致() {
    for dim in [3, 64, 512, 768, 1536, 4097] {
        let n = 17;
        let vectors: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 17 % 37) as f32 - 18.0) * 0.001)
            .collect();
        let factors: Vec<f64> = vectors
            .chunks_exact(dim)
            .map(|v| 1.0 / (1.0 - crate::metric::norm_squared_f64(v)))
            .collect();
        let slots: Vec<usize> = (0..n).collect();
        let space = crate::graph::NativeSpace {
            vectors: &vectors,
            delta: &[],
            rerank: None,
            boundary_factors: &factors,
            dimensions: dim,
            curvature: 1.0,
        };
        let reference = Bq2Store::from_vectors_parallel(&vectors, dim);
        let mut expected_bytes = Vec::new();
        reference.write_planar_to(&mut expected_bytes).unwrap();
        let appended = &vectors[..dim];
        let signature = Bq2Signature::from_vector(appended);

        for operation in 0..4 {
            let mut store = Bq2Store::from_vectors_parallel(&vectors, dim);
            let shared =
                std::sync::Arc::new(crate::graph::HotSmStore::build(&space, &store, &slots, &[]));
            let weak = std::sync::Arc::downgrade(&shared);
            store.reclaim_for_colocated(shared);
            assert_eq!(store.len(), n);
            assert!(store.interleaved.is_empty());
            assert!(store.legacy_soa.is_none());
            for idx in 0..n {
                assert_eq!(store.pos_slice(idx), reference.pos_slice(idx));
                assert_eq!(store.strong_slice(idx), reference.strong_slice(idx));
                assert_eq!(
                    store.distance_to_sig(idx, &signature, dim),
                    reference.distance_to_sig(idx, &signature, dim),
                );
                store.prefetch_sig(idx);
                store.prefetch_node_exact(idx);
            }
            let mut bytes = Vec::new();
            store.write_planar_to(&mut bytes).unwrap();
            assert_eq!(bytes, expected_bytes);

            match operation {
                0 => store.reserve(1),
                1 => store.push_sig(&signature),
                2 => store.push_from_vector(appended),
                3 => store.set_contiguous_layout(false),
                _ => unreachable!(),
            }
            assert!(weak.upgrade().is_none(), "可变存储恢复后应释放旧合表");
            for idx in 0..n {
                assert_eq!(store.pos_slice(idx), reference.pos_slice(idx));
                assert_eq!(store.strong_slice(idx), reference.strong_slice(idx));
            }
            if operation == 1 || operation == 2 {
                assert_eq!(store.len(), n + 1);
                assert_eq!(store.pos_slice(n), signature.pos.as_slice());
                assert_eq!(store.strong_slice(n), signature.strong.as_slice());
            } else {
                assert_eq!(store.len(), n);
            }
        }
    }
}

#[test]
fn bq2动态签名支持超过三千零七十二维() {
    let dim: usize = 4097;
    let left = (0..dim)
        .map(|i| (i as f32 * 0.13).sin())
        .collect::<Vec<_>>();
    let right = (0..dim)
        .map(|i| (i as f32 * 0.29).cos())
        .collect::<Vec<_>>();
    let left = Bq2Signature::from_vector(&left);
    let right = Bq2Signature::from_vector(&right);
    assert_eq!(left.pos.len(), dim.div_ceil(64));
    assert_eq!(
        left.distance(&right, dim),
        bq2_distance_raw(
            left.pos.as_ptr(),
            left.strong.as_ptr(),
            right.pos.as_ptr(),
            right.strong.as_ptr(),
            dim,
        )
    );
}

#[test]
fn sm五路位计数在任意维度与尾块掩码下与标量严格一致() {
    let mut state = 0x517C_C1B7_2722_0A95u64;
    let mut rand_u64 = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for dim in [
        1usize, 2, 3, 32, 63, 64, 65, 127, 128, 256, 384, 511, 512, 768, 4097,
    ] {
        let chunks = dim.div_ceil(64);
        for _ in 0..64 {
            let pa = (0..chunks).map(|_| rand_u64()).collect::<Vec<_>>();
            let sa = (0..chunks).map(|_| rand_u64()).collect::<Vec<_>>();
            let pb = (0..chunks).map(|_| rand_u64()).collect::<Vec<_>>();
            let sb = (0..chunks).map(|_| rand_u64()).collect::<Vec<_>>();
            let expected =
                sm_dot_counts_raw_scalar(pa.as_ptr(), sa.as_ptr(), pb.as_ptr(), sb.as_ptr(), dim);
            let actual = sm_dot_counts_raw(pa.as_ptr(), sa.as_ptr(), pb.as_ptr(), sb.as_ptr(), dim);
            assert_eq!(actual, expected, "维度 {dim} 的 SM 5-popcount 结果不一致");
        }
    }
}

#[test]
fn sm两级符号幅值内积与显式浮点重构严格一致且零向量安全() {
    let zero = vec![0.0f32; 64];
    let sig_zero = Bq2Signature::from_vector(&zero);
    let (p_zero, r_zero) = SmCodeParams::from_vector_and_sig(&zero, &sig_zero);
    assert_eq!(r_zero, 0.0);
    assert_eq!(p_zero.sigma, 0.0);

    for dim in [2, 63, 64, 128, 384, 768] {
        let va = (0..dim)
            .map(|i| ((i as f32 * 0.37).sin().powi(3)) * 1.7 - 0.15)
            .collect::<Vec<_>>();
        let vb = (0..dim)
            .map(|i| ((i as f32 * 0.23 + 0.5).cos()) * 0.9 + 0.05)
            .collect::<Vec<_>>();
        let mut store = Bq2Store::new(dim);
        store.push_from_vector(&va);
        store.push_from_vector(&vb);

        let (pa, ra) = SmCodeParams::from_vector_and_store(&va, &store, 0);
        let (pb, rb) = SmCodeParams::from_vector_and_store(&vb, &store, 1);
        assert!(ra > 0.0 && rb > 0.0);

        let sa = store.get_sig(0);
        let sb = store.get_sig(1);
        let mut explicit_dot = 0.0f64;
        let mut explicit_norm_a = 0.0f64;
        for j in 0..dim {
            let sign_a = if ((sa.pos[j / 64] >> (j % 64)) & 1) == 1 {
                1.0
            } else {
                -1.0
            };
            let sign_b = if ((sb.pos[j / 64] >> (j % 64)) & 1) == 1 {
                1.0
            } else {
                -1.0
            };
            let ha = ((sa.strong[j / 64] >> (j % 64)) & 1) as f64;
            let hb = ((sb.strong[j / 64] >> (j % 64)) & 1) as f64;
            let qa = sign_a * (f64::from(pa.weak) + ha * f64::from(pa.delta));
            let qb = sign_b * (f64::from(pb.weak) + hb * f64::from(pb.delta));
            explicit_dot += qa * qb;
            explicit_norm_a += qa * qa;
        }
        let counts = store.sm_counts(0, 1, dim);
        let actual_dot = pa.reconstructed_dot(pb, counts, dim);
        assert!(
            (actual_dot - explicit_dot).abs() < 1e-6,
            "dim={dim}: actual={actual_dot}, explicit={explicit_dot}"
        );
        assert!((pa.exact_norm_sq(dim) - explicit_norm_a).abs() < 1e-10);
    }
}

#[test]
fn 并行构建紧凑签名存储与顺序追加逐位一致() {
    for (n, dim) in [
        (0usize, 64usize),
        (17, 63),
        (33, 384),
        (2300, 129),
        (2100, 768),
    ] {
        let vectors: Vec<f32> = (0..n * dim)
            .map(|k| ((k as f32 * 0.173 + 0.31).sin()) * 1.3 - 0.1)
            .collect();
        let par_store = Bq2Store::from_vectors_parallel(&vectors, dim);
        let mut seq_store = Bq2Store::new(dim);
        if dim > 0 {
            for chunk in vectors.chunks_exact(dim) {
                seq_store.push_from_vector(chunk);
            }
        }
        assert_eq!(par_store.len(), seq_store.len());
        assert_eq!(par_store.chunks(), seq_store.chunks());
        assert_eq!(par_store.collect_pos_data(), seq_store.collect_pos_data());
        assert_eq!(
            par_store.collect_strong_data(),
            seq_store.collect_strong_data()
        );
    }
}

#[test]
fn sm768全部微架构位核与查表非对称核逐位严格等价() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand_u64 = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt") {
        for _ in 0..128 {
            let pa = (0..12).map(|_| rand_u64()).collect::<Vec<_>>();
            let sa = (0..12).map(|_| rand_u64()).collect::<Vec<_>>();
            let pb = (0..12).map(|_| rand_u64()).collect::<Vec<_>>();
            let sb = (0..12).map(|_| rand_u64()).collect::<Vec<_>>();
            let expected =
                sm_dot_counts_raw_scalar(pa.as_ptr(), sa.as_ptr(), pb.as_ptr(), sb.as_ptr(), 768);
            unsafe {
                assert_eq!(
                    sm_dot_counts_raw_avx2(pa.as_ptr(), sa.as_ptr(), pb.as_ptr(), sb.as_ptr(), 768),
                    expected
                );
                assert_eq!(
                    sm_dot_counts_768_popcnt_unrolled(
                        pa.as_ptr(),
                        sa.as_ptr(),
                        pb.as_ptr(),
                        sb.as_ptr()
                    ),
                    expected
                );
                assert_eq!(
                    sm_dot_counts_768_popcnt_csa(
                        pa.as_ptr(),
                        sa.as_ptr(),
                        pb.as_ptr(),
                        sb.as_ptr()
                    ),
                    expected
                );
                assert_eq!(
                    sm_dot_counts_768_hybrid_port1_port5(
                        pa.as_ptr(),
                        sa.as_ptr(),
                        pb.as_ptr(),
                        sb.as_ptr()
                    ),
                    expected
                );

                let q_i16: Vec<i16> = (0..768)
                    .map(|j| (((rand_u64() as i32) % 401) - 200) as i16 + (j as i16 % 7))
                    .collect();
                let ref_asym = asym_sm_masked_sums_scalar(&q_i16, &pa, &sa, 12);
                assert_eq!(asym_sm_masked_sums_avx2(&q_i16, &pa, &sa, 12), ref_asym);
                assert_eq!(asym_sm_masked_sums_avx2_lut(&q_i16, &pa, &sa, 12), ref_asym);
            }
        }
    }
}

#[test]
fn 非对称校准sm算子与标量严格一致且自余弦精确归一() {
    let mut asym = AsymQuerySm::new();
    let zero = vec![0.0f32; 128];
    let mut zero_store = Bq2Store::new(128);
    zero_store.push_from_vector(&zero);
    let (p_zero, r_zero) = SmCodeParams::from_vector_and_store(&zero, &zero_store, 0);
    asym.encode_into(&zero, r_zero);
    assert_eq!(asym.calibrated_cosine(&zero_store, 0, p_zero), 0.0);

    // 验证极端稀疏 16-lane 对齐单位向量不会触发 i16 通道溢出
    let mut aligned = vec![0.0f32; 256];
    for m in 0..16 {
        aligned[m * 16] = 0.25;
    }
    let mut aligned_store = Bq2Store::new(256);
    aligned_store.push_from_vector(&aligned);
    let (p_al, r_al) = SmCodeParams::from_vector_and_store(&aligned, &aligned_store, 0);
    asym.encode_into(&aligned, r_al);
    let expected_sums = asym_sm_masked_sums_scalar(
        &asym.q_i16,
        aligned_store.pos_slice(0),
        aligned_store.strong_slice(0),
        aligned_store.chunks(),
    );
    let actual_sums = asym_sm_masked_sums(
        &asym.q_i16,
        aligned_store.pos_slice(0),
        aligned_store.strong_slice(0),
        aligned_store.chunks(),
    );
    assert_eq!(actual_sums, expected_sums);
    let self_cos = asym.calibrated_cosine(&aligned_store, 0, p_al);
    assert!((self_cos - 1.0).abs() < 1e-3, "self_cos={self_cos}");

    // 验证跨维度标量/SIMD 一致性与相对对称 SM 的重构误差显著下降
    for dim in [63usize, 64, 129, 256, 768] {
        let mut store = Bq2Store::new(dim);
        let mut vecs = Vec::new();
        let mut params = Vec::new();
        let mut radii = Vec::new();
        for row in 0..32usize {
            let v: Vec<f32> = (0..dim)
                .map(|j| {
                    (((row * 131 + j * 19) as f32 * 0.11).sin()).powi(3)
                        + 0.15 * ((row + j) as f32 * 0.29).cos()
                })
                .collect();
            store.push_from_vector(&v);
            let (sm, r) = SmCodeParams::from_vector_and_store(&v, &store, row);
            vecs.push(v);
            params.push(sm);
            radii.push(r);
        }
        asym.encode_into(&vecs[0], radii[0]);
        // 自身非对称校准余弦必须极其逼近 1.0
        let c00 = asym.calibrated_cosine(&store, 0, params[0]);
        assert!((c00 - 1.0).abs() < 2e-3, "dim={dim}, c00={c00}");

        let mut err_sq_asym = 0.0f64;
        let mut err_sq_sym = 0.0f64;
        let sig0 = store.get_sig(0);
        for row in 1..32 {
            let s_scalar = asym_sm_masked_sums_scalar(
                &asym.q_i16,
                store.pos_slice(row),
                store.strong_slice(row),
                store.chunks(),
            );
            let s_simd = asym_sm_masked_sums(
                &asym.q_i16,
                store.pos_slice(row),
                store.strong_slice(row),
                store.chunks(),
            );
            assert_eq!(
                s_simd, s_scalar,
                "dim={dim}, row={row} 非对称掩码求和不一致"
            );

            let mut dot = 0.0f64;
            for j in 0..dim {
                dot += f64::from(vecs[0][j]) * f64::from(vecs[row][j]);
            }
            let exact_cos = (dot / (radii[0] * radii[row])).clamp(-1.0, 1.0);
            let asym_cos = asym.calibrated_cosine(&store, row, params[row]);
            let counts = store.sm_counts_to_sig(row, &sig0, dim);
            let sym_cos =
                crate::geometry::sm_calibrated_cosine_fast(params[0], params[row], counts, dim);
            err_sq_asym += (asym_cos - exact_cos).powi(2);
            err_sq_sym += (sym_cos - exact_cos).powi(2);
        }
        assert!(
            err_sq_asym < err_sq_sym,
            "dim={dim}: 非对称误差 {err_sq_asym} 未低于对称误差 {err_sq_sym}"
        );
    }
}
