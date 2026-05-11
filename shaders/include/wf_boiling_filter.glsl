// ============================================================
// wf_boiling_filter.glsl — RTXDI boiling filter (GLSL port)
//
// Port of NVIDIA RTXDI-Library/Include/Rtxdi/Utils/BoilingFilter.hlsli
// and the GI/DI wrappers (Rtxdi/GI/BoilingFilter.hlsli,
// Rtxdi/DI/BoilingFilter.hlsli), called by the temporal kernels after
// reservoir finalize.
//
// Reservoir-agnostic: takes a scalar proxy weight. DI passes
//   targetPdf · W; GI passes luminance(Lo) · W.
//
// Algorithm:
//   "Boiling" happens when spatial / temporal reuse latches onto an
//   unlikely bright sample that propagates through the neighbourhood,
//   visible as a flash / halo. Detect it by:
//     1. Computing each reservoir's proxy weight.
//     2. Reducing within the 8×8 workgroup to a non-zero average.
//     3. Flagging reservoirs whose weight > average × threshold.
//     4. The caller resets flagged reservoirs to empty.
//
// threshold = (10 / filterStrength) − 9. With filterStrength = 0.2 →
// threshold = 41; samples > 41× the tile's mean non-zero weight get
// dropped. Tighter strength = more aggressive clipping. RTXPT defaults
// to 0.2 for DI and 0.35 (multiplier ≈19.6) for GI.
//
// Requires:
//   - workgroup size 8×8 = 64 threads (matches wf_di_* and wf_gi_*)
//   - subgroup size ≥ 32 (shared memory sized for worst-case 2 waves)
//   - GL_KHR_shader_subgroup_{basic,arithmetic,ballot}
//   - caller guarantees ALL 64 threads participate (no early-return
//     before the filter — use an `inBounds` flag with weight=0)
// ============================================================

#ifndef WF_BOILING_FILTER_GLSL
#define WF_BOILING_FILTER_GLSL

#define WF_BF_GROUP_SIZE            8u
#define WF_BF_GROUP_THREADS         (WF_BF_GROUP_SIZE * WF_BF_GROUP_SIZE)
#define WF_BF_MIN_LANE_COUNT        32u
#define WF_BF_WAVES_MAX             ((WF_BF_GROUP_THREADS + WF_BF_MIN_LANE_COUNT - 1u) / WF_BF_MIN_LANE_COUNT)

shared float wf_bf_s_weights[WF_BF_WAVES_MAX];
shared uint  wf_bf_s_count  [WF_BF_WAVES_MAX];

// Returns true if the reservoir's `reservoirWeight` is an outlier vs.
// the 8×8 tile non-zero average. The caller (temporal kernel) resets
// the reservoir to empty when this returns true.
//
// MUST be called by EVERY thread in the workgroup (64 threads). Use
// reservoirWeight = 0 for out-of-bounds / sky / invalid pixels so
// they participate in the reduction without skewing the average.
bool wf_boiling_filter(uvec2 LocalIndex, float filterStrength, float reservoirWeight) {
    float boilingFilterMultiplier = 10.0 / clamp(filterStrength, 1e-6, 1.0) - 9.0;

    // Wave-local reduction: sum of weights and count of non-zero lanes.
    float waveWeight = subgroupAdd(reservoirWeight);
    uvec4 nonZeroBallot = subgroupBallot(reservoirWeight > 0.0);
    uint  waveCount    = subgroupBallotBitCount(nonZeroBallot);

    // First lane of each wave writes the wave-local result to shared memory.
    uint linearThreadIndex = LocalIndex.x + LocalIndex.y * WF_BF_GROUP_SIZE;
    uint waveIndex         = linearThreadIndex / gl_SubgroupSize;
    if (subgroupElect()) {
        wf_bf_s_weights[waveIndex] = waveWeight;
        wf_bf_s_count  [waveIndex] = waveCount;
    }

    barrier();
    memoryBarrierShared();

    // Second reduction: a single wave reduces the per-wave entries
    // into a single group-wide non-zero average stored at slot 0.
    uint numWaves = (WF_BF_GROUP_THREADS + gl_SubgroupSize - 1u) / gl_SubgroupSize;
    if (linearThreadIndex < numWaves) {
        waveWeight = wf_bf_s_weights[linearThreadIndex];
        waveCount  = wf_bf_s_count  [linearThreadIndex];

        waveWeight = subgroupAdd(waveWeight);
        waveCount  = subgroupAdd(waveCount);

        if (linearThreadIndex == 0u) {
            wf_bf_s_weights[0] = (waveCount > 0u)
                ? (waveWeight / float(waveCount))
                : 0.0;
        }
    }

    barrier();
    memoryBarrierShared();

    float averageNonzeroWeight = wf_bf_s_weights[0];
    return reservoirWeight > averageNonzeroWeight * boilingFilterMultiplier;
}

#endif // WF_BOILING_FILTER_GLSL
