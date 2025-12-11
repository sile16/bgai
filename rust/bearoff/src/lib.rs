//! Bearoff generator matching gnubg's makebearoff two-sided logic.
//!
//! Version 2: Tiered storage with dual perspectives.
//!
//! - Uses gnubg's bearoff ID ordering (PositionBearoff).
//! - Exact CubeEquity rule (double/take/pass) on cubeless equities.
//! - Stores BOTH perspectives (i,j) and (j,i) for each upper triangle entry.
//! - Tiered storage:
//!   - Tier 1 (win-only): Both players have borne off ≥1 checker (no gammons possible)
//!   - Tier 2 (gammon-possible): At least one player has all checkers on board
//! - All values stored as uint24 (3 bytes, little endian).
//!
//! Tier 1 layout per entry (10 values = 30 bytes):
//!   [win_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
//!    win_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]
//!
//! Tier 2 layout per entry (14 values = 42 bytes):
//!   [win_ij, gam_win_ij, gam_loss_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
//!    win_ji, gam_win_ji, gam_loss_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]

use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use serde::Serialize;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

const NUM_POINTS: usize = 6;
const EQUITY_P1: i16 = 0x7FFF; // gnubg: +1.0
const EQUITY_M1: i16 = !EQUITY_P1; // gnubg: -1.0

const DICE_OUTCOMES: [(u8, u8); 21] = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6),
    (4, 5), (4, 6),
    (5, 6),
];

type Position = [u8; NUM_POINTS];

fn combination(n: usize, r: usize) -> usize {
    if r == 0 || r == n {
        return 1;
    }
    let r = r.min(n - r);
    let mut num = 1usize;
    let mut den = 1usize;
    for i in 0..r {
        num *= n - i;
        den *= i + 1;
    }
    num / den
}

fn position_from_bearoff(id: usize, max_checkers: usize) -> Position {
    fn position_inv(id: usize, n: usize, r: usize) -> usize {
        if r == 0 {
            return 0;
        }
        if n == r {
            return (1usize << n) - 1;
        }
        let nc = combination(n - 1, r);
        if id >= nc {
            (1usize << (n - 1)) | position_inv(id - nc, n - 1, r - 1)
        } else {
            position_inv(id, n - 1, r)
        }
    }

    let mut pos = [0u8; NUM_POINTS];
    let fbits = position_inv(id, max_checkers + NUM_POINTS, NUM_POINTS);
    let mut j = NUM_POINTS - 1;
    for i in 0..(max_checkers + NUM_POINTS) {
        if (fbits >> i) & 1 == 1 {
            if j == 0 {
                break;
            }
            j -= 1;
        } else {
            pos[j] += 1;
        }
    }
    pos
}

fn position_sum(pos: &Position) -> usize {
    pos.iter().map(|&x| x as usize).sum()
}

fn highest_point(pos: &Position) -> Option<usize> {
    (0..NUM_POINTS).rev().find(|&i| pos[i] > 0)
}

fn legal_move(pos: &Position, src: usize, roll: u8) -> bool {
    let dest = src as isize - roll as isize;
    if dest >= 0 {
        return true;
    }
    if let Some(n_back) = highest_point(pos) {
        n_back < NUM_POINTS && (src == n_back || dest == -1)
    } else {
        false
    }
}

fn apply_sub_move(pos: &Position, src: usize, roll: u8) -> Option<Position> {
    let dest = src as isize - roll as isize;
    if src >= NUM_POINTS || dest >= src as isize || pos[src] == 0 {
        return None;
    }
    let mut next = *pos;
    next[src] -= 1;
    if dest >= 0 {
        next[dest as usize] += 1;
    }
    Some(next)
}

fn generate_moves_sub(
    pos: &Position,
    rolls: &[u8; 4],
    depth: usize,
    start: usize,
    allow_partial: bool,
    out: &mut Vec<Position>,
) -> bool {
    if depth > 3 || rolls[depth] == 0 {
        return true;
    }

    let mut used = false;
    for src in (0..=start.min(NUM_POINTS - 1)).rev() {
        if pos[src] == 0 || !legal_move(pos, src, rolls[depth]) {
            continue;
        }
        used = true;
        if let Some(next) = apply_sub_move(pos, src, rolls[depth]) {
            let next_start = if rolls[0] == rolls[1] { src } else { NUM_POINTS - 1 };
            if generate_moves_sub(&next, rolls, depth + 1, next_start, allow_partial, out) {
                out.push(next);
            }
        }
    }

    !used || allow_partial
}

fn generate_moves(pos: &Position, d0: u8, d1: u8) -> Vec<Position> {
    let mut results = Vec::new();

    let rolls1 = [d0, d1, if d0 == d1 { d0 } else { 0 }, if d0 == d1 { d0 } else { 0 }];
    generate_moves_sub(pos, &rolls1, 0, NUM_POINTS - 1, false, &mut results);

    if d0 != d1 {
        let rolls2 = [d1, d0, 0, 0];
        generate_moves_sub(pos, &rolls2, 0, NUM_POINTS - 1, false, &mut results);
    }

    results.sort();
    results.dedup();
    if results.is_empty() {
        results.push(*pos);
    }
    results
}

fn build_pos_index(positions: &[Position]) -> HashMap<Position, usize> {
    positions.iter().enumerate().map(|(i, p)| (*p, i)).collect()
}

fn packed_index(i: usize, j: usize, n: usize) -> usize {
    // i <= j
    i * n - (i * (i - 1)) / 2 + (j - i)
}

fn pair_index(us: usize, them: usize, n: usize) -> usize {
    us * n + them
}

fn negate_equity(v: i16) -> i16 {
    !v
}

fn cube_equity(nd: i16, dt: i16, dp: i16) -> i16 {
    if dt >= nd / 2 && dp >= nd {
        if dt >= dp / 2 { dp } else { 2 * dt }
    } else {
        nd
    }
}

/// Result from solve_equity_full: [cubeless, owner, center, opponent]
/// Plus gammon probabilities for positions where gammons are possible.
#[derive(Clone, Copy, Default)]
struct EquityResult {
    cubeless: i16,
    owner: i16,
    center: i16,
    opponent: i16,
    gammon_win: f32,   // P(win gammon) - only non-zero if opponent has all checkers
    gammon_loss: f32,  // P(lose gammon) - only non-zero if we have all checkers
}

fn solve_equity_full(
    us: usize,
    them: usize,
    n_positions: usize,
    max_checkers: usize,
    move_table: &[Vec<Vec<usize>>],
    position_sums: &[usize],
    table: &mut Vec<EquityResult>,
    computed: &mut Vec<bool>,
) -> EquityResult {
    let idx = pair_index(us, them, n_positions);
    if computed[idx] {
        return table[idx];
    }

    let us_sum = position_sums[us];
    let them_sum = position_sums[them];

    // Base cases
    if us_sum == 0 {
        // We won - check if gammon (opponent has all checkers)
        let is_gammon = them_sum == max_checkers;
        let res = EquityResult {
            cubeless: EQUITY_P1,
            owner: EQUITY_P1,
            center: EQUITY_P1,
            opponent: EQUITY_P1,
            gammon_win: if is_gammon { 1.0 } else { 0.0 },
            gammon_loss: 0.0,
        };
        table[idx] = res;
        computed[idx] = true;
        return res;
    }
    if them_sum == 0 {
        // We lost - check if gammon (we have all checkers)
        let is_gammon = us_sum == max_checkers;
        let res = EquityResult {
            cubeless: EQUITY_M1,
            owner: EQUITY_M1,
            center: EQUITY_M1,
            opponent: EQUITY_M1,
            gammon_win: 0.0,
            gammon_loss: if is_gammon { 1.0 } else { 0.0 },
        };
        table[idx] = res;
        computed[idx] = true;
        return res;
    }

    let mut totals = [0i32; 4];
    let mut gammon_win_total = 0.0f64;
    let mut gammon_loss_total = 0.0f64;

    for (dice_idx, (_d0, _d1)) in DICE_OUTCOMES.iter().enumerate() {
        let weight: i32 = if _d0 == _d1 { 1 } else { 2 };
        let weight_f = weight as f64;

        let mut best = [i16::MIN; 4];
        let mut best_gam_win = 0.0f32;
        let mut best_gam_loss = 1.0f32; // Start high, we want min gammon loss for best move

        let next_us = them;
        let next_positions = &move_table[us][dice_idx];

        for &next_them in next_positions {
            let child = solve_equity_full(
                next_us,
                next_them,
                n_positions,
                max_checkers,
                move_table,
                position_sums,
                table,
                computed,
            );

            let cand_cubeless = negate_equity(child.cubeless);
            let cand_owner = negate_equity(child.opponent);
            let k_center = cube_equity(child.center, child.opponent, EQUITY_P1);
            let cand_center = negate_equity(k_center);
            let k_opp = cube_equity(child.owner, child.opponent, EQUITY_P1);
            let cand_opp = negate_equity(k_opp);

            // Gammon probs swap perspective
            let cand_gam_win = child.gammon_loss;
            let cand_gam_loss = child.gammon_win;

            // Best move maximizes cubeless equity
            if cand_cubeless > best[0] {
                best[0] = cand_cubeless;
                best[1] = cand_owner;
                best[2] = cand_center;
                best[3] = cand_opp;
                best_gam_win = cand_gam_win;
                best_gam_loss = cand_gam_loss;
            }
        }

        for k in 0..4 {
            totals[k] += weight * best[k] as i32;
        }
        gammon_win_total += weight_f * best_gam_win as f64;
        gammon_loss_total += weight_f * best_gam_loss as f64;
    }

    let res = EquityResult {
        cubeless: (totals[0] / 36) as i16,
        owner: (totals[1] / 36) as i16,
        center: (totals[2] / 36) as i16,
        opponent: (totals[3] / 36) as i16,
        gammon_win: (gammon_win_total / 36.0) as f32,
        gammon_loss: (gammon_loss_total / 36.0) as f32,
    };
    table[idx] = res;
    computed[idx] = true;
    res
}

#[derive(Serialize)]
struct HeaderV2 {
    version: u32,
    max_checkers: usize,
    max_points: usize,
    layout: &'static str,
    dtype: &'static str,
    n_positions: usize,
    tier1_entries: usize,
    tier1_values_per_entry: usize,
    tier2_entries: usize,
    tier2_values_per_entry: usize,
    tier1_bytes: usize,
    tier2_bytes: usize,
    quantization: &'static str,
}

fn encode_uint24(v: f32, min_val: f32, max_val: f32) -> [u8; 3] {
    let normalized = (v - min_val) / (max_val - min_val);
    let clamped = normalized.max(0.0).min(1.0);
    let q = ((clamped * 16_777_215.0).round() as u32).min(16_777_215);
    [
        (q & 0xff) as u8,
        ((q >> 8) & 0xff) as u8,
        ((q >> 16) & 0xff) as u8,
    ]
}

fn encode_prob(v: f32) -> [u8; 3] {
    encode_uint24(v, 0.0, 1.0)
}

fn encode_equity(v: f32) -> [u8; 3] {
    encode_uint24(v, -1.0, 1.0)
}

// uint16 encoding for streaming output
fn encode_prob_u16(v: f32) -> u16 {
    let clamped = v.max(0.0).min(1.0);
    (clamped * 65535.0).round() as u16
}

fn encode_equity_u16(v: f32) -> u16 {
    let normalized = (v + 1.0) / 2.0;
    let clamped = normalized.max(0.0).min(1.0);
    (clamped * 65535.0).round() as u16
}

#[pyfunction]
fn generate_tiered_bearoff(
    py: Python<'_>,
    max_checkers: usize,
) -> PyResult<(PyObject, Py<PyArray2<u8>>, Py<PyArray2<u8>>)> {
    let n_positions = combination(max_checkers + NUM_POINTS, NUM_POINTS);

    // Build position table
    let mut positions: Vec<Position> = Vec::with_capacity(n_positions);
    for id in 0..n_positions {
        positions.push(position_from_bearoff(id, max_checkers));
    }
    let pos_to_idx = build_pos_index(&positions);
    let position_sums: Vec<usize> = positions.iter().map(position_sum).collect();

    // Build move table
    let mut move_table: Vec<Vec<Vec<usize>>> =
        vec![vec![Vec::new(); DICE_OUTCOMES.len()]; n_positions];
    for us_idx in 0..n_positions {
        let pos = positions[us_idx];
        for (dice_idx, (d0, d1)) in DICE_OUTCOMES.iter().enumerate() {
            let mut next_idxs = Vec::new();
            for child in generate_moves(&pos, *d0, *d1) {
                let idx = *pos_to_idx.get(&child).expect("missing position");
                next_idxs.push(idx);
            }
            next_idxs.sort();
            next_idxs.dedup();
            move_table[us_idx][dice_idx] = next_idxs;
        }
    }

    // Solve all positions
    let total_states = n_positions * n_positions;
    let mut eq_table: Vec<EquityResult> = vec![EquityResult::default(); total_states];
    let mut computed: Vec<bool> = vec![false; total_states];

    // Pre-compute all by iterating through full matrix
    for i in 0..n_positions {
        for j in 0..n_positions {
            solve_equity_full(
                i, j, n_positions, max_checkers,
                &move_table, &position_sums,
                &mut eq_table, &mut computed,
            );
        }
    }

    // Count tiers
    // Tier 2: gammon possible = at least one player has sum == max_checkers
    // Position 0 has sum 0 (all borne off), positions 1+ have sum >= 1
    // Gammon possible when i==0 or j==0 (one player has all checkers on board... wait no)
    // Actually: gammon = winner bears off all while loser hasn't borne off any
    // So gammon possible when either player still has all max_checkers on board
    // That means position_sums[i] == max_checkers OR position_sums[j] == max_checkers

    let mut tier1_count = 0usize;
    let mut tier2_count = 0usize;

    for i in 0..n_positions {
        for j in i..n_positions {
            let gammon_possible = position_sums[i] == max_checkers || position_sums[j] == max_checkers;
            if gammon_possible {
                tier2_count += 1;
            } else {
                tier1_count += 1;
            }
        }
    }

    // Tier 1: 10 values per entry (5 per perspective) = 30 bytes
    // Tier 2: 14 values per entry (7 per perspective) = 42 bytes
    const TIER1_VALUES: usize = 10;
    const TIER2_VALUES: usize = 14;

    let mut tier1_arr = ndarray::Array2::<u8>::zeros((tier1_count, TIER1_VALUES * 3));
    let mut tier2_arr = ndarray::Array2::<u8>::zeros((tier2_count, TIER2_VALUES * 3));

    let mut tier1_idx = 0usize;
    let mut tier2_idx = 0usize;

    for i in 0..n_positions {
        for j in i..n_positions {
            let eq_ij = eq_table[pair_index(i, j, n_positions)];
            let eq_ji = eq_table[pair_index(j, i, n_positions)];

            let gammon_possible = position_sums[i] == max_checkers || position_sums[j] == max_checkers;

            // Convert to float
            let win_ij = 0.5 * (1.0 + eq_ij.cubeless as f32 / EQUITY_P1 as f32);
            let win_ji = 0.5 * (1.0 + eq_ji.cubeless as f32 / EQUITY_P1 as f32);
            let eq_cl_ij = eq_ij.cubeless as f32 / EQUITY_P1 as f32;
            let eq_cl_ji = eq_ji.cubeless as f32 / EQUITY_P1 as f32;
            let eq_own_ij = eq_ij.owner as f32 / EQUITY_P1 as f32;
            let eq_own_ji = eq_ji.owner as f32 / EQUITY_P1 as f32;
            let eq_ctr_ij = eq_ij.center as f32 / EQUITY_P1 as f32;
            let eq_ctr_ji = eq_ji.center as f32 / EQUITY_P1 as f32;
            let eq_opp_ij = eq_ij.opponent as f32 / EQUITY_P1 as f32;
            let eq_opp_ji = eq_ji.opponent as f32 / EQUITY_P1 as f32;

            if gammon_possible {
                // Tier 2: [win, gam_win, gam_loss, eq_cl, eq_own, eq_ctr, eq_opp] x 2
                let values: [[u8; 3]; TIER2_VALUES] = [
                    encode_prob(win_ij),
                    encode_prob(eq_ij.gammon_win),
                    encode_prob(eq_ij.gammon_loss),
                    encode_equity(eq_cl_ij),
                    encode_equity(eq_own_ij),
                    encode_equity(eq_ctr_ij),
                    encode_equity(eq_opp_ij),
                    encode_prob(win_ji),
                    encode_prob(eq_ji.gammon_win),
                    encode_prob(eq_ji.gammon_loss),
                    encode_equity(eq_cl_ji),
                    encode_equity(eq_own_ji),
                    encode_equity(eq_ctr_ji),
                    encode_equity(eq_opp_ji),
                ];
                for (v_idx, bytes) in values.iter().enumerate() {
                    for b in 0..3 {
                        tier2_arr[(tier2_idx, v_idx * 3 + b)] = bytes[b];
                    }
                }
                tier2_idx += 1;
            } else {
                // Tier 1: [win, eq_cl, eq_own, eq_ctr, eq_opp] x 2
                let values: [[u8; 3]; TIER1_VALUES] = [
                    encode_prob(win_ij),
                    encode_equity(eq_cl_ij),
                    encode_equity(eq_own_ij),
                    encode_equity(eq_ctr_ij),
                    encode_equity(eq_opp_ij),
                    encode_prob(win_ji),
                    encode_equity(eq_cl_ji),
                    encode_equity(eq_own_ji),
                    encode_equity(eq_ctr_ji),
                    encode_equity(eq_opp_ji),
                ];
                for (v_idx, bytes) in values.iter().enumerate() {
                    for b in 0..3 {
                        tier1_arr[(tier1_idx, v_idx * 3 + b)] = bytes[b];
                    }
                }
                tier1_idx += 1;
            }
        }
    }

    let header = HeaderV2 {
        version: 2,
        max_checkers,
        max_points: NUM_POINTS,
        layout: "tiered_dual_perspective",
        dtype: "uint24",
        n_positions,
        tier1_entries: tier1_count,
        tier1_values_per_entry: TIER1_VALUES,
        tier2_entries: tier2_count,
        tier2_values_per_entry: TIER2_VALUES,
        tier1_bytes: tier1_count * TIER1_VALUES * 3,
        tier2_bytes: tier2_count * TIER2_VALUES * 3,
        quantization: "prob:[0,1]->uint24, equity:[-1,1]->uint24",
    };

    let header_json = serde_json::to_string(&header)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let tier1_py = tier1_arr.into_pyarray(py).to_owned();
    let tier2_py = tier2_arr.into_pyarray(py).to_owned();

    Ok((header_json.into_py(py), tier1_py, tier2_py))
}

// Keep old function for backwards compatibility
#[pyfunction]
fn generate_packed_bearoff(
    py: Python<'_>,
    max_checkers: usize,
    tolerance: f32,
    max_iter: usize,
) -> PyResult<(PyObject, Py<numpy::PyArray3<u8>>)> {
    let _ = tolerance;
    let _ = max_iter;

    let n_positions = combination(max_checkers + NUM_POINTS, NUM_POINTS);
    let mut positions: Vec<Position> = Vec::with_capacity(n_positions);
    for id in 0..n_positions {
        positions.push(position_from_bearoff(id, max_checkers));
    }
    let pos_to_idx = build_pos_index(&positions);
    let position_sums: Vec<usize> = positions.iter().map(position_sum).collect();

    let mut move_table: Vec<Vec<Vec<usize>>> =
        vec![vec![Vec::new(); DICE_OUTCOMES.len()]; n_positions];
    for us_idx in 0..n_positions {
        let pos = positions[us_idx];
        for (dice_idx, (d0, d1)) in DICE_OUTCOMES.iter().enumerate() {
            let mut next_idxs = Vec::new();
            for child in generate_moves(&pos, *d0, *d1) {
                let idx = *pos_to_idx.get(&child).expect("missing position");
                next_idxs.push(idx);
            }
            next_idxs.sort();
            next_idxs.dedup();
            move_table[us_idx][dice_idx] = next_idxs;
        }
    }

    let total_states = n_positions * n_positions;
    let mut eq_table: Vec<EquityResult> = vec![EquityResult::default(); total_states];
    let mut computed: Vec<bool> = vec![false; total_states];

    let entries = packed_index(n_positions - 1, n_positions - 1, n_positions) + 1;
    let mut arr = ndarray::Array3::<u8>::zeros((entries, 7, 3));

    for i in 0..n_positions {
        for j in i..n_positions {
            let eqs = solve_equity_full(
                i, j, n_positions, max_checkers,
                &move_table, &position_sums,
                &mut eq_table, &mut computed,
            );
            let cubeless_f = eqs.cubeless as f32 / EQUITY_P1 as f32;
            let win = 0.5 * (1.0 + cubeless_f);
            let idx = packed_index(i, j, n_positions);
            let loss = 1.0 - win;
            let eq_center = eqs.center as f32 / EQUITY_P1 as f32;
            let eq_owner = eqs.owner as f32 / EQUITY_P1 as f32;
            let eq_opp = eqs.opponent as f32 / EQUITY_P1 as f32;

            let mut store = |slot: usize, bytes: [u8; 3]| {
                arr[(idx, slot, 0)] = bytes[0];
                arr[(idx, slot, 1)] = bytes[1];
                arr[(idx, slot, 2)] = bytes[2];
            };

            store(0, encode_prob(win));
            store(1, encode_prob(eqs.gammon_win));
            store(2, encode_prob(loss));
            store(3, encode_prob(eqs.gammon_loss));
            store(4, encode_equity(eq_center));
            store(5, encode_equity(eq_owner));
            store(6, encode_equity(eq_opp));
        }
    }

    #[derive(Serialize)]
    struct Header {
        version: u32,
        max_checkers: usize,
        max_points: usize,
        packed: bool,
        slots: usize,
        dtype: &'static str,
        layout: &'static str,
        cubeful: bool,
        entries: usize,
        n_positions: usize,
        quantization: &'static str,
    }

    let header = Header {
        version: 1,
        max_checkers,
        max_points: NUM_POINTS,
        packed: true,
        slots: 7,
        dtype: "uint24_fixed",
        layout: "packed_upper",
        cubeful: true,
        entries,
        n_positions,
        quantization: "prob:[0,1]->uint24, equity:[-1,1]->uint24",
    };
    let header_obj = serde_json::to_string(&header)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let py_arr = arr.into_pyarray(py).to_owned();
    Ok((header_obj.into_py(py), py_arr))
}

/// Streaming uint16 generator - writes directly to files to minimize memory.
/// Full perspective format: stores BOTH (i,j) and (j,i) for each entry.
///
/// Version 4 format (12 values = 24 bytes per entry):
/// - Removes redundant win probability (derivable from eq_cl: win = (eq_cl + 1) / 2)
/// - Uses CONDITIONAL gammon probabilities (like GNUBG) for better precision
///
/// Layout per entry:
///   [gam_win_cond_ij, gam_loss_cond_ij, eq_cl_ij, eq_own_ij, eq_ctr_ij, eq_opp_ij,
///    gam_win_cond_ji, gam_loss_cond_ji, eq_cl_ji, eq_own_ji, eq_ctr_ji, eq_opp_ji]
///
/// To recover full probabilities at lookup time:
///   win = (eq_cl + 1) / 2
///   gam_win_unconditional = win * gam_win_cond
///   gam_loss_unconditional = (1 - win) * gam_loss_cond
#[pyfunction]
fn generate_streaming_bearoff_u16(
    _py: Python<'_>,
    max_checkers: usize,
    output_path: &str,
    header_path: &str,
) -> PyResult<()> {
    let n_positions = combination(max_checkers + NUM_POINTS, NUM_POINTS);
    eprintln!("Generating {}-checker bearoff: {} positions", max_checkers, n_positions);

    // Build position table
    let mut positions: Vec<Position> = Vec::with_capacity(n_positions);
    for id in 0..n_positions {
        positions.push(position_from_bearoff(id, max_checkers));
    }
    let pos_to_idx = build_pos_index(&positions);
    let position_sums: Vec<usize> = positions.iter().map(position_sum).collect();
    eprintln!("Built position table");

    // Build move table
    let mut move_table: Vec<Vec<Vec<usize>>> =
        vec![vec![Vec::new(); DICE_OUTCOMES.len()]; n_positions];
    for us_idx in 0..n_positions {
        let pos = positions[us_idx];
        for (dice_idx, (d0, d1)) in DICE_OUTCOMES.iter().enumerate() {
            let mut next_idxs = Vec::new();
            for child in generate_moves(&pos, *d0, *d1) {
                let idx = *pos_to_idx.get(&child).expect("missing position");
                next_idxs.push(idx);
            }
            next_idxs.sort();
            next_idxs.dedup();
            move_table[us_idx][dice_idx] = next_idxs;
        }
    }
    eprintln!("Built move table");

    // Allocate eq_table and computed - this is the main memory usage
    let total_states = n_positions * n_positions;
    let mut eq_table: Vec<EquityResult> = vec![EquityResult::default(); total_states];
    let mut computed: Vec<bool> = vec![false; total_states];
    eprintln!("Allocated eq_table: {} entries ({:.2} GB)",
              total_states,
              (total_states * std::mem::size_of::<EquityResult>()) as f64 / 1e9);

    // Pre-compute all positions (this builds the full eq_table in memory)
    eprintln!("Computing all positions...");
    for i in 0..n_positions {
        if i % 5000 == 0 {
            eprintln!("  Progress: {}/{} ({:.1}%)", i, n_positions, 100.0 * i as f64 / n_positions as f64);
        }
        for j in 0..n_positions {
            solve_equity_full(
                i, j, n_positions, max_checkers,
                &move_table, &position_sums,
                &mut eq_table, &mut computed,
            );
        }
    }
    eprintln!("Computation complete");

    // Now stream output to file
    let file = File::create(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, file); // 64MB buffer

    // Calculate total entries
    let total_entries = n_positions * (n_positions + 1) / 2;
    eprintln!("Writing {} entries to {}", total_entries, output_path);

    const VALUES_PER_ENTRY: usize = 12;
    let mut entries_written = 0usize;

    for i in 0..n_positions {
        if i % 5000 == 0 {
            eprintln!("  Writing: {}/{} ({:.1}%)", i, n_positions, 100.0 * i as f64 / n_positions as f64);
        }
        for j in i..n_positions {
            let eq_ij = eq_table[pair_index(i, j, n_positions)];
            let eq_ji = eq_table[pair_index(j, i, n_positions)];

            // Convert to float
            let win_ij = 0.5 * (1.0 + eq_ij.cubeless as f32 / EQUITY_P1 as f32);
            let win_ji = 0.5 * (1.0 + eq_ji.cubeless as f32 / EQUITY_P1 as f32);

            // Convert unconditional gammon probs to conditional
            // gam_win_cond = P(gammon | win) = P(gammon ∩ win) / P(win)
            // gam_loss_cond = P(gammon | loss) = P(gammon ∩ loss) / P(loss)
            let gam_win_cond_ij = if win_ij > 0.0 { eq_ij.gammon_win / win_ij } else { 0.0 };
            let gam_loss_cond_ij = if win_ij < 1.0 { eq_ij.gammon_loss / (1.0 - win_ij) } else { 0.0 };
            let gam_win_cond_ji = if win_ji > 0.0 { eq_ji.gammon_win / win_ji } else { 0.0 };
            let gam_loss_cond_ji = if win_ji < 1.0 { eq_ji.gammon_loss / (1.0 - win_ji) } else { 0.0 };

            let eq_cl_ij = eq_ij.cubeless as f32 / EQUITY_P1 as f32;
            let eq_cl_ji = eq_ji.cubeless as f32 / EQUITY_P1 as f32;
            let eq_own_ij = eq_ij.owner as f32 / EQUITY_P1 as f32;
            let eq_own_ji = eq_ji.owner as f32 / EQUITY_P1 as f32;
            let eq_ctr_ij = eq_ij.center as f32 / EQUITY_P1 as f32;
            let eq_ctr_ji = eq_ji.center as f32 / EQUITY_P1 as f32;
            let eq_opp_ij = eq_ij.opponent as f32 / EQUITY_P1 as f32;
            let eq_opp_ji = eq_ji.opponent as f32 / EQUITY_P1 as f32;

            // 12 uint16 values per entry (no win, conditional gammons)
            let values: [u16; VALUES_PER_ENTRY] = [
                encode_prob_u16(gam_win_cond_ij),
                encode_prob_u16(gam_loss_cond_ij),
                encode_equity_u16(eq_cl_ij),
                encode_equity_u16(eq_own_ij),
                encode_equity_u16(eq_ctr_ij),
                encode_equity_u16(eq_opp_ij),
                encode_prob_u16(gam_win_cond_ji),
                encode_prob_u16(gam_loss_cond_ji),
                encode_equity_u16(eq_cl_ji),
                encode_equity_u16(eq_own_ji),
                encode_equity_u16(eq_ctr_ji),
                encode_equity_u16(eq_opp_ji),
            ];

            // Write as little-endian bytes
            for v in values.iter() {
                writer.write_all(&v.to_le_bytes())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            }
            entries_written += 1;
        }
    }

    writer.flush()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    eprintln!("Wrote {} entries", entries_written);

    // Write header
    #[derive(Serialize)]
    struct HeaderV4 {
        version: u32,
        max_checkers: usize,
        max_points: usize,
        layout: &'static str,
        dtype: &'static str,
        n_positions: usize,
        total_entries: usize,
        values_per_entry: usize,
        bytes_per_entry: usize,
        total_bytes: usize,
        quantization: &'static str,
        value_order: &'static str,
        notes: &'static str,
    }

    let header = HeaderV4 {
        version: 4,
        max_checkers,
        max_points: NUM_POINTS,
        layout: "full_perspective_upper_triangle",
        dtype: "uint16",
        n_positions,
        total_entries: entries_written,
        values_per_entry: VALUES_PER_ENTRY,
        bytes_per_entry: VALUES_PER_ENTRY * 2,
        total_bytes: entries_written * VALUES_PER_ENTRY * 2,
        quantization: "prob:[0,1]->uint16, equity:[-1,1]->uint16",
        value_order: "gam_win_cond_ij,gam_loss_cond_ij,eq_cl_ij,eq_own_ij,eq_ctr_ij,eq_opp_ij,gam_win_cond_ji,gam_loss_cond_ji,eq_cl_ji,eq_own_ji,eq_ctr_ji,eq_opp_ji",
        notes: "win derived from eq_cl: win=(eq_cl+1)/2. Gammons are conditional: P(gam|win) and P(gam|loss)",
    };

    let header_json = serde_json::to_string_pretty(&header)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    std::fs::write(header_path, &header_json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    eprintln!("Done! Output: {}, Header: {}", output_path, header_path);
    Ok(())
}

#[pymodule]
fn bearoff(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(generate_packed_bearoff, m)?)?;
    m.add_function(wrap_pyfunction!(generate_tiered_bearoff, m)?)?;
    m.add_function(wrap_pyfunction!(generate_streaming_bearoff_u16, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_move_overshoot_requires_highest() {
        let pos = [0, 0, 0, 0, 1, 0];
        assert!(legal_move(&pos, 4, 6));
        let pos_blocked = [1, 0, 0, 0, 1, 1];
        assert!(!legal_move(&pos_blocked, 4, 6));
    }

    #[test]
    fn generate_moves_handles_doubles_and_partials() {
        let pos = [0, 0, 0, 0, 0, 2];
        let res = generate_moves(&pos, 6, 6);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_tier_classification() {
        // Position 0 = all borne off (sum = 0)
        // Any other position has sum >= 1
        // Gammon possible when one side has sum == max_checkers
        let max_checkers = 6;
        let n = combination(max_checkers + NUM_POINTS, NUM_POINTS);

        let mut positions: Vec<Position> = Vec::new();
        for id in 0..n {
            positions.push(position_from_bearoff(id, max_checkers));
        }
        let sums: Vec<usize> = positions.iter().map(position_sum).collect();

        // First position should have all checkers (sum = max_checkers for one-sided)
        // Actually position 0 has the pattern that gives sum = 0 in gnubg ordering
        // Let's verify
        assert_eq!(sums[0], 0); // All borne off

        // Find position with max sum
        let max_sum_pos = sums.iter().position(|&s| s == max_checkers);
        assert!(max_sum_pos.is_some());
    }
}
