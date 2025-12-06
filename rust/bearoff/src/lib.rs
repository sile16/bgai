//! Bearoff generator matching gnubg's makebearoff two-sided logic.
//! - Uses gnubg's bearoff ID ordering (PositionBearoff).
//! - Exact CubeEquity rule (double/take/pass) on cubeless equities.
//! - Computes only the upper triangle (i <= j) and mirrors the rest.
//! - Outputs packed upper triangle (i <= j) with 7 uint24-fixed values
//!   (3 bytes each, little endian), fields:
//!   [win, gammon_win, loss, gammon_loss, eq_center, eq_owner, eq_opponent].
//!   Probabilities use [0,1] scaling; equities use [-1,1] mapped to [0,1].

use bkgm::{Dice as BgDice, Position as BgPosition, O_BAR, X_BAR};
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use serde::Serialize;
use serde_json;
use std::collections::HashMap;

const NUM_POINTS: usize = 6;
const EQUITY_P1: i16 = 0x7FFF; // gnubg: +1.0
const EQUITY_M1: i16 = !EQUITY_P1; // gnubg: -1.0

const DICE_OUTCOMES: [(u8, u8); 21] = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 5),
    (4, 6),
    (5, 6),
];

type Position = [u8; NUM_POINTS]; // point 0 is borne off

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

#[allow(dead_code)]
fn position_bearoff(pos: &Position, max_checkers: usize) -> usize {
    // Port of PositionBearoff from gnubg/positionid.c
    let mut j: usize = NUM_POINTS - 1;
    for i in 0..NUM_POINTS {
        j += pos[i] as usize;
    }
    let mut fbits: usize = 1usize << j;
    for i in 0..NUM_POINTS - 1 {
        j -= pos[i] as usize + 1;
        fbits |= 1usize << j;
    }
    fn position_f(bits: usize, n: usize, r: usize) -> usize {
        if n == r {
            return 0;
        }
        if (bits >> (n - 1)) & 1 == 1 {
            combination(n - 1, r) + position_f(bits, n - 1, r - 1)
        } else {
            position_f(bits, n - 1, r)
        }
    }
    position_f(fbits, max_checkers + NUM_POINTS, NUM_POINTS)
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

fn to_bg_position(us: &Position, them: &Position, max_checkers: usize) -> Option<BgPosition> {
    let us_sum = position_sum(us);
    let them_sum = position_sum(them);
    if us_sum > max_checkers || them_sum > max_checkers {
        return None;
    }

    let mut pips = [0i8; 26];
    for i in 0..NUM_POINTS {
        let c_us = us[i] as i8;
        if c_us > 0 {
            pips[i + 1] = c_us;
        }
        let c_them = them[i] as i8;
        if c_them > 0 {
            // Opponent home board is mirrored.
            pips[24 - i] = -c_them;
        }
    }

    let x_off = (max_checkers - us_sum) as u8;
    let o_off = (max_checkers - them_sum) as u8;

    // No checkers on the bar in bearoff.
    pips[X_BAR] = 0;
    pips[O_BAR] = 0;

    Some(BgPosition { pips, x_off, o_off })
}

fn from_bg_position(bg: &BgPosition, max_checkers: usize) -> Option<(Position, Position)> {
    if bg.pips[X_BAR] != 0 || bg.pips[O_BAR] != 0 {
        return None;
    }
    let mut us = [0u8; NUM_POINTS];
    let mut them = [0u8; NUM_POINTS];

    // Current player (x) home board is pips 1..6
    for i in 0..NUM_POINTS {
        let pip = i + 1;
        let c = bg.pips[pip] as i8;
        if c < 0 {
            return None;
        }
        us[i] = c as u8;
    }

    // Opponent home board is mirrored at 24..19.
    for i in 0..NUM_POINTS {
        let pip = 24 - i;
        let c = bg.pips[pip] as i8;
        if c > 0 {
            return None;
        }
        them[i] = (-c) as u8;
    }

    // All other points must be empty in pure bearoff.
    for pip in 7..=18 {
        if bg.pips[pip] != 0 {
            return None;
        }
    }

    if position_sum(&us) + bg.x_off as usize != max_checkers {
        return None;
    }
    if position_sum(&them) + bg.o_off as usize != max_checkers {
        return None;
    }

    Some((us, them))
}

fn highest_point(pos: &Position) -> Option<usize> {
    (0..NUM_POINTS).rev().find(|&i| pos[i] > 0)
}

fn legal_move(pos: &Position, src: usize, roll: u8) -> bool {
    let dest = src as isize - roll as isize;
    if dest >= 0 {
        return true;
    }

    // Bearing off with overthrows: allowed only if all checkers are in the home
    // board and this checker is on the highest occupied point (or exact bear off).
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
            let next_start = if rolls[0] == rolls[1] {
                src
            } else {
                NUM_POINTS - 1
            };
            if generate_moves_sub(&next, rolls, depth + 1, next_start, allow_partial, out) {
                out.push(next);
            }
        }
    }

    !used || allow_partial
}

fn generate_moves(pos: &Position, d0: u8, d1: u8) -> Vec<Position> {
    let mut results = Vec::new();

    let rolls1 = [
        d0,
        d1,
        if d0 == d1 { d0 } else { 0 },
        if d0 == d1 { d0 } else { 0 },
    ];
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
        if dt >= dp / 2 {
            dp
        } else {
            2 * dt
        }
    } else {
        nd
    }
}

fn solve_equity(
    us: usize,
    them: usize,
    n_positions: usize,
    move_table: &[Vec<Vec<usize>>],
    position_sums: &[usize],
    table: &mut Vec<[i16; 4]>,
    computed: &mut Vec<bool>,
) -> [i16; 4] {
    let idx = pair_index(us, them, n_positions);
    if computed[idx] {
        return table[idx];
    }

    let us_sum = position_sums[us];
    let them_sum = position_sums[them];

    // Base cases
    if us_sum == 0 {
        let res = [EQUITY_P1; 4];
        table[idx] = res;
        computed[idx] = true;
        return res;
    }
    if them_sum == 0 {
        let res = [EQUITY_M1; 4];
        table[idx] = res;
        computed[idx] = true;
        return res;
    }

    let mut totals = [0i32; 4];

    for (dice_idx, (_d0, _d1)) in DICE_OUTCOMES.iter().enumerate() {
        let weight: i32 = if _d0 == _d1 { 1 } else { 2 };
        let mut best = [i16::MIN; 4];

        let next_us = them; // roles swap after we move.
        let next_positions = &move_table[us][dice_idx];

        for &next_them in next_positions {
            let child = solve_equity(
                next_us,
                next_them,
                n_positions,
                move_table,
                position_sums,
                table,
                computed,
            );

            let cand_cubeless = negate_equity(child[0]);
            if cand_cubeless > best[0] {
                best[0] = cand_cubeless;
            }

            let cand_owner = negate_equity(child[3]); // we own cube -> opp owns in child view
            if cand_owner > best[1] {
                best[1] = cand_owner;
            }

            let k_center = cube_equity(child[2], child[3], EQUITY_P1);
            let cand_center = negate_equity(k_center);
            if cand_center > best[2] {
                best[2] = cand_center;
            }

            let k_opp = cube_equity(child[1], child[3], EQUITY_P1);
            let cand_opp = negate_equity(k_opp);
            if cand_opp > best[3] {
                best[3] = cand_opp;
            }
        }

        for k in 0..4 {
            totals[k] += weight * best[k] as i32;
        }
    }

    let res = [
        (totals[0] / 36) as i16,
        (totals[1] / 36) as i16,
        (totals[2] / 36) as i16,
        (totals[3] / 36) as i16,
    ];
    table[idx] = res;
    computed[idx] = true;
    res
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

#[pyfunction]
fn generate_packed_bearoff(
    py: Python<'_>,
    max_checkers: usize,
    tolerance: f32,
    max_iter: usize,
) -> PyResult<(PyObject, Py<PyArray3<u8>>)> {
    // tolerance and max_iter kept for API compatibility (unused in exact solver)
    let _ = tolerance;
    let _ = max_iter;

    let n_positions = combination(max_checkers + NUM_POINTS, NUM_POINTS);
    let mut positions: Vec<Position> = Vec::with_capacity(n_positions);
    for id in 0..n_positions {
        positions.push(position_from_bearoff(id, max_checkers));
    }
    let pos_to_idx = build_pos_index(&positions);

    let position_sums: Vec<usize> = positions.iter().map(position_sum).collect();

    // For bearoff, legal moves depend only on the current player's board and dice.
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
    let mut eq_table: Vec<[i16; 4]> = vec![[0; 4]; total_states];
    let mut computed: Vec<bool> = vec![false; total_states];

    let entries = packed_index(n_positions - 1, n_positions - 1, n_positions) + 1;
    // Shape: (entries, 7 slots, 3 bytes per uint24)
    let mut arr = ndarray::Array3::<u8>::zeros((entries, 7, 3));

    let encode_unit = |v: f32| -> [u8; 3] {
        let clamped = if v < 0.0 {
            0.0
        } else if v > 1.0 {
            1.0
        } else {
            v
        };
        let q = ((clamped * 16_777_215.0).round() as u32).min(16_777_215);
        [
            (q & 0xff) as u8,
            ((q >> 8) & 0xff) as u8,
            ((q >> 16) & 0xff) as u8,
        ]
    };

    let encode_prob = |v: f32| encode_unit(v);
    let encode_equity = |v: f32| encode_unit(0.5 * (v.max(-1.0).min(1.0) + 1.0));

    for i in 0..n_positions {
        for j in i..n_positions {
            let eqs = solve_equity(
                i,
                j,
                n_positions,
                &move_table,
                &position_sums,
                &mut eq_table,
                &mut computed,
            );
            let cubeless_f = eqs[0] as f32 / EQUITY_P1 as f32;
            let win = 0.5 * (1.0 + cubeless_f);
            let idx = packed_index(i, j, n_positions);
            let loss = 1.0 - win;
            let eq_center = eqs[2] as f32 / EQUITY_P1 as f32; // centered
            let eq_owner = eqs[1] as f32 / EQUITY_P1 as f32; // owner
            let eq_opp = eqs[3] as f32 / EQUITY_P1 as f32; // opponent

            let mut store = |slot: usize, bytes: [u8; 3]| {
                arr[(idx, slot, 0)] = bytes[0];
                arr[(idx, slot, 1)] = bytes[1];
                arr[(idx, slot, 2)] = bytes[2];
            };

            store(0, encode_prob(win));
            store(1, encode_prob(0.0)); // gammon win placeholder
            store(2, encode_prob(loss));
            store(3, encode_prob(0.0)); // gammon loss placeholder
            store(4, encode_equity(eq_center));
            store(5, encode_equity(eq_owner));
            store(6, encode_equity(eq_opp));
        }
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

#[pymodule]
fn bearoff(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(generate_packed_bearoff, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_move_overshoot_requires_highest() {
        // Highest checker on point 4 (distance 5): can bear off with 6.
        let pos = [0, 0, 0, 0, 1, 0];
        assert!(legal_move(&pos, 4, 6));

        // Another checker behind blocks bearing off from lower point.
        let pos_blocked = [1, 0, 0, 0, 1, 1];
        assert!(!legal_move(&pos_blocked, 4, 6));
    }

    #[test]
    fn generate_moves_handles_doubles_and_partials() {
        // Two checkers on the 6-point, rolling double sixes should bear both off.
        let pos = [0, 0, 0, 0, 0, 2];
        let res = generate_moves(&pos, 6, 6);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn generate_moves_non_double_orders() {
        // Check that swapping dice is considered (both orders end up bearing off).
        let pos = [0, 0, 0, 0, 1, 1];
        let res = generate_moves(&pos, 5, 6);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 0, 0, 0, 0, 0]);
    }
}
