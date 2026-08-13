"""Torch-free numpy re-implementation of GreyKT-v3's white-box branch,
black-box confidence, reliability gate, and probability-space fusion, used
only to verify the MATH by hand-computed cases, since torch/torch_geometric
could not be installed in this sandbox. This is a throwaway verification
script, not part of the shipped codebase -- the real module is
dh2a_kt/models/greykt.py.
"""
import numpy as np


def _build_multihop_prereq_lookup_np(prereq_edges, n_concepts, max_hops, hop_decay):
    direct_parents = [[] for _ in range(n_concepts)]
    for p, c in zip(*prereq_edges):
        direct_parents[c].append(p)

    lookup = [[] for _ in range(n_concepts)]
    for c in range(n_concepts):
        if not direct_parents[c]:
            continue
        visited = {}
        frontier = list(direct_parents[c])
        hop = 1
        while frontier and hop <= max_hops:
            next_frontier = []
            for node in frontier:
                if node in visited:
                    continue
                visited[node] = hop
                next_frontier.extend(direct_parents[node])
            frontier = next_frontier
            hop += 1
        lookup[c] = [(anc, hop_decay ** (h - 1)) for anc, h in visited.items()]
    return lookup


def whitebox_branch_np(
    concept_ids,
    responses,
    prereq_edges,
    n_concepts,
    prior_mean=0.5,
    prior_strength=4.0,
    max_hops=3,
    hop_decay=0.5,
    recency_decay=0.98,
    kappa_w=4.0,
):
    B, T = concept_ids.shape
    lookup = _build_multihop_prereq_lookup_np(prereq_edges, n_concepts, max_hops, hop_decay)
    alpha0 = prior_mean * prior_strength
    beta0 = (1.0 - prior_mean) * prior_strength

    running_correct = np.zeros((B, n_concepts))
    running_count = np.zeros((B, n_concepts))
    white_prob = np.full((B, T), float(prior_mean))
    white_confidence = np.zeros((B, T))

    for t in range(T):
        for b in range(B):
            c = int(concept_ids[b, t])
            ancestors = lookup[c]
            if not ancestors:
                continue
            idx = np.array([a for a, _ in ancestors], dtype=int)
            w_pc = np.array([w for _, w in ancestors])
            n_eff_p = running_count[b, idx]
            s_eff_p = running_correct[b, idx]
            m_p = (alpha0 + s_eff_p) / (alpha0 + beta0 + n_eff_p)
            c_p = n_eff_p / (n_eff_p + kappa_w)

            numerator = (w_pc * c_p * m_p).sum()
            denominator = (w_pc * c_p).sum()
            pooled_n_eff = (w_pc * n_eff_p).sum()

            white_prob[b, t] = numerator / denominator if denominator > 0 else prior_mean
            white_confidence[b, t] = pooled_n_eff / (pooled_n_eff + kappa_w)

        running_count *= recency_decay
        running_correct *= recency_decay
        for b in range(B):
            c = int(concept_ids[b, t])
            running_count[b, c] += 1.0
            running_correct[b, c] += float(responses[b, t])

    return white_prob, white_confidence


def black_confidence_table_np(concept_train_freq, n_concepts, kappa_b):
    if concept_train_freq is None:
        return np.zeros(n_concepts)
    return concept_train_freq / (concept_train_freq + kappa_b)


def reliability_gate_np(cb, cw, eps=1e-6):
    return cb / (cb + cw + eps)


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


# --- Case 1: no prerequisites at all -> always prior, always zero white confidence ---
prereq_edges_empty = ([], [])
concept_ids1 = np.random.randint(0, 8, size=(4, 5))
responses1 = np.random.randint(0, 2, size=(4, 5)).astype(float)
wp1, cw1 = whitebox_branch_np(concept_ids1, responses1, prereq_edges_empty, n_concepts=8, prior_mean=0.37)
check("no-prereq case: white_confidence all zero", np.allclose(cw1, 0.0))
check("no-prereq case: white_prob all == prior_mean", np.allclose(wp1, 0.37))

# --- Case 2: single prerequisite, single observation -> shrinkage, not raw 1.0 ---
prereq_edges2 = ([0], [1])  # 0 -> 1
concept_ids2 = np.array([[0, 1]])
responses2 = np.array([[1.0, 0.0]])
prior_mean2, prior_strength2, kappa_w2 = 0.5, 4.0, 4.0
wp2, cw2 = whitebox_branch_np(
    concept_ids2, responses2, prereq_edges2, n_concepts=4,
    prior_mean=prior_mean2, prior_strength=prior_strength2, kappa_w=kappa_w2,
)
check("t=0 white_confidence == 0 (concept 0 has no prereqs)", cw2[0, 0] == 0.0)
check("t=0 white_prob == prior_mean", wp2[0, 0] == 0.5)
# single prereq -> the confidence-weighted average over 1 prerequisite reduces
# exactly to that prerequisite's own Beta-Binomial posterior mastery.
expected_m_p = (prior_mean2 * prior_strength2 + 1.0) / (prior_strength2 + 1.0)  # (2+1)/(4+1) = 0.6
check("t=1 white_prob == single-prereq Beta-Binomial posterior (0.6), NOT raw 1.0", abs(wp2[0, 1] - expected_m_p) < 1e-9)
check("t=1 white_prob strictly shrunk below 1.0", wp2[0, 1] < 1.0)
expected_cw_t1 = 1.0 / (1.0 + kappa_w2)  # pooled N_eff=1 -> 1/(1+4)=0.2
check("t=1 white_confidence == pooled N_eff/(N_eff+kappa_w) == 0.2", abs(cw2[0, 1] - expected_cw_t1) < 1e-9)
check("t=1 white_confidence strictly below 1.0 (continuous)", cw2[0, 1] < 1.0)

# --- Case 3: two prerequisites, contradictory evidence, EQUAL confidence -> symmetric cancellation ---
# concept 3's prereqs are {0, 1}, both direct (hop 1, w_pc=1). Student got
# concept 0 right 3 times, concept 1 wrong 3 times, then queries concept 3.
prereq_edges3 = ([0, 1], [3, 3])
concept_ids3 = np.array([[0, 0, 0, 1, 1, 1, 3]])
responses3 = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
wp3, cw3 = whitebox_branch_np(
    concept_ids3, responses3, prereq_edges3, n_concepts=5,
    prior_mean=0.5, prior_strength=4.0, kappa_w=4.0, recency_decay=1.0,
)
check("contradictory-but-symmetric evidence: white_prob == 0.5 (cancels out)", abs(wp3[0, -1] - 0.5) < 1e-9)
check("contradictory evidence: white_confidence > 0 (both prereqs have evidence)", cw3[0, -1] > 0.0)

# --- Case 4: multi-hop reachability, bounded by max_hops ---
prereq_edges4 = ([0, 1], [1, 2])  # 0 -> 1 -> 2
concept_ids4 = np.array([[0, 2]])
responses4 = np.array([[1.0, 0.0]])
_, cw4_1hop = whitebox_branch_np(concept_ids4, responses4, prereq_edges4, n_concepts=4, max_hops=1)
check("max_hops=1: concept 2's confidence unaffected by 2-hop ancestor 0", cw4_1hop[0, 1] == 0.0)
_, cw4_2hop = whitebox_branch_np(concept_ids4, responses4, prereq_edges4, n_concepts=4, max_hops=2, hop_decay=0.5)
check("max_hops=2: concept 2's confidence now > 0 (ancestor 0 reachable)", cw4_2hop[0, 1] > 0.0)

# --- Case 5: black confidence table + reliability gate ---
concept_train_freq = np.array([100.0, 1.0, 0.0, 50.0])
kappa_b = 4.0
cb_table = black_confidence_table_np(concept_train_freq, n_concepts=4, kappa_b=kappa_b)
check("C_B(0) high-frequency concept -> high confidence", cb_table[0] == 100.0 / 104.0)
check("C_B(2) zero-frequency concept -> zero confidence", cb_table[2] == 0.0)

# gate: C_B high, C_W low -> gate near 1 (trust black-box)
gate_black_favored = reliability_gate_np(cb=np.array([0.9]), cw=np.array([0.05]))
check("gate favors black-box when C_B >> C_W", gate_black_favored[0] > 0.9)
# gate: C_B low, C_W high -> gate near 0 (trust white-box)
gate_white_favored = reliability_gate_np(cb=np.array([0.05]), cw=np.array([0.9]))
check("gate favors white-box when C_W >> C_B", gate_white_favored[0] < 0.1)
# gate: both zero -> eps-guarded, lands at 0 (not NaN), not a crash
gate_both_zero = reliability_gate_np(cb=np.array([0.0]), cw=np.array([0.0]), eps=1e-6)
check("gate both-zero case doesn't NaN/crash", np.isfinite(gate_both_zero[0]))
# gate: both equal and nonzero -> gate == 0.5 (symmetric, unlike a min()-based gate)
gate_symmetric = reliability_gate_np(cb=np.array([0.4]), cw=np.array([0.4]))
check("gate == 0.5 when C_B == C_W (relative-reliability, not a floor)", abs(gate_symmetric[0] - 0.5) < 1e-6)

# --- Case 6: probability-space fusion arithmetic ---
gate6 = 0.5
p_black6 = 0.9
p_white6 = 0.1
fused6 = gate6 * p_black6 + (1 - gate6) * p_white6
check("prob-space fusion at gate=0.5 == arithmetic mean of the two probs", abs(fused6 - 0.5) < 1e-9)
check("gate=0 fusion reduces to pure white-box probability", (0 * p_black6 + 1 * p_white6) == p_white6)
check("gate=1 fusion reduces to pure black-box probability", (1 * p_black6 + 0 * p_white6) == p_black6)

# --- Case 7: v4a temperature scaling is monotone (AUC-preserving) ---
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


logits7 = np.array([-2.0, -0.5, 0.3, 1.7, 3.1])
for T in (0.5, 1.0, 2.0, 5.0):
    scaled = sigmoid(logits7 / T)
    check(
        f"v4a T={T}: temperature scaling preserves ordering (AUC unchanged)",
        np.all(np.diff(scaled) > 0),
    )
check("v4a T=1.0 is exactly the identity", np.allclose(sigmoid(logits7 / 1.0), sigmoid(logits7)))
check("v4a T>1 shrinks probabilities toward 0.5 (fixes overconfidence)",
      np.all(np.abs(sigmoid(logits7 / 3.0) - 0.5) < np.abs(sigmoid(logits7) - 0.5)))
check("v4a T<1 sharpens probabilities away from 0.5",
      np.all(np.abs(sigmoid(logits7 / 0.5) - 0.5) > np.abs(sigmoid(logits7) - 0.5)))


# --- Case 8: v4b absolute-reliability backoff ---
def absolute_backoff_np(mixture, cb, cw, kappa_r, backoff_prob):
    r = cb + cw
    alpha = r / (r + kappa_r)
    return alpha * mixture + (1.0 - alpha) * backoff_prob, alpha


base_rate = 0.62
mixture8 = 0.90  # both branches happen to agree on a confident prediction

# The exact case v3 could not express: identical g_B=0.5, opposite meanings.
p_both_strong, alpha_strong = absolute_backoff_np(mixture8, 0.9, 0.9, 1.0, base_rate)
p_both_weak, alpha_weak = absolute_backoff_np(mixture8, 0.05, 0.05, 1.0, base_rate)

gate_strong = reliability_gate_np(np.array([0.9]), np.array([0.9]))[0]
gate_weak = reliability_gate_np(np.array([0.05]), np.array([0.05]))[0]
check("v3 gate CANNOT distinguish both-strong from both-weak (both give ~0.5)",
      abs(gate_strong - 0.5) < 1e-4 and abs(gate_weak - 0.5) < 1e-4)
# The eps guard is additive, so it biases the gate proportionally MORE when both
# confidences are small: at C_B=C_W=0.05 the gate is 0.499995, not exactly 0.5,
# a slight tilt toward the white-box branch. Negligible at eps=1e-6, but it grows
# if eps is ever raised, so it is pinned down here rather than left to be
# discovered as a puzzling asymmetry during a training run.
check("gate_eps biases the low-confidence case slightly toward white-box",
      gate_weak < gate_strong and (0.5 - gate_weak) > (0.5 - gate_strong))
check("that eps bias stays below 1e-4 at the default eps=1e-6", (0.5 - gate_weak) < 1e-4)
check("v4b alpha DOES distinguish them", alpha_strong > 0.6 and alpha_weak < 0.15)
check("v4b: both-strong keeps the branch mixture nearly intact",
      abs(p_both_strong - mixture8) < abs(p_both_weak - mixture8))
check("v4b: both-weak is pulled close to the base rate",
      abs(p_both_weak - base_rate) < abs(p_both_weak - mixture8))
check("v4b output stays a valid probability in both cases",
      0.0 <= p_both_strong <= 1.0 and 0.0 <= p_both_weak <= 1.0)

# Degenerate and boundary behavior.
p_zero, alpha_zero = absolute_backoff_np(mixture8, 0.0, 0.0, 1.0, base_rate)
check("v4b: zero evidence on both branches -> exactly the base rate",
      abs(p_zero - base_rate) < 1e-12 and alpha_zero == 0.0)
p_max, alpha_max = absolute_backoff_np(mixture8, 1.0, 1.0, 1e-9, base_rate)
check("v4b: kappa_r -> 0 recovers pure v3 behavior (no backoff)",
      abs(p_max - mixture8) < 1e-6 and alpha_max > 0.999)

# alpha is monotone increasing in total evidence -- the property that makes it
# interpretable as "how much to trust the branch mixture at all".
alphas = [absolute_backoff_np(mixture8, c, c, 1.0, base_rate)[1] for c in (0.0, 0.2, 0.5, 0.8, 1.0)]
check("v4b: alpha increases monotonically with total evidence", np.all(np.diff(alphas) > 0))

# v4b must not disturb the relative split: with backoff off (alpha=1), the
# result is exactly v3's mixture.
check("v4b disabled (alpha=1) reproduces v3 exactly", abs(1.0 * mixture8 + 0.0 * base_rate - mixture8) < 1e-12)

print("\nAll GreyKT v3 + v4a/v4b logic checks passed (numpy simulation).")
print("This does NOT substitute for running tests/test_greykt.py with real torch + torch_geometric.")
