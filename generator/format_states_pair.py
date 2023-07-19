from lean_dojo import TacticState
from typing import List


def format_states_pair(state_before_str: str, state_after_str: str) -> List[str]:
    state_before = TacticState(state_before_str, 19)
    if state_after_str == "no goals":
        goals_before = state_before.goals
        goals_after = []
    else:
        state_after = TacticState(state_after_str, 11)
        # Filter out unchanged goals
        goals_before = [g for g in state_before.goals if g not in state_after.goals]
        goals_after = [g for g in state_after.goals if g not in state_before.goals]

    # We are going to assume that tactic always work on the first goal.
    # If it's not the case, e.g. it's a tactic combinator cracking up many goals we
    # can't establish a relationship which goal was created from where so we drop these
    # datapoints.
    if len(goals_before) != 1:
        return []
    from_goal = goals_before[0]
    result = []
    if len(goals_after) == 0:
        result.append("goals accomplished")
    for to_goal in goals_after:
        new_hyps = [
            hyp for hyp in to_goal.assumptions if not hyp in from_goal.assumptions
        ]
        result.extend(
            [
                "...\n{} : {}\n⊢ {}".format(
                    hyp.ident, hyp.lean_type, to_goal.conclusion
                )
                for hyp in new_hyps
            ]
        )
        if from_goal.conclusion != to_goal.conclusion:
            result.append("...\n⊢ {}".format(to_goal.conclusion))

    hyps_before = "\n".join(
        ["{} : {}".format(h.ident, h.lean_type) for h in from_goal.assumptions]
    )
    before = "before\n{}\n⊢ {}".format(hyps_before, from_goal.conclusion)
    return ["{}\n\nafter\n{}".format(before, after) for after in result]


def test(before: str, after: str):
    print("\n=========\n".join(format_states_pair(before, after)))


# Testing examples
# test_state = """R : Type u_2
# B : Type u_1
# F : Type u_3
# E : B → Type ?u.410614
# inst✝⁸ : NontriviallyNormedField R
# inst✝⁷ : (x : B) → AddCommMonoid (E x)
# inst✝⁶ : (x : B) → Module R (E x)
# inst✝⁵ : NormedAddCommGroup F
# inst✝⁴ : NormedSpace R F
# inst✝³ : TopologicalSpace B
# inst✝² : TopologicalSpace (TotalSpace E)
# inst✝¹ : (x : B) → TopologicalSpace (E x)
# inst✝ : FiberBundle F E
# ι : Type u_4
# Z : VectorBundleCore R B F ι
# b✝ : B
# a : F
# i j : ι
# b : B
# hb : b ∈ baseSet Z i
# v : F"""

# test(
#     "{}\n{}".format(
#         test_state,
#         "⊢ Trivialization.symm (localTriv Z i) b v = ↑(coordChange Z i (indexAt Z b) b) v",
#     ),
#     "{}\n{}".format(test_state, "⊢ True"),
# )
#
# test("⊢ p -> q", "h : p\n⊢ q")
# test(
#     "q p : Prop\nhp : p\nhq : q\n⊢ p ∧ q",
#     "q p : Prop\nhp : p\nhq : q\n⊢ p\n\nq p : Prop\nhp : p\nhq : q\n⊢ q",
# )
# test(
#     "a b c : Nat\n⊢ c + (a + b) = c + b + a\n\n⊢ True",
#     "a b c : Nat\n⊢ c + (b + a) = c + b + a\n\n⊢ True",
# )
# test("a : 1\nb : 2 \nc : 3\n⊢ goalA", "a : 666\nb : 2 \nc : 3\nm : 777\n⊢ goalA")
# test(
#     "case inl\np : Prop\nh✝ : p\n⊢ True\n\ncase inr\np : Prop\nh✝ : p\n⊢ True",
#     "case inr\np : Prop\nh✝ : p\n⊢ True",
# )
