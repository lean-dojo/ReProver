from typing import List, Optional
from dataclasses import dataclass


@dataclass(unsafe_hash=True)
class Hyp:
    name: str
    type: str


@dataclass(unsafe_hash=True)
class Goal:
    hyps: List[Hyp]
    name: Optional[str]
    type: str


@dataclass(unsafe_hash=True)
class State:
    goals: List[Goal]


def parse_hyps(s: str) -> List[Hyp]:
    names, type = s.split(" : ")
    return [Hyp(name, type) for name in names.split()]


def parse_goal(s: str) -> Goal:
    lines = s.split("\n")
    assert len(lines) >= 1
    assert lines[-1].startswith("⊢")
    goal_type = lines[-1]
    lines = lines[:-1]
    goal_name = None
    if len(lines) >= 1 and not ":" in lines[0]:
        goal_name = lines[0]
        lines = lines[1:]
    return Goal([h for hs in lines for h in parse_hyps(hs)], goal_name, goal_type)


def parse_state(s: str) -> State:
    goals = s.split("\n\n")
    return State([parse_goal(g) for g in goals])


def format_states_pair(state_before_str: str, state_after_str: str) -> List[str]:
    state_before = parse_state(state_before_str)
    state_after = parse_state(state_after_str)
    # Filter out unchanged goals
    goals_before = [g for g in state_before.goals if not g in state_after.goals]
    goals_after = [g for g in state_after.goals if not g in state_before.goals]
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
        new_hyps = [hyp for hyp in to_goal.hyps if not hyp in from_goal.hyps]
        result.extend(
            [
                "...\n{} : {}\n{}".format(hyp.name, hyp.type, to_goal.type)
                for hyp in new_hyps
            ]
        )
        if from_goal.type != to_goal.type:
            result.append("...\n{}".format(to_goal.type))

    hyps_before = "\n".join(["{} : {}".format(h.name, h.type) for h in from_goal.hyps])
    before = "before\n{}\n{}".format(hyps_before, from_goal.type)
    return ["{}\n\nafter\n{}".format(before, after) for after in result]


def test(before: str, after: str):
    print("\n=========\n".join(format_states_pair(before, after)))


# test("⊢ p -> q", "h : p\n⊢ q")
# test("q p : Prop\nhp : p\nhq : q\n⊢ p ∧ q",
#      "q p : Prop\nhp : p\nhq : q\n⊢ p\n\nq p : Prop\nhp : p\nhq : q\n⊢ q")
# test("a b c : Nat\n⊢ c + (a + b) = c + b + a\n\n⊢ True",
#      "a b c : Nat\n⊢ c + (b + a) = c + b + a\n\n⊢ True")
# test("a : 1\nb : 2 \nc : 3\n⊢ goalA",
#      "a : 666\nb : 2 \nc : 3\nm : 777\n⊢ goalA")
test(
    "case inl\np : Prop\nh✝ : p\n⊢ True\n\ncase inr\np : Prop\nh✝ : p\n⊢ True",
    "case inr\np : Prop\nh✝ : p\n⊢ True",
)
