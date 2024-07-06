from torchtune.data import InstructTemplate
from typing import Mapping, Any, Optional, Dict


class StateTacticPairTemplate(InstructTemplate):
    template = "### State:\n{state}\n\n### Tactic:"
    # template = "[GOAL]\n{state}\n[PROOFSTEP]\n"

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        column_map = column_map or {}
        key_state = column_map.get("state", "state")
        return cls.template.format(state=sample[key_state])
