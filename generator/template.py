from torchtune.data import InstructTemplate
from typing import Mapping, Any, Optional, Dict


class StateTacticPairTemplate(InstructTemplate):
    template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n[GOAL]\n{state}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n[PROOFSTEP]\n"

    @classmethod
    def format(
        cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None
    ) -> str:
        column_map = column_map or {}
        key_state = column_map.get("state", "state")
        return cls.template.format(state=sample[key_state])
