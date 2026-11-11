"""Policy server for GR00T N1.6 (GR-1), run with the python of the Isaac-GR00T n1.6.1 environment (`.venv` of the clone).

It builds the same objects in the same order as the public gr00t/eval/run_gr00t_server.py (Isaac-GR00T n1.6.1-release,
commit 5dc80c4a) and adds a seeded endpoint and a record of the language input. The public endpoints (get_action and the
others) stay available, so the public gr00t/eval/rollout_policy.py can also connect to this server.

Public path (file:line at that commit):
  - gr00t/eval/run_gr00t_server.py:66-72   Gr00tPolicy(embodiment_tag, model_path, device, strict)
  - gr00t/eval/run_gr00t_server.py:91-94   --use-sim-policy-wrapper -> Gr00tSimPolicyWrapper(policy)
  - gr00t/eval/run_gr00t_server.py:96-103  PolicyServer(policy, host, port).run()
Added endpoints:
  - infer(observation, seed, want_prompt): torch.manual_seed(seed), then the wrapper's get_action (the function behind the
    public get_action endpoint). The action noise of GR00T N1.6 is torch.randn (gr00t/model/gr00t_n1d6/gr00t_n1d6.py:310),
    so the seed fixes it. Returns the action chunk, a NaN flag, and the language actually used:
      text_in    = the string that Gr00tPolicy._to_vla_step_data puts into VLAStepData.text (gr00t_policy.py:127-142);
      lang_model = the string after formalize_language (lower case, punctuation removed, processing_gr00t_n1d6.py:369-373)
                   that the processor puts into the chat template (_apply_vlm_processing, :256-289);
      prompt     = the chat-template text (only when want_prompt is true; otherwise its sha1).
    No code on this path replaces an empty string with another sentence; the record lets the worker check that the empty
    condition reaches the model as "".
  - info(): embodiment tag, language keys, formalize_language flag, torch version.
Usage: python gr1/gr1_server.py --port 5761 --model-path <local snapshot of nvidia/GR00T-N1.6-3B>
"""
import argparse
import hashlib
import os

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    base = Gr00tPolicy(embodiment_tag=EmbodimentTag.GR1, model_path=a.model_path, device="cuda", strict=True)
    policy = Gr00tSimPolicyWrapper(base)
    lang_keys = base.modality_configs["language"].modality_keys
    assert lang_keys == ["task"], lang_keys  # processor_config.json gr1.language (the wrapper reads it from coarse_action)

    rec = {}
    orig_vla = base._to_vla_step_data
    proc = base.processor
    orig_vlm = proc._apply_vlm_processing

    def spy_vla(observation):
        out = orig_vla(observation)
        rec.setdefault("text_in", []).append(out.text)
        return out

    def spy_vlm(images, language):
        out = orig_vlm(images, language)
        rec.setdefault("lang_model", []).append(language)
        rec.setdefault("prompt", []).append(out["vlm_content"]["text"])
        return out

    base._to_vla_step_data = spy_vla        # wrapped on the instance; the upstream files are not changed
    proc._apply_vlm_processing = spy_vlm

    def infer(observation, seed, want_prompt=False):
        rec.clear()
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        action, _info = policy.get_action(observation)
        nan = any(bool(np.isnan(np.asarray(v)).any()) for v in action.values())
        prompts = rec.get("prompt", [])
        return {"action": {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}, "nan": nan,
                "text_in": rec.get("text_in", []), "lang_model": rec.get("lang_model", []),
                "prompt_sha1": [hashlib.sha1(p.encode()).hexdigest() for p in prompts],
                "prompt": prompts if want_prompt else None}

    def info():
        return {"embodiment_tag": base.embodiment_tag.value, "language_keys": lang_keys,
                "formalize_language": bool(getattr(proc, "formalize_language", None)), "torch": torch.__version__}

    srv = PolicyServer(policy=policy, host=a.host, port=a.port)
    srv.register_endpoint("infer", infer)
    srv.register_endpoint("info", info, requires_input=False)
    print(f"GR-1 server ready port={a.port} pid={os.getpid()}", flush=True)
    srv.run()


if __name__ == "__main__":
    main()
