"""Interface QA server: the real single-origin app with a scripted model double instead of Groq. No key, no network, no spend.

    python -m scripts.qa_web [--port 8000]

For checking how the interface renders answers and failure states only. Health reports `mock: true`, which the interface shows as a banner.
Replies are fixed, validator-checked answers built from the loaded case records; a message starting with "qa " triggers a failure state
(qa offline, qa rate, qa failed, qa slow). Never use it as evidence of what Syrus answers.
"""
import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import uvicorn

from backend.syrus_app.llm import Completion, LlmError
from backend.syrus_app.settings import load_settings
from backend.web import create_web

U = lambda text, rec=(), fnd=(), kind="model_evidence": {"text": text, "kind": kind, "record_ids": list(rec), "finding_ids": list(fnd)}
LABEL = U("The repository's transcription of the provider failure list labels 2020-05-29 23:30 to 2020-05-30 06:00 as \"Air leak, high stress\"; this is not an independent "
          "diagnosis, and AeroXAI's evidence does not establish it.", ["label.incident"], ["F-LABEL"], "dataset_label")
IDENT = U("Leakage and higher downstream demand cannot be told apart with the data recorded for this asset.",
          ["hyp.air_leakage", "hyp.downstream_demand", "unavailable:demand_measurement"], ["F-IDENT"], "hypothesis")
WHY = [U("The smoothed score was 58,989 against a threshold of 52,942 (+11.4%).", ["alert.smoothed", "alert.threshold", "alert.margin"], ["F-MARGIN"]),
       U("At 21:20 the bin's own raw score was 35.8, below the 73,727 carried from 21:15, so the smoothing state carried from earlier bins accounts for 99.99% of the current smoothed score.",
         ["alert.time", "alert.raw", "alert.prev_smoothed", "alert.memory"], ["F-EWMA-MEMORY"]),
       U("dv_pressure made up 77.6% of the smoothed score.", ["contrib.dv_pressure"])]
DOCS = [U("No OEM manual, SOP or maintenance ticket for this asset is loaded.", ["unavailable:documents"], [], "unavailable")]


def reply(units, purposes, refusal=None):
    return json.dumps({"lang": "en", "purposes": purposes, "units": units, "alternatives": [], "gaps": [], "next_test": None, "refusal": refusal, "operator_statement": False})


class Scripted:
    async def complete(self, messages, schema, max_tokens, timeout):
        last = messages[-1]["content"].lower()
        if "qa offline" in last:
            raise LlmError("provider_error", indeterminate=False)
        if "qa rate" in last:
            raise LlmError("rate_limited", 12.0, indeterminate=False)
        if "qa slow" in last:
            await asyncio.sleep(8)
        if "qa failed" in last:
            return Completion("not json", 1000, 10)
        if "leak" in last:
            return Completion(reply([LABEL, IDENT], ["evidence_and_sources"], "diagnosis_request"), 1600, 320)
        if "manual" in last or "missing" in last:
            return Completion(reply(DOCS, ["evidence_and_sources"]), 1500, 120)
        return Completion(reply(WHY, ["what_happened"]), 1900, 300)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interface QA server with a scripted model double.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="syrus_qa_"))
    env = {"SYRUS_LLM_API_KEY": "qa-fixture-not-a-key", "SYRUS_CAPS_CONFIRMED": "1", "SYRUS_BUDGET_FILE": str(tmp / "budget.json")}
    uvicorn.run(create_web(load_settings(env), Scripted()), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
