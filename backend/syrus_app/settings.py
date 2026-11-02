"""Service settings: configs/syrus_service.yaml plus server environment variables. Secrets never leave the server."""
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.syrus_app.records import ROOT

CONFIG = ROOT / "configs" / "syrus_service.yaml"


@dataclass(frozen=True)
class Settings:
    cfg: dict
    key: str | None
    caps_confirmed: bool
    kill: bool
    secret: bytes
    ip_salt: bytes
    budget_file: Path

    @property
    def limits(self) -> dict:
        return self.cfg["limits"]

    @property
    def model(self) -> str:
        return self.cfg["model"]

    def offline_reason(self) -> str | None:
        """Why no model call may be made at all, before any budget check."""
        if self.kill:
            return "kill_switch"
        if not self.key:
            return "key_missing"
        if not self.caps_confirmed:
            return "caps_unconfirmed"
        return None


def load_settings(env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    return Settings(cfg=yaml.safe_load(CONFIG.read_text("utf-8")), key=env.get("SYRUS_LLM_API_KEY") or None,
                    caps_confirmed=env.get("SYRUS_CAPS_CONFIRMED") == "1", kill=env.get("SYRUS_KILL_SWITCH") == "1",
                    secret=(env.get("SYRUS_SESSION_SECRET") or secrets.token_hex(32)).encode(),
                    ip_salt=(env.get("SYRUS_IP_SALT") or secrets.token_hex(16)).encode(),
                    budget_file=Path(env.get("SYRUS_BUDGET_FILE") or ROOT / "outputs" / "syrus_budget.json"))
