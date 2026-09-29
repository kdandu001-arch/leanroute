"""Deploy the Leanroute server to a free Hugging Face Space (Docker, CPU).

The Space runs the guard, router and Laya for the website's playground. It has no LLM key, and the
chat gateway, stats and dashboard data are locked behind a random API key kept in the Space's secrets,
so the public can only use the rate-limited playground endpoints.

  hf auth login                     # once, with a Write token
  python deploy/hf_space.py [--space leanroute]
"""
from __future__ import annotations

import argparse
import secrets
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SITE_ORIGINS = "https://leanroute.online,http://leanroute.online,https://www.leanroute.online,http://www.leanroute.online"

DOCKERFILE = """\
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.cache/huggingface
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user app ./app
COPY --chown=user web ./web
USER user

# Download every model at build time so the Space starts fast
RUN python -c "import laya; laya.Router(preload=True)" && \\
    python -c "from app.guard import ProtectAIGuard; ProtectAIGuard().score('warm up')" && \\
    python -c "from app.learned_router import LearnedRouter; LearnedRouter().needs_strong('warm up')"

ENV WEB_DIR=/home/user/app/web LEANROUTE_DB=/tmp/leanroute.db GUARD_MODE=precise ROUTER=leanroute \\
    PUBLIC_PLAYGROUND=true PLAYGROUND_RPM=20 TRUST_PROXY=true HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
"""

README = """\
---
title: Leanroute
emoji: 🚦
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Open-source toll gate in front of any LLM
---

# Leanroute (live demo)

Blocks prompt injections, routes requests to a cheaper model when its answer will be good enough, and
proves the savings. This Space powers the playground on **[leanroute.online](https://leanroute.online)**.

- Code, docs and measured results: [github.com/kdandu001-arch/leanroute](https://github.com/kdandu001-arch/leanroute)
- Use it in your own app: `pip install leanroute`

This demo has no LLM attached: it answers the playground's decision questions only.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="leanroute")
    args = ap.parse_args()

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{args.space}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copytree(ROOT / "server" / "app", out / "app", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "web", out / "web")
        shutil.copy(ROOT / "server" / "requirements.txt", out / "requirements.txt")
        (out / "Dockerfile").write_text(DOCKERFILE)
        (out / "README.md").write_text(README)
        for name in ("LICENSE", "NOTICE"):
            shutil.copy(ROOT / name, out / name)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(out),
                          commit_message="Deploy Leanroute", delete_patterns=["*"])

    api.add_space_variable(repo_id, "CORS_ORIGINS", SITE_ORIGINS)
    # Locks the gateway, stats and dashboard data; nobody needs it for the playground. A fresh random key
    # is set on every deploy (secrets can't be read back), never printed or stored anywhere else.
    api.add_space_secret(repo_id, "LEANROUTE_API_KEYS", "owner:lr_" + secrets.token_urlsafe(24))
    print(f"deployed: https://huggingface.co/spaces/{repo_id}")
    print(f"app URL:  https://{repo_id.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
