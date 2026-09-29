# Contributing to Leanroute

Thanks for helping. Bug reports, benchmark results and pull requests are all welcome.

## Set up

```bash
git clone https://github.com/kdandu001-arch/leanroute && cd leanroute/server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -e ../sdk
```

## Run the tests

```bash
cd server && pytest -q        # gateway, guard, router, dashboard API
cd ../sdk && pytest -q        # Python package
```

Tests use stand-in engines, so they run in seconds with no model downloads. CI runs both suites on every push.

## Changing guard or routing behaviour

Claims about accuracy must come from the evaluation, not from a few hand-picked examples:

```bash
pip install -r eval/requirements.txt
python eval/build_dataset.py && python eval/score.py && python eval/score_protectai.py
python eval/train_router.py && python eval/report.py
```

Include the before/after numbers from `eval/results.md` in your pull request. Thresholds are tuned on the dev half and reported on the test half; please keep it that way.

## Pull requests

- Keep changes focused, with tests for new behaviour.
- Never commit API keys, `.env` files or prompt data.
- Update `CHANGELOG.md` under an "Unreleased" heading.
