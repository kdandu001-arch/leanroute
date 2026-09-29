import os

# Tests must never write to the developer's real ~/.leanroute/usage.db
os.environ.setdefault("LEANROUTE_DB", ":memory:")
