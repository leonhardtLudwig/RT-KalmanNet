# RT-KalmanNet project docs

Working documentation for the RT-KalmanNet project (robust EKF + GRU-learned
tolerance `c_t`). Split into three files so updates are easy to find and
don't require reading one giant document:

- **[STATUS.md](STATUS.md)** — one-page snapshot of where the project
  stands against the assignment's 4 tasks (what's done, what's not, and the
  main open point). Start here for a quick "point of the situation."
- **[CHANGELOG.md](CHANGELOG.md)** — dated, reverse-chronological log of
  *what changed*: which files, what exact edits, and the one-line reason
  for each. Read this to catch up on "what happened since I last looked."
- **[FINDINGS.md](FINDINGS.md)** — *what we learned*, tagged GOOD / BAD /
  OBSERVATION, organized by experiment. This is the place for results,
  surprises, root-cause analyses, and things worth remembering that aren't
  simply "a bug we fixed." Read this to understand *why* things are the way
  they are.
- **[ROADMAP.md](ROADMAP.md)** — current status of Tasks 1-4 (per the
  assignment) and the prioritized next steps, kept up to date as decisions
  are made. Read this first if you're asking "what should I work on next."

Historical note: `PROGRESS.md` (in `CODE/RT_KFNET/`) was the original
single-file tracker for the Phase 1 `c_t`-collapse fix; its content has
been folded into the three files above and it's now just a pointer here.