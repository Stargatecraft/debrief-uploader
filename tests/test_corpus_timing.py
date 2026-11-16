"""Replay the real archive's battle cadence through the matching engine.

Synthetic unit tests prove the rules; this proves the rules survive the actual
distribution of the user's play -- including the 5% of battles that close
within about five minutes of each other, which is what makes a naive time
window wrong. Skipped when the archive isn't on this machine.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from debrief_uploader.config import DEFAULTS
from debrief_uploader.matcher import (
    AUTO, HOLD, AWAITING_SHOTS, UPLOADED, ReplayRec, ShotRec,
    cluster_shots, match_cluster,
)

ARCHIVE = os.path.expanduser("~/WowsLegendsReplayArchive")
NAMED = re.compile(r"^(\d{8})_(\d{6})_")


def load_closes():
    """(mtime, name) for every archived replay, oldest first."""
    out = []
    try:
        names = os.listdir(ARCHIVE)
    except OSError:
        return out
    for n in names:
        if n.endswith(".wowsreplay") and NAMED.match(n):
            out.append((os.path.getmtime(os.path.join(ARCHIVE, n)), n))
    out.sort()
    return out


def sessions(closes, split=2 * 3600):
    """Split the archive into play sessions on gaps larger than `split`."""
    cur, out = [], []
    for c in closes:
        if cur and c[0] - cur[-1][0] > split:
            out.append(cur)
            cur = []
        cur.append(c)
    if cur:
        out.append(cur)
    return out


class TestAgainstRealCadence(unittest.TestCase):
    """Pairing never asks (Greg, 2026-11-16): each set of screenshots goes to
    the battle that finished last before them. File times cannot order two
    battles that both finished inside the screenshot delay (the player left
    one early, played the next, and only then took the first one's
    scorecards). The battle time printed on the screenshots settles those,
    and the site sets aside a scorecard whose players are another battle's.
    So the claim proved here is exactly that: a wrong pairing happens only in
    a session where two battles closed within the delay of each other."""

    @classmethod
    def setUpClass(cls):
        cls.closes = load_closes()
        if len(cls.closes) < 50:
            raise unittest.SkipTest("replay archive not on this machine")
        cls.t = {n: t for t, n in cls.closes}

    def _run_session(self, batch, shot_delay, cfg, every=1):
        """Simulate: after each battle closes (every `every`-th one; the
        rest are skipped), the player takes two scorecards `shot_delay`
        seconds later. Returns (correct, [(true, got)...], held, missed)."""
        replays = [ReplayRec(key=n, t_close=t, state=AWAITING_SHOTS,
                             match_group="pvp") for t, n in batch]
        by_key = {r.key: r for r in replays}
        correct, wrong, held, missed = 0, [], 0, 0
        for i, (t, n) in enumerate(batch):
            if i % every:
                continue
            sh = [ShotRec("a", t + shot_delay), ShotRec("b", t + shot_delay + 7)]
            now = sh[-1].t + cfg["pair_settle"] + 1
            closed, _ = cluster_shots(sh, cfg["pair_gap"], cfg["pair_settle"], now)
            d = match_cluster(closed[0], replays, cfg)
            if d.kind == AUTO and d.replay.key == n:
                correct += 1
                by_key[n].state = UPLOADED
            elif d.kind == AUTO:
                wrong.append((n, d.replay.key))
                d.replay.state = UPLOADED
            elif d.kind == HOLD:
                held += 1
            else:
                missed += 1
        return correct, wrong, held, missed

    def _run(self, shot_delay, every=1):
        """(correct, wrong, unmatched) over the whole archive.

        Asserts the one way it may go wrong: only in a session where two
        battles closed within the screenshot delay (plus grace) of each
        other. Once one of those has taken the other's scorecards, later sets
        in the same session can follow on, which is why this is per session
        and not per pair."""
        cfg = dict(DEFAULTS)
        reach = shot_delay + cfg["grace"]
        correct, wrong, held, missed = 0, 0, 0, 0
        for batch in sessions(self.closes):
            c, w, h, m = self._run_session(batch, shot_delay, cfg, every)
            correct, held, missed = correct + c, held + h, missed + m
            wrong += len(w)
            tight = any(b[0] - a[0] <= reach for a, b in zip(batch, batch[1:]))
            if not tight:
                self.assertEqual(w, [], "no two battles in this session closed "
                                        "within %d s, so file times alone "
                                        "order them" % reach)
        self.assertEqual(held, 0, "pairing never asks")
        return correct, wrong, missed

    def test_every_battle_pairs_with_itself(self):
        correct, wrong, missed = self._run(45)
        print("\n  by file time alone, %d battles: %d correct, %d WRONG, %d unmatched"
              % (correct + wrong + missed, correct, wrong, missed))
        # Only a pair that closed seconds apart can go wrong (or be left
        # unattached), and battles are minutes long: the archive's one such
        # pair (31 Oct, the week the PC clock was moved) is all it may cost.
        self.assertLessEqual(wrong + missed, 2)

    def test_slow_screenshotter_errs_only_after_close_battles(self):
        """A player who takes minutes to grab the scorecards: anything wrong
        follows two battles that closed within that delay. A set whose own
        battle was given another's goes to an older battle still waiting, or
        is left unattached when there is none within 25 minutes."""
        for delay in (10, 60, 120, 240):
            correct, wrong, missed = self._run(delay)
            print("\n  by file time alone, scorecards %3d s late: %d correct, %d WRONG, %d unmatched"
                  % (delay, correct, wrong, missed))

    def test_skipping_screenshots_pairs_the_battle_just_played(self):
        """The player screenshots only every other battle: each set goes to
        the battle just played, never to the skipped one before it."""
        correct, wrong, missed = self._run(45, every=2)
        print("\n  by file time alone, every-other-battle: %d correct, %d WRONG, %d unmatched"
              % (correct, wrong, missed))
        self.assertLessEqual(wrong + missed, 2)

    def test_tightest_real_gaps_are_handled(self):
        """Sanity: the corpus really does contain the fast back-to-back gaps
        that make this hard -- otherwise the tests above prove nothing."""
        gaps = sorted(b[0] - a[0] for a, b in zip(self.closes, self.closes[1:])
                      if 0 < b[0] - a[0] < 7200)
        self.assertGreater(len(gaps), 300)
        p05 = gaps[int(0.05 * len(gaps))]
        tight = sum(1 for g in gaps if g < 300)
        print("\n  %d in-session gaps, p05 = %.0fs, %d under 5 min" % (len(gaps), p05, tight))
        # The COUNT of genuinely tight gaps is the substantive claim: without
        # them the matching tests above would be proving nothing. The
        # percentile is a loose sanity bound only -- this archive grows every
        # time Greg plays, and an earlier version pinned p05 < 330 s and broke
        # a day later when it drifted to 331.
        self.assertGreater(tight, 10, "corpus must contain real sub-5-minute gaps")
        self.assertLess(p05, 600, "back-to-back battles should still be common")


if __name__ == "__main__":
    unittest.main(verbosity=2)
