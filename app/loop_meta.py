"""Loops: what carries over from one run to the next.

The story: the disaster hits at the end of every run, the player is sent back in time to
being a child, and keeps their memories. Each run is one loop. Between loops the player
keeps:

- remembered skills: chosen inside the going-back cutscene from the skills they owned at
  the end of the last loop; they start the next loop already in their kind's slot,
- keepsakes: six of them, one per townsperson below. Passing that person's quest unlocks
  it in the shop; it's bought with stars (every loop's stars go into a bank when the loop
  ends). Bought keepsakes stay for every later loop. They only change things around town
  (never the quest timer, stars or scoring),
- a journal of every loop: the threat that came, the Holland code and career it ended on.

Each loop also rolls a threat (typhoon, flood, earthquake, fire) - story only.

One document per player in `player_loop_meta`. Finishing a run closes the loop
(`complete_loop`, idempotent); "Go Back" starts the next one (`begin_loop`); New Game
wipes everything (`reset`). Every finished run in `session_runs` is stamped with its loop
number, threat, remembered skills and keepsakes, so later loops can be told apart from the
first one in the data.
"""

import random
from datetime import datetime
from typing import Any

THREATS = ["typhoon", "flood", "earthquake", "fire"]

# keepsake id -> (question code of the quest that unlocks it, price in stars). One per
# RIASEC kind, across all three towns. Effects live in the game; none touch the timer,
# stars or scoring. A full run earns up to 240 stars (48 quests x 5).
MEMENTOS = {
    "compass": ("I6", 40),    # Nico (Elementary): the hint arrow shows up faster
    "ribbon": ("A6", 50),     # Paolo (Elementary): +1 skill pick in each new town
    "sneakers": ("E2", 40),   # Jun (Elementary): walk faster in town
    "lantern": ("S8", 60),    # Lola Ising (High School): +1 skill pick each night
    "locket": ("C5", 70),     # Aling Nena (High School): remember 4 skills
    "whistle": ("R5", 50),    # Mang Erning (College): +1 request each morning
}
BASE_MEMORY_SLOTS = 3


class LoopError(Exception):
    """A request that doesn't fit the player's current loop (shown to the player)."""


def memory_slots(owned: list[str]) -> int:
    return BASE_MEMORY_SLOTS + (1 if "locket" in (owned or []) else 0)


def unlocks_from_records(records: list[dict[str, Any]]) -> list[str]:
    """Keepsakes unlocked by the passed quests in these run records."""
    passed = {str(r.get("question_code") or "").upper() for r in records or [] if r.get("completed")}
    return [m for m, (code, _) in MEMENTOS.items() if code in passed]


def stars_from_records(records: list[dict[str, Any]]) -> int:
    """Stars the player saw (0-5 per passed quest) across these run records."""
    total = 0
    for r in records or []:
        if not r.get("completed"):
            continue
        s5 = r.get("stars5")
        if s5 is None or int(s5) < 0:
            s5 = round(int(r.get("stars") or 0) * 5 / 3)
        total += max(0, int(s5))
    return total


def _roll_threat(previous: str | None, seen: list[str]) -> str:
    choices = [t for t in THREATS if t != previous]
    unseen = [t for t in choices if t not in (seen or [])]
    return random.choice(unseen or choices)


def _new_doc(player_id: str) -> dict[str, Any]:
    now = datetime.utcnow()
    return {
        "player_id": player_id,
        "current_loop": 1,
        "loops_completed": 0,
        "current_finished": False,
        "threat": _roll_threat(None, []),
        "threats_seen": [],
        "kept_skills": [],
        "unlocked_mementos": [],
        "owned_mementos": [],
        "stars": 0,
        "history": [],
        "last_owned_skills": [],
        "last_passed": [],
        "last_failed": [],
        "created_at": now,
        "updated_at": now,
    }


def get(repo, player_id: str) -> dict[str, Any]:
    """The player's loop record, created (loop 1, a random threat) if there is none yet."""
    doc = repo.player_loop_meta.find_one({"player_id": player_id})
    if doc:
        return doc
    doc = _new_doc(player_id)
    repo.player_loop_meta.update_one({"player_id": player_id}, {"$setOnInsert": doc}, upsert=True)
    return repo.player_loop_meta.find_one({"player_id": player_id}) or doc


def reset(repo, player_id: str) -> dict[str, Any]:
    """New Game: forget every loop and start again from loop 1."""
    repo.player_loop_meta.delete_one({"player_id": player_id})
    return get(repo, player_id)


def _unlock(repo, player_id: str, ids: list[str]) -> None:
    if ids:
        repo.player_loop_meta.update_one(
            {"player_id": player_id},
            {"$addToSet": {"unlocked_mementos": {"$each": ids}}, "$set": {"updated_at": datetime.utcnow()}},
        )


def sync_with_run_state(repo, player_id: str) -> dict[str, Any]:
    """The loop record, up to date with the current run: keepsakes unlocked so far this loop
    are kept (so Continue on any device still has them), and the loop is closed if the run
    is already finished (runs finished before loops existed, or a finish whose loop update
    didn't land)."""
    doc = get(repo, player_id)
    state = repo.get_run_state(player_id)
    if state and not doc.get("current_finished"):
        new = [m for m in unlocks_from_records(state.get("quest_records") or []) if m not in (doc.get("unlocked_mementos") or [])]
        if new:
            _unlock(repo, player_id, new)
            doc = get(repo, player_id)
        if state.get("run_finished"):
            doc = complete_loop(repo, player_id)
    return doc


def complete_loop(repo, player_id: str) -> dict[str, Any]:
    """End of a run: write the loop into the journal, bank its stars, keep the unlocks,
    remember what the player had. Safe to call twice - the second call changes nothing."""
    doc = get(repo, player_id)
    if doc.get("current_finished"):
        return doc

    state = repo.get_run_state(player_id) or {}
    records = state.get("quest_records") or []
    passed = sorted({str(r.get("quest_id")) for r in records if r.get("completed") and r.get("quest_id")})
    failed = sorted({str(r.get("quest_id")) for r in records if not r.get("completed") and r.get("quest_id")} - set(passed))
    _unlock(repo, player_id, unlocks_from_records(records))
    earned = stars_from_records(records)
    doc = get(repo, player_id)

    latest = repo.find_latest_run_for_player(player_id) or {}
    loop_no = int(doc.get("current_loop") or 1)
    threat = doc.get("threat") or ""
    now = datetime.utcnow()

    entry = {
        "loop": loop_no,
        "threat": threat,
        "holland_code": str(latest.get("cluster_holland_code") or latest.get("holland_code") or ""),
        "career": str(latest.get("career_result") or latest.get("career_family") or latest.get("cluster_label") or ""),
        "passed": len(passed),
        "failed": len(failed),
        "kept_skills": list(doc.get("kept_skills") or []),
        "mementos": list(doc.get("loop_start_mementos") or []),
        "stars_earned": earned,
        "finished_at": now,
    }

    seen = list(doc.get("threats_seen") or [])
    if threat and threat not in seen:
        seen.append(threat)

    repo.player_loop_meta.update_one(
        {"player_id": player_id},
        {
            "$set": {
                "current_finished": True,
                "loops_completed": loop_no,
                "threats_seen": seen,
                "last_owned_skills": list(state.get("owned_skills") or []),
                "last_passed": passed,
                "last_failed": failed,
                "updated_at": now,
            },
            "$inc": {"stars": earned},
            "$push": {"history": entry},
        },
    )

    # Label the finished run itself, so analysis can separate loop 1 from later loops.
    if latest.get("_id") is not None:
        repo.session_runs.update_one(
            {"_id": latest["_id"]},
            {"$set": {
                "loop": loop_no,
                "threat": threat,
                "kept_skills": entry["kept_skills"],
                "mementos": entry["mementos"],
            }},
        )
    return get(repo, player_id)


def buy_memento(repo, player_id: str, memento_id: str) -> dict[str, Any]:
    """Shop: buy an unlocked keepsake with banked stars."""
    if memento_id not in MEMENTOS:
        raise LoopError("Unknown keepsake.")
    doc = sync_with_run_state(repo, player_id)
    if memento_id in (doc.get("owned_mementos") or []):
        return doc
    if memento_id not in (doc.get("unlocked_mementos") or []):
        raise LoopError("Locked.")
    cost = MEMENTOS[memento_id][1]
    # Atomic: only spends if the stars are there.
    result = repo.player_loop_meta.update_one(
        {"player_id": player_id, "stars": {"$gte": cost}},
        {"$inc": {"stars": -cost}, "$addToSet": {"owned_mementos": memento_id},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise LoopError("Not enough stars.")
    return get(repo, player_id)


def begin_loop(repo, player_id: str, kept_skills: list[str]) -> dict[str, Any]:
    """Go Back: start the next loop with the skills the player chose to remember. Every
    keepsake they own comes along."""
    doc = sync_with_run_state(repo, player_id)
    if not doc.get("current_finished"):
        raise LoopError("Finish this loop first.")

    owned = list(doc.get("owned_mementos") or [])
    allowed = set(doc.get("last_owned_skills") or []) | set(doc.get("kept_skills") or [])
    kept = [s for s in dict.fromkeys(kept_skills or []) if s in allowed][:memory_slots(owned)]

    previous = doc.get("threat")
    repo.player_loop_meta.update_one(
        {"player_id": player_id},
        {"$set": {
            "current_loop": int(doc.get("current_loop") or 1) + 1,
            "current_finished": False,
            "threat": _roll_threat(previous, doc.get("threats_seen") or []),
            "kept_skills": kept,
            "loop_start_mementos": owned,   # what this loop started with (for the run label)
            "updated_at": datetime.utcnow(),
        }},
    )
    # The finished run's save is done with (its records are already on the finished
    # session_run). Clearing it here too means a stale "finished" save can never close the
    # new loop, even if the game's own reset call didn't land.
    repo.clear_run_state(player_id)
    return get(repo, player_id)
