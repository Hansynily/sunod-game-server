"""Loops: what carries over from one run to the next.

The story: the disaster hits at the end of every run, the player is sent back in time to
being a child, and keeps their memories. Each run is one loop. Between loops the player
keeps:

- remembered skills (picked from the skills they owned at the end of the last loop) that
  start the next loop already in their kind's slot,
- keepsakes ("mementos") bought with Sparks, which only change things around town
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

# id -> Sparks cost. Effects live in the game; none of them touch the quest timer, stars
# or scoring.
MEMENTOS = {
    "compass": 4,    # Old Compass: the arrow points to a request after a few seconds, not 30
    "ribbon": 5,     # Lucky Ribbon: the nightly skill pick always offers your least-used kind
    "locket": 6,     # Memory Locket: remember 4 skills instead of 3
    "lantern": 4,    # Lola's Lantern: your home starts one step rebuilt in every town
    "sneakers": 5,   # Old Sneakers: walk faster around town (not inside quests)
    "coin": 3,       # Lucky Coin: +2 Sparks when the loop ends
}
MAX_EQUIPPED = 2
BASE_MEMORY_SLOTS = 3


class LoopError(Exception):
    """A request that doesn't fit the player's current loop (shown to the player)."""


def memory_slots(equipped: list[str]) -> int:
    return BASE_MEMORY_SLOTS + (1 if "locket" in (equipped or []) else 0)


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
        "equipped_mementos": [],
        "owned_mementos": [],
        "sparks": 0,
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


def sync_with_run_state(repo, player_id: str) -> dict[str, Any]:
    """The loop record, with the current loop closed if the run is already finished (runs
    finished before loops existed, or a finish whose loop update didn't land)."""
    doc = get(repo, player_id)
    if not doc.get("current_finished"):
        state = repo.get_run_state(player_id)
        if state and state.get("run_finished"):
            doc = complete_loop(repo, player_id)
    return doc


def complete_loop(repo, player_id: str) -> dict[str, Any]:
    """End of a run: write the loop into the journal, pay out Sparks, remember what the
    player had. Safe to call twice - the second call changes nothing."""
    doc = get(repo, player_id)
    if doc.get("current_finished"):
        return doc

    state = repo.get_run_state(player_id) or {}
    records = state.get("quest_records") or []
    passed = sorted({str(r.get("quest_id")) for r in records if r.get("completed") and r.get("quest_id")})
    failed = sorted({str(r.get("quest_id")) for r in records if not r.get("completed") and r.get("quest_id")} - set(passed))

    latest = repo.find_latest_run_for_player(player_id) or {}
    equipped = list(doc.get("equipped_mementos") or [])
    earned = 3 + len(passed) // 6 + (2 if "coin" in equipped else 0)
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
        "mementos": equipped,
        "sparks_earned": earned,
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
            "$inc": {"sparks": earned},
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
                "mementos": equipped,
            }},
        )
    return get(repo, player_id)


def begin_loop(repo, player_id: str, kept_skills: list[str], equipped: list[str]) -> dict[str, Any]:
    """Go Back: start the next loop with the skills to remember and the keepsakes to bring."""
    doc = sync_with_run_state(repo, player_id)
    if not doc.get("current_finished"):
        raise LoopError("Finish this loop first.")

    owned = set(doc.get("owned_mementos") or [])
    equipped = [m for m in dict.fromkeys(equipped or []) if m in owned][:MAX_EQUIPPED]

    allowed = set(doc.get("last_owned_skills") or []) | set(doc.get("kept_skills") or [])
    kept = [s for s in dict.fromkeys(kept_skills or []) if s in allowed][:memory_slots(equipped)]

    previous = doc.get("threat")
    repo.player_loop_meta.update_one(
        {"player_id": player_id},
        {"$set": {
            "current_loop": int(doc.get("current_loop") or 1) + 1,
            "current_finished": False,
            "threat": _roll_threat(previous, doc.get("threats_seen") or []),
            "kept_skills": kept,
            "equipped_mementos": equipped,
            "updated_at": datetime.utcnow(),
        }},
    )
    # The finished run's save is done with (its records are already on the finished
    # session_run). Clearing it here too means a stale "finished" save can never close the
    # new loop, even if the game's own reset call didn't land.
    repo.clear_run_state(player_id)
    return get(repo, player_id)


def buy_memento(repo, player_id: str, memento_id: str) -> dict[str, Any]:
    cost = MEMENTOS.get(memento_id)
    if cost is None:
        raise LoopError("That keepsake doesn't exist.")
    doc = get(repo, player_id)
    if memento_id in (doc.get("owned_mementos") or []):
        return doc
    # Atomic: only spends if the Sparks are there.
    result = repo.player_loop_meta.update_one(
        {"player_id": player_id, "sparks": {"$gte": cost}},
        {"$inc": {"sparks": -cost}, "$addToSet": {"owned_mementos": memento_id},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise LoopError("Not enough Sparks.")
    return get(repo, player_id)
