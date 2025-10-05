import conductor


def test_select_prefers_non_queue_agents_when_empty():
    conductor.save_agents = lambda *a, **k: None
    conductor._INCOMING_QUEUE.clear()
    conductor.PDVS = {"talk": 0.5}
    conductor.CLASSES = {
        "listener": {
            "triggering_pdv": "talk",
            "pdv_adjustments": [],
            "reads_message_queue": True,
        },
        "speaker": {"triggering_pdv": "talk", "pdv_adjustments": []},
        "starter": {"triggering_pdv": "talk", "pdv_adjustments": []},
    }
    start = {"name": "Start", "agent_class": "starter", "groups_out": ["G"], "groups_in": []}
    listener = {
        "name": "Listener",
        "agent_class": "listener",
        "groups_in": ["G"],
        "groups_out": ["H"],
    }
    speaker = {
        "name": "Speaker",
        "agent_class": "speaker",
        "groups_in": ["G"],
        "groups_out": ["H"],
    }
    sink = {
        "name": "Sink",
        "agent_class": "speaker",
        "groups_in": ["H"],
        "groups_out": [],
    }
    conductor.AGENTS = [start, listener, speaker, sink]
    conductor.AGENTS_BY_NAME = {a["name"]: a for a in conductor.AGENTS}
    conductor.AGENTS_BY_GROUP_IN = {"G": {"Listener", "Speaker"}, "H": {"Sink"}}
    nxt = conductor.select_next_agent("Start")
    assert nxt["name"] == "Speaker"
