import conductor


def test_select_next_agent():
    conductor.save_agents = lambda *a, **k: None
    conductor.PDVS = {"talk": 0.9}
    conductor.CLASSES = {
        "ruminator": {"triggering_pdv": "talk", "pdv_adjustments": []},
        "speaker": {"triggering_pdv": "talk", "pdv_adjustments": []},
    }
    ava = {
        "name": "Ava",
        "agent_class": "ruminator",
        "groups_out": ["G"],
        "groups_in": [],
    }
    echo = {
        "name": "Echo",
        "agent_class": "speaker",
        "groups_in": ["G"],
        "groups_out": ["Core"],
    }
    sink = {
        "name": "Sink",
        "agent_class": "speaker",
        "groups_in": ["Core"],
        "groups_out": [],
    }
    conductor.AGENTS = [ava, echo, sink]
    conductor.AGENTS_BY_NAME = {"Ava": ava, "Echo": echo, "Sink": sink}
    conductor.AGENTS_BY_GROUP_IN = {"G": {"Echo"}, "Core": {"Sink"}}
    nxt = conductor.select_next_agent("Ava")
    assert nxt["name"] == "Echo"


def test_dead_end_flagging():
    conductor.save_agents = lambda *a, **k: None
    conductor.PDVS = {"talk": 0.5}
    conductor.CLASSES = {"ruminator": {"triggering_pdv": "talk", "pdv_adjustments": []}}
    solo = {
        "name": "Solo",
        "agent_class": "ruminator",
        "groups_out": ["G2"],
        "groups_in": [],
    }
    conductor.AGENTS = [solo]
    conductor.AGENTS_BY_NAME = {"Solo": solo}
    conductor.AGENTS_BY_GROUP_IN = {}
    res = conductor.select_next_agent("Solo")
    assert res is None
    assert solo.get("flag_no_downstream") is True
    assert solo.get("missing_out_groups") == ["G2"]
