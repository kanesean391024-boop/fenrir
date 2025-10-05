import json
from config_loader import (
    save_globals,
    load_globals,
    save_pdvs,
    load_pdvs,
    save_agents,
    load_agents,
    save_classes,
    load_classes,
    save_state,
    load_state,
)


def test_roundtrip(tmp_path):
    g = {
        "debug_level": "INFO",
        "model": "m",
        "temperature": 0.5,
        "system_prompt": "",
        "pre_context_message": "",
        "post_context_message": "",
        "max_context_tokens": 100,
        "pdv_gamma": 2.0,
    }
    gp = tmp_path / "globals.json"
    save_globals(g, str(gp))
    assert load_globals(str(gp)) == g
    assert not (tmp_path / "globals.json.tmp").exists()

    pd = {"pdvs": [{"name": "p", "description": "", "value": 0.5}]}
    pp = tmp_path / "pdvs.json"
    save_pdvs({"p": pd["pdvs"][0]}, str(pp))
    assert load_pdvs(str(pp)) == {"p": pd["pdvs"][0]}

    cl = {"classes": [{"name": "c", "triggering_pdv": "p", "pdv_adjustments": []}]}
    cp = tmp_path / "classes.json"
    save_classes({"c": cl["classes"][0]}, str(cp))
    assert load_classes(str(cp)) == {"c": cl["classes"][0]}

    ag = {
        "agents": [
            {
                "name": "a",
                "agent_class": "c",
                "groups_in": ["x"],
                "groups_out": ["y"],
            }
        ]
    }
    ap = tmp_path / "agents.json"
    save_agents(ag["agents"], str(ap))
    assert load_agents(str(ap)) == ag["agents"]

    st = {"current_agent": "a"}
    sp = tmp_path / "state.json"
    save_state(st, str(sp))
    assert load_state(str(sp)) == st


def test_schema_error(tmp_path):
    gp = tmp_path / "globals.json"
    gp.write_text("{}", encoding="utf-8")
    try:
        load_globals(str(gp))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
