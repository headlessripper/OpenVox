from openvox.agent.turn import Turn, ConversationHistory


def test_turn_fields():
    t = Turn("user", "hello")
    assert t.role == "user" and t.text == "hello"


def test_history_caps_and_orders():
    h = ConversationHistory(max_turns=3)
    for i in range(5):
        h.append(Turn("user", str(i)))
    assert len(h) == 3
    assert [t.text for t in h] == ["2", "3", "4"]   # last 3, oldest-first
    assert h[-1].text == "4"
