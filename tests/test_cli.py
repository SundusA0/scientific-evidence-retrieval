from sci_evidence.cli import main


def test_offline_demo(capsys):
    main(["demo"])
    text = capsys.readouterr().out
    assert "OFFLINE DEMO" in text
    assert "Toy metrics" in text
    assert "Solar panels" in text
