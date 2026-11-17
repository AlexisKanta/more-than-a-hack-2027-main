import numpy as np
import pytest

from bsd.io_epts import read_epts

META = """<?xml version="1.0"?>
<main xmlns:tns="ns"><Metadata>
<GlobalConfig><ProviderGlobalParameters>
<ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
<ProviderParameter><Name>first_half_end</Name><Value>2</Value></ProviderParameter>
<ProviderParameter><Name>second_half_start</Name><Value>3</Value></ProviderParameter>
<ProviderParameter><Name>second_half_end</Name><Value>4</Value></ProviderParameter>
<ProviderParameter><Name>first_extra_half_start</Name><Value/></ProviderParameter>
</ProviderGlobalParameters></GlobalConfig>
<Teams><Team id="TA"/><Team id="TB"/></Teams>
<Players><Player id="p1" teamId="TA"/><Player id="p2" teamId="TB"/><Player id="p3" teamId="TA"/></Players>
<PlayerChannels>
<PlayerChannel channelId="x" id="a_x" playerId="p1"/><PlayerChannel channelId="y" id="a_y" playerId="p1"/>
<PlayerChannel channelId="x" id="b_x" playerId="p2"/><PlayerChannel channelId="y" id="b_y" playerId="p2"/>
<PlayerChannel channelId="x" id="c_x" playerId="p3"/><PlayerChannel channelId="y" id="c_y" playerId="p3"/>
</PlayerChannels></Metadata></main>
"""

RAW = """1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,0.0:0.25,0.75
2:0.5,0.5;0.0,1.0;NaN,NaN:NaN,NaN
3:0.5,0.5;0.0,1.0;0.3,0.3:0.5,0.5
4:0.5,0.5;0.0,1.0;0.3,0.3:0.5,0.5
"""


@pytest.fixture
def tr(tmp_path):
    (tmp_path / "m.xml").write_text(META)
    (tmp_path / "r.txt").write_text(RAW)
    return read_epts(tmp_path / "r.txt", tmp_path / "m.xml", (105.3, 68.0))


def test_epts_first_record_is_index_0(tr):
    assert tr.players.shape == (4, 3, 2)
    assert tr.players[0, 2, 0] == pytest.approx(0.1 * 105.3, abs=1e-3)


def test_epts_metres(tr):
    assert tr.players[0, 0] == pytest.approx([52.65, 34.0], abs=1e-3)
    assert tr.ball[0] == pytest.approx([0.25 * 105.3, 0.75 * 68.0], abs=1e-3)


def test_epts_nan_and_two_value_groups(tr):
    assert np.isnan(tr.players[0, 1]).all() and np.isnan(tr.ball[1]).all()
    assert tr.players[1, 0] == pytest.approx([52.65, 34.0], abs=1e-3)
    assert np.isnan(tr.players[1, 2]).all()


def test_epts_team_from_meta(tr):
    assert tr.team.tolist() == [0, 1, 0]


def test_epts_periods_from_meta(tr):
    assert tr.periods == [(1, 0, 1), (2, 2, 3)]


SPEC_META = META.replace(
    "</Metadata>",
    """<DataFormatSpecifications>
<DataFormatSpecification startFrame="1" endFrame="2" separator=":">
<StringRegister name="frameCount"/>
<SplitRegister separator=";">
<SplitRegister separator=","><PlayerChannelRef playerChannelId="c_x"/><PlayerChannelRef playerChannelId="c_y"/></SplitRegister>
<SplitRegister separator=","><PlayerChannelRef playerChannelId="a_x"/><PlayerChannelRef playerChannelId="a_y"/></SplitRegister>
</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
</DataFormatSpecification>
<DataFormatSpecification startFrame="3" endFrame="4" separator=":">
<StringRegister name="frameCount"/>
<SplitRegister separator=";">
<SplitRegister separator=","><PlayerChannelRef playerChannelId="b_x"/><PlayerChannelRef playerChannelId="b_y"/></SplitRegister>
</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>
</DataFormatSpecification>
</DataFormatSpecifications></Metadata>""",
)
SPEC_RAW = """1:0.1,0.2;0.5,0.5:0.5,0.5
2:0.3,0.4;NaN,NaN:0.5,0.5
3:0.0,1.0:0.5,0.5
4:1.0,0.0:0.5,0.5
"""


def _write(tmp_path, meta, raw):
    (tmp_path / "m.xml").write_text(meta)
    (tmp_path / "r.txt").write_text(raw)
    return tmp_path / "r.txt", tmp_path / "m.xml"


def test_epts_spec_remaps_groups_to_slots(tmp_path):
    raw, meta = _write(tmp_path, SPEC_META, SPEC_RAW)
    t = read_epts(raw, meta, (100.0, 50.0))
    assert t.players.shape == (4, 3, 2)
    # frame 1: group0 -> slot 2, group1 -> slot 0; slot 1 absent
    assert t.players[0, 2] == pytest.approx([10.0, 10.0], abs=1e-3)
    assert t.players[0, 0] == pytest.approx([50.0, 25.0], abs=1e-3)
    assert np.isnan(t.players[0, 1]).all()
    assert t.players[1, 2] == pytest.approx([30.0, 20.0], abs=1e-3)
    assert np.isnan(t.players[1, 0]).all()
    # second range: single group -> slot 1
    assert t.players[2, 1] == pytest.approx([0.0, 50.0], abs=1e-3)
    assert np.isnan(t.players[2, 0]).all() and np.isnan(t.players[2, 2]).all()
    assert t.players[3, 1] == pytest.approx([100.0, 0.0], abs=1e-3)


def test_epts_spec_mismatch_raises(tmp_path):
    raw, meta = _write(tmp_path, SPEC_META, SPEC_RAW.replace("3:0.0,1.0:", "3:0.0,1.0;0.1,0.1:"))
    with pytest.raises(ValueError, match="DataFormatSpecification"):
        read_epts(raw, meta, (100.0, 50.0))


def test_epts_frame_gap_raises(tmp_path):
    raw, meta = _write(tmp_path, META, RAW.replace("3:", "5:"))
    with pytest.raises(ValueError, match="frame id"):
        read_epts(raw, meta, (105.3, 68.0))
