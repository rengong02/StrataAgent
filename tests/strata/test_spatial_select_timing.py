import json

from scripts import measure_spatial_select as timing


def test_selector_stage_does_not_run_an_extra_plot(monkeypatch, tmp_path):
    state = {'data_store': {'countries': [None] * 45, 'earthquakes': [None] * 9176}}
    plot_calls = []
    selection_calls = []

    class LoadStub:
        name = 'load_geodata'

        def execute(self, **arguments):
            return f"Loaded {arguments['output_geodataframe_name']}"

    class SelectStub:
        name = 'select_features_by_spatial_relationship'

        def execute(self, **arguments):
            selection_calls.append(arguments['plot_result'])
            state['data_store'][arguments['output_geodataframe_name']] = [None] * 45
            return 'Selected 45 features'

    def fake_plot(reference, selected, predicates):
        plot_calls.append((reference, selected, predicates))
        return {'png_bytes': 123, 'plotted_rows': len(selected)}

    destination = tmp_path / 'timing.json'
    monkeypatch.setattr(timing, 'benchmark_state', lambda: state)
    monkeypatch.setattr(timing, 'build_geobench_tools',
                        lambda **arguments: [LoadStub(), SelectStub()])
    monkeypatch.setattr(timing, 'plot_like_the_tool', fake_plot)
    monkeypatch.setattr(timing.sys, 'argv',
                        ['probe', '--skip-duplicate-plot', '--out', str(destination)])

    assert timing.main() == 0
    assert selection_calls == [False, True]
    assert len(plot_calls) == 1
    result = json.loads(destination.read_text())
    selector = next(stage for stage in result['stages']
                    if stage['stage'] == 'tool_end_to_end_with_plot')
    assert selector['note']['rows_selected'] == 45
    assert 'png_bytes' not in selector['note']
