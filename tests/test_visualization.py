"""
Unit tests for 3D visualization and sample state generation — Phase 4.

Covers:
- SampleState creation and properties
- create_sample pipeline
- plot_sample_3d rendering for percolating and non-percolating samples
- plot_sample_3d toggle parameters (electrodes, box, tubes, contacts, cluster, path)
- plot_sample_3d saving to file (PNG)
- plot_sample_3d with custom Axes3D
- Conversion to SimulationResult
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from percolation.domain.cnt import MaterialBox, Nanotube, SampleState
from percolation.graph.contact_graph import build_graph, find_contacts
from percolation.simulation.experiment import create_sample
from percolation.visualization.renderer import plot_sample_3d

# Ensure headless matplotlib backend for test execution
matplotlib.use("Agg")


@pytest.fixture
def sample_box() -> MaterialBox:
    return MaterialBox(size_x=10.0, size_y=5.0, size_z=5.0)


@pytest.fixture
def synthetic_percolating_sample(sample_box: MaterialBox) -> SampleState:
    """Construct a known percolating sample:
    t0 (touches LEFT), t1 (bridge), t2 (touches RIGHT), t3 (isolated).
    """
    t0 = Nanotube(
        center=np.array([1.5, 2.5, 2.5]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=3.0,
        diameter=0.1,
    )
    t1 = Nanotube(
        center=np.array([5.0, 2.5, 2.5]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=4.5,
        diameter=0.1,
    )
    t2 = Nanotube(
        center=np.array([8.5, 2.5, 2.5]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=3.0,
        diameter=0.1,
    )
    t3 = Nanotube(
        center=np.array([5.0, 4.5, 4.5]),
        direction=np.array([0.0, 1.0, 0.0]),
        length=1.0,
        diameter=0.1,
    )
    tubes = [t0, t1, t2, t3]
    contacts = find_contacts(tubes, threshold=0.1)
    graph = build_graph(tubes, contacts, sample_box, electrode_threshold=0.1)

    from percolation.graph.percolation import (
        check_percolation,
        find_percolating_cluster,
        find_percolating_path,
    )

    percolates = check_percolation(graph)
    cluster = find_percolating_cluster(graph)
    path = find_percolating_path(graph)

    return SampleState(
        box=sample_box,
        tubes=tubes,
        contacts=contacts,
        graph=graph,
        percolates=percolates,
        percolating_cluster=cluster,
        percolating_path=path,
        concentration=0.005,
        cnt_length=3.0,
        cnt_diameter=0.1,
        contact_threshold=0.1,
    )


@pytest.fixture
def synthetic_non_percolating_sample(sample_box: MaterialBox) -> SampleState:
    """Construct a known non-percolating sample: two isolated tubes."""
    t0 = Nanotube(
        center=np.array([2.0, 2.5, 2.5]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=2.0,
        diameter=0.1,
    )
    t1 = Nanotube(
        center=np.array([8.0, 2.5, 2.5]),
        direction=np.array([1.0, 0.0, 0.0]),
        length=2.0,
        diameter=0.1,
    )
    tubes = [t0, t1]
    contacts = find_contacts(tubes, threshold=0.1)
    graph = build_graph(tubes, contacts, sample_box, electrode_threshold=0.1)

    return SampleState(
        box=sample_box,
        tubes=tubes,
        contacts=contacts,
        graph=graph,
        percolates=False,
        percolating_cluster=set(),
        percolating_path=None,
        concentration=0.002,
        cnt_length=2.0,
        cnt_diameter=0.1,
        contact_threshold=0.1,
    )


class TestCreateSample:
    def test_create_sample_returns_valid_state(self, sample_box: MaterialBox):
        rng = np.random.default_rng(42)
        sample = create_sample(
            concentration=0.005,
            box=sample_box,
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            rng=rng,
        )
        assert isinstance(sample, SampleState)
        assert sample.nanotube_count > 0
        assert sample.nanotube_count == len(sample.tubes)
        assert sample.contact_count == len(sample.contacts)
        assert isinstance(sample.percolates, bool)
        assert isinstance(sample.percolating_cluster, set)
        if sample.percolates:
            assert len(sample.percolating_cluster) > 0
            assert sample.percolating_path is not None
        else:
            assert len(sample.percolating_cluster) == 0

    def test_sample_state_to_simulation_result(self, sample_box: MaterialBox):
        rng = np.random.default_rng(123)
        sample = create_sample(
            concentration=0.004,
            box=sample_box,
            cnt_length=1.0,
            cnt_diameter=0.1,
            contact_threshold=0.05,
            rng=rng,
        )
        sim_res = sample.to_simulation_result()
        assert sim_res.concentration == sample.concentration
        assert sim_res.nanotube_count == sample.nanotube_count
        assert sim_res.percolates == sample.percolates
        assert sim_res.contact_count == sample.contact_count


class TestPlotSample3D:
    def test_render_percolating_sample(self, synthetic_percolating_sample: SampleState):
        fig = plot_sample_3d(synthetic_percolating_sample)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_non_percolating_sample(self, synthetic_non_percolating_sample: SampleState):
        fig = plot_sample_3d(synthetic_non_percolating_sample)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_options_toggles(self, synthetic_percolating_sample: SampleState):
        """Verify that turning off components runs cleanly without errors."""
        fig = plot_sample_3d(
            synthetic_percolating_sample,
            show_electrodes=False,
            show_box=False,
            show_tubes=False,
            show_contacts=False,
            highlight_cluster=False,
            highlight_path=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_custom_axes(self, synthetic_percolating_sample: SampleState):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        returned_fig = plot_sample_3d(
            synthetic_percolating_sample,
            ax=ax,
            title="Custom Title",
        )
        assert returned_fig is fig
        plt.close(fig)

    def test_save_figure_to_disk(
        self, synthetic_percolating_sample: SampleState, tmp_path: Path
    ):
        out_file = tmp_path / "test_render.png"
        fig = plot_sample_3d(
            synthetic_percolating_sample,
            save_path=out_file,
            dpi=100,
        )
        assert out_file.exists()
        assert out_file.stat().st_size > 0
        plt.close(fig)
