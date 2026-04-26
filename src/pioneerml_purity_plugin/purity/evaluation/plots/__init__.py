from __future__ import annotations

from typing import Any, Callable

from .endpoint_histograms_by_particle import (
    PurityEndpointErrorHistogramsByParticlePlot,
    PurityEndpointPredTruthHistogramsByParticlePlot,
)
from .endpoint_mse_curves import (
    PurityEndpointMSECurvesPlot,
    PurityEndpointMSEByParticleCurvesPlot,
)
from .pdg_accuracy_curves import PurityPDGAccuracyCurvesPlot
from .phase_loss_curves import (
    PurityPhase1LossCurvesPlot,
    PurityPhase2LossCurvesPlot,
    PurityPhase3LossCurvesPlot,
)
from .staged_loss_curves import PurityStagedLossCurvesPlot
from .task_aux_curves import PurityTaskAuxCurvesPlot
from .task_loss_curves import PurityTaskLossCurvesPlot
from .task_metric_curves import PurityTaskMetricCurvesPlot
from .target_diagnostics_shared import collect_purity_target_diagnostics
from .target_event_builder_score_hist_plot import PurityEventBuilderScoreHistPlot
from .target_node_pdg_class_accuracy_plot import PurityNodePDGClassAccuracyPlot
from .target_pion_stop_error_hist_plot import PurityPionStopErrorHistPlot
from .target_positron_angle_error_hist_plot import PurityPositronAngleErrorHistPlot
from .target_positron_energy_error_hist_plot import PurityPositronEnergyErrorHistPlot
from .target_positron_energy_scatter_plot import PurityPositronEnergyScatterPlot
from .target_positron_theta_scatter_plot import PurityPositronThetaScatterPlot
from .target_positron_time_error_hist_plot import PurityPositronTimeErrorHistPlot
from .target_positron_time_scatter_plot import PurityPositronTimeScatterPlot
from .target_slice_multi_score_hist_plot import PuritySliceMultiScoreHistPlot
from .target_slice_pdg_class_accuracy_plot import PuritySlicePDGClassAccuracyPlot
from .target_summary_energy_scatter_plot import PuritySummaryEnergyScatterPlot
from .target_summary_time_scatter_plot import PuritySummaryTimeScatterPlot
from .target_trigger_slice_score_hist_plot import PurityTriggerSliceScoreHistPlot


def _wrap(cls) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        return cls().render(*args, **kwargs)

    return _fn


plot_purity_staged_loss_curves = _wrap(PurityStagedLossCurvesPlot)
plot_purity_pdg_accuracy_curves = _wrap(PurityPDGAccuracyCurvesPlot)
plot_purity_endpoint_mse_curves = _wrap(PurityEndpointMSECurvesPlot)
plot_purity_endpoint_mse_by_particle_curves = _wrap(PurityEndpointMSEByParticleCurvesPlot)
plot_purity_endpoint_pred_truth_histograms_by_particle = _wrap(PurityEndpointPredTruthHistogramsByParticlePlot)
plot_purity_endpoint_error_histograms_by_particle = _wrap(PurityEndpointErrorHistogramsByParticlePlot)
plot_purity_phase_1_loss_curves = _wrap(PurityPhase1LossCurvesPlot)
plot_purity_phase_2_loss_curves = _wrap(PurityPhase2LossCurvesPlot)
plot_purity_phase_3_loss_curves = _wrap(PurityPhase3LossCurvesPlot)
plot_purity_node_pdg_class_accuracy = _wrap(PurityNodePDGClassAccuracyPlot)
plot_purity_slice_pdg_class_accuracy = _wrap(PuritySlicePDGClassAccuracyPlot)
plot_purity_trigger_slice_score_hist = _wrap(PurityTriggerSliceScoreHistPlot)
plot_purity_slice_multi_score_hist = _wrap(PuritySliceMultiScoreHistPlot)
plot_purity_event_builder_score_hist = _wrap(PurityEventBuilderScoreHistPlot)
plot_purity_pion_stop_error_hist = _wrap(PurityPionStopErrorHistPlot)
plot_purity_positron_angle_error_hist = _wrap(PurityPositronAngleErrorHistPlot)
plot_purity_positron_theta_scatter = _wrap(PurityPositronThetaScatterPlot)
plot_purity_positron_energy_scatter = _wrap(PurityPositronEnergyScatterPlot)
plot_purity_positron_energy_error_hist = _wrap(PurityPositronEnergyErrorHistPlot)
plot_purity_positron_time_scatter = _wrap(PurityPositronTimeScatterPlot)
plot_purity_positron_time_error_hist = _wrap(PurityPositronTimeErrorHistPlot)
plot_purity_summary_energy_scatter = _wrap(PuritySummaryEnergyScatterPlot)
plot_purity_summary_time_scatter = _wrap(PuritySummaryTimeScatterPlot)
plot_purity_task_aux_curves = _wrap(PurityTaskAuxCurvesPlot)
plot_purity_task_loss_curves = _wrap(PurityTaskLossCurvesPlot)
plot_purity_task_metric_curves = _wrap(PurityTaskMetricCurvesPlot)

__all__ = [
    "PurityStagedLossCurvesPlot",
    "PurityPDGAccuracyCurvesPlot",
    "PurityEndpointMSECurvesPlot",
    "PurityEndpointMSEByParticleCurvesPlot",
    "PurityEndpointErrorHistogramsByParticlePlot",
    "PurityEndpointPredTruthHistogramsByParticlePlot",
    "PurityPhase1LossCurvesPlot",
    "PurityPhase2LossCurvesPlot",
    "PurityPhase3LossCurvesPlot",
    "PurityNodePDGClassAccuracyPlot",
    "PuritySlicePDGClassAccuracyPlot",
    "PurityTriggerSliceScoreHistPlot",
    "PuritySliceMultiScoreHistPlot",
    "PurityEventBuilderScoreHistPlot",
    "PurityPionStopErrorHistPlot",
    "PurityPositronAngleErrorHistPlot",
    "PurityPositronThetaScatterPlot",
    "PurityPositronEnergyScatterPlot",
    "PurityPositronEnergyErrorHistPlot",
    "PurityPositronTimeScatterPlot",
    "PurityPositronTimeErrorHistPlot",
    "PuritySummaryEnergyScatterPlot",
    "PuritySummaryTimeScatterPlot",
    "PurityTaskAuxCurvesPlot",
    "PurityTaskLossCurvesPlot",
    "PurityTaskMetricCurvesPlot",
    "collect_purity_target_diagnostics",
    "plot_purity_staged_loss_curves",
    "plot_purity_pdg_accuracy_curves",
    "plot_purity_endpoint_mse_curves",
    "plot_purity_endpoint_mse_by_particle_curves",
    "plot_purity_endpoint_error_histograms_by_particle",
    "plot_purity_endpoint_pred_truth_histograms_by_particle",
    "plot_purity_phase_1_loss_curves",
    "plot_purity_phase_2_loss_curves",
    "plot_purity_phase_3_loss_curves",
    "plot_purity_node_pdg_class_accuracy",
    "plot_purity_slice_pdg_class_accuracy",
    "plot_purity_trigger_slice_score_hist",
    "plot_purity_slice_multi_score_hist",
    "plot_purity_event_builder_score_hist",
    "plot_purity_pion_stop_error_hist",
    "plot_purity_positron_angle_error_hist",
    "plot_purity_positron_theta_scatter",
    "plot_purity_positron_energy_scatter",
    "plot_purity_positron_energy_error_hist",
    "plot_purity_positron_time_scatter",
    "plot_purity_positron_time_error_hist",
    "plot_purity_summary_energy_scatter",
    "plot_purity_summary_time_scatter",
    "plot_purity_task_aux_curves",
    "plot_purity_task_loss_curves",
    "plot_purity_task_metric_curves",
]
