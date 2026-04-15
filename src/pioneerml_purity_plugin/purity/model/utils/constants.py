from __future__ import annotations

MODALITY_ATAR_XZ = 0
MODALITY_ATAR_YZ = 1
MODALITY_LYSO = 2

# LYSO calorimeter
NORM_POS_LYSO = 100.0  # mm
NORM_E_LYSO = 70.0  # MeV
NORM_T_LYSO = 500.0  # ns

# ATAR
NORM_POS_ATAR = 10.0  # mm
NORM_E_ATAR = 1.0  # MeV
NORM_T_ATAR = 500.0  # ns

# Coincidence / physics windows
SIGMA_COINC_NS = 2.0
TOF_NS = 0.5

# Acceptance criteria
ACCEPT_Z_MIN_MM = 1.2
ACCEPT_Z_MAX_MM = 4.8
ACCEPT_XY_MAX_MM = 8.0
ACCEPT_ANGLE_MAX_DEG = 120.0

# PDG bit flags (matches Omar unified_reco/constants.py and pileup_mixer.py).
PDG_PION = 0b000001
PDG_MUON = 0b000010
PDG_POSITRON = 0b000100
PDG_ELECTRON = 0b001000
PDG_GAMMA = 0b010000
PDG_OTHER = 0b100000

# Backward-compat alias used by plugin edge helpers.
NORM_T = NORM_T_ATAR
