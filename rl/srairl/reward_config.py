"""SR-AIRL constants supplied for the current study (SI units)."""
FRAME_STATE_DIM = 16
HISTORY_LEN = 4

GAMMA = 0.99

# Anisotropic interaction kernel (meters)
L_LONG = 15.0
L_LAT = 3.5

# Rear/front directional sensitivity
DIR_LAMBDA = 0.3

# Initial interaction contribution
ALPHA_INIT = 0.1

# Current normalization
POS_LAT_SCALE = 25.0
POS_LONG_SCALE = 500.0
VEL_SCALE = 80.0
VEL_OFFSET = -40.0
