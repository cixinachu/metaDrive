"""SR-AIRL constants supplied for the current study (SI units)."""
FRAME_STATE_DIM = 16
HISTORY_LEN = 11

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

# Verified against upstream be71aade7acff6455fffc8f1143953ac72bdbca6
# Step1_Reward_Recovery/reward_config.py on 2026-11-09.
DT = 0.1
VEL_MIN = VEL_OFFSET
ACC_SCALE = 10.0
JERK_SCALE = 20.0
PRED_HORIZON = 1.0
PRED_STEPS = 11

# Counterfactual preference supervision.
RANK_LAMBDA = 0.1
RANK_TEMPERATURE = 0.1
RANK_BATCH_SIZE = 16
RANK_WINDOW = 20

MAX_EPISODES = 15000
SEED = 42
