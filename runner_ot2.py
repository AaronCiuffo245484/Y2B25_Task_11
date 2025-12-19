import subprocess
import sys

# CONFIGURATIONS
TOTAL_TIMESTEPS = 100000
N_STEPS = 2048
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

TRAIN_SCRIPT = "viktor_train_ot2.py"


def main():
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--total_timesteps", str(TOTAL_TIMESTEPS),
        "--n_steps", str(N_STEPS),
        "--batch_size", str(BATCH_SIZE),
    ]

    if LEARNING_RATE is not None:
        cmd += ["--learning_rate", str(LEARNING_RATE)]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
