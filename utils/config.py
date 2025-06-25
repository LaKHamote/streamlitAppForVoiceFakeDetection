import os
import re

def load_env_from_sh(file_path='components/VoCoderRecognition/scripts/env.sh'):
    """
    Loads specific environment variables (SPEAKERS, NOISE_LEVEL_LIST)
    from a given .sh file.
    """
    env_vars = {}
    if not os.path.exists(file_path):
        print(f"Warning: env.sh file not found at {file_path}. Using default empty lists.")
        return {"SPEAKERS": [], "NOISE_LEVEL_LIST": []}

    with open(file_path, 'r') as f:
        content = f.read()

    speakers_match = re.search(r'export SPEAKERS=\(([^)]*)\)', content)
    noise_match = re.search(r'export NOISE_LEVEL_LIST=\(([^)]*)\)', content)

    if speakers_match:
        speakers_raw = speakers_match.group(1).strip()
        env_vars["SPEAKERS"] = [s.strip().strip('"\'') for s in speakers_raw.split() if s.strip()]
    else:
        env_vars["SPEAKERS"] = []
        print(f"Warning: SPEAKERS not found in {file_path}. Using an empty list.")

    if noise_match:
        noise_raw = noise_match.group(1).strip()
        try:
            env_vars["NOISE_LEVEL_LIST"] = [int(n) if float(n).is_integer() else float(n) for n in noise_raw.split() if n.strip()]
        except ValueError:
            env_vars["NOISE_LEVEL_LIST"] = []
            print(f"Warning: Could not parse NOISE_LEVEL_LIST in {file_path}. Using an empty list.")
    else:
        env_vars["NOISE_LEVEL_LIST"] = []
        print(f"Warning: NOISE_LEVEL_LIST not found in {file_path}. Using an empty list.")

    return env_vars.values()