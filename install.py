import os
import subprocess
import requests
import tempfile
import shutil
from tqdm.auto import tqdm


def download_installer(url, dir):
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    file_size = int(response.headers.get('Content-Length', 0))

    with tqdm.wrapattr(response.raw, "read", total=file_size, desc=url) as read:
        dest = os.path.join(dir, "NIMSetup.exe")
        with open(dest, "wb") as f:
            shutil.copyfileobj(read, f)

        return dest


def run_installer(installer_path):
    # Run the installer via powershell with elevated privileges and wait for it to finish
    print("Launching installer...")
    powershell_command = [
        "powershell.exe",
        "-Command",
        f"""
        try {{
            $process = Start-Process '{installer_path}' -Verb RunAs -Wait -PassThru
            exit $process.ExitCode
        }} catch {{
            exit 403  # UAC cancellation
        }}
        """
    ]

    print("Waiting for completion...")
    try: 
        result = subprocess.run(powershell_command, check=True)
        if result.returncode == 0:
            print("Install NIMSetup successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install with error code {e.returncode}")

