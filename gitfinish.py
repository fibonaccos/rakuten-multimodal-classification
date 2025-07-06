import subprocess
import sys


REPO_URL = "https://github.com/fibonaccos/rakuten-multimodal-classification.git"


def run_cmd(cmd, check=True):
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"Error while executing command : {cmd}")
        sys.exit(1)


def get_current_branch():
    result = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def main():
    branch = get_current_branch()
    print(f"Current branch : {branch}")
    
    confirm = input("Please confirm you want to push changes to this branch (y/Y to confirm) : ").strip().lower()
    if confirm != "y":
        print("Push cancelled.")
        sys.exit(0)

    run_cmd("git add .")

    msg = input("Enter commit message : ").strip()
    if not msg:
        msg = "Empty"

    run_cmd(f'git commit -m "{msg}"', check=False)
    run_cmd(f"git push origin {branch}")

    print(f"Changes on branch `{branch}` successfully pushed.")
    print(f"You can now create a pull request on {REPO_URL} to merge your changes on branch `dev`.")


if __name__ == "__main__":
    main()
