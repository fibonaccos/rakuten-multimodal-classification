import os
import subprocess
import sys


REPO_URL = "https://github.com/fibonaccos/rakuten-multimodal-classification.git"
REPO_SSH = "git@github.com:fibonaccos/rakuten-multimodal-classification.git"


def run_cmd(cmd, cwd=None, capture=False):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"Error while executing command : {cmd}")
        sys.exit(1)
    return result.stdout.strip() if capture else None


def get_git_username():
    return run_cmd("git config user.name", capture=True)


def main():
    if len(sys.argv) < 2:
        print("A name of branch is required.")
        print("For example, the command `python begin.py features-engineering` will create the branch `dev-<username>-features-engineering`.")
        sys.exit(1)

    feature_name = sys.argv[1]
    user = get_git_username().lower().replace(" ", "-")
    full_branch_name = f"dev-{user}-{feature_name}"

    print("Cloning repo ...")
    run_cmd(f"git clone {REPO_SSH}")

    repo_name = REPO_URL.rstrip('/').split('/')[-1].replace(".git", "")
    os.chdir(repo_name)

    print("Updating branch `dev` ...")
    run_cmd("git checkout dev")
    run_cmd("git pull origin dev")

    print(f"Creating branch `{full_branch_name}` ...")
    run_cmd(f"git checkout -b {full_branch_name}")
    run_cmd(f"git push -u origin {full_branch_name}")

    print(f"`{full_branch_name}` successfully created. You can now start working on your branch.")


if __name__ == "__main__":
    main()
