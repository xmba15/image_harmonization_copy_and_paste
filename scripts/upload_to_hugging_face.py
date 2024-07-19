import argparse
import os

from huggingface_hub import HfApi, create_repo


def get_args():
    parser = argparse.ArgumentParser("upload model weights to huggingface hub")
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--model_weights_path", type=str, required=True)
    parser.add_argument("--path_in_repo", type=str, required=True)
    parser.add_argument("--commit_message", type=str, required=True)

    parser.add_argument("to_create_repo", action="store_false")

    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.isfile(args.model_weights_path)

    if args.to_create_repo:
        create_repo(repo_id=args.repo_id)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=args.model_weights_path,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
