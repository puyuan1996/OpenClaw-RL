#!/usr/bin/env python3
import json
import os

import ray

from slime.utils.arguments import parse_args


def main():
    address = os.environ.get("RAY_HEAD_ADDR") or os.environ.get("RAY_ADDRESS") or "auto"
    runtime_env = None
    runtime_env_json = os.environ.get("OPENCLAW_RUNTIME_ENV_JSON")
    if runtime_env_json:
        runtime_env = json.loads(runtime_env_json)
    ray.init(address=address, runtime_env=runtime_env, log_to_driver=True)
    args = parse_args()
    if args.colocate:
        from train import train
    else:
        from train_async import train

    train(args)


if __name__ == "__main__":
    main()
