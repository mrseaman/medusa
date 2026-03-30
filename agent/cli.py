"""CLI chat loop for the MEDUSA agent."""

import json
import logging
import readline
import sys

from .models import ModelRegistry
from .state import SessionState
from .client import AgentClient

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def print_tool_call(name, args, result):
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
    print(f"  {DIM}> {name}({args_str}){RESET}")

    if isinstance(result, dict) and "error" in result:
        print(f"  {YELLOW}  error: {result['error']}{RESET}")


def print_status(state: SessionState):
    print(f"\n{BOLD}Session Status{RESET}")
    if state.source_path:
        print(f"  Loaded: {state.source_path}")
    else:
        print("  No spectrum loaded")

    if state.spectrum is not None:
        spec = state.spectrum
        print(f"  Points: {len(spec.masses)}")
        print(f"  Mass range: {spec.masses.min():.4f} - {spec.masses.max():.4f}")

    if state.labels is not None:
        n = state.get_cluster_count()
        print(f"  Clusters: {n}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MEDUSA Analysis Agent")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    print(f"{BOLD}MEDUSA Analysis Agent{RESET}")
    print("Loading models...", end=" ", flush=True)

    registry = ModelRegistry()
    registry.load_all()

    n_deiso = len(registry.deisotoping_models)
    mlp_ok = registry.mlp_model is not None
    print(f"done ({n_deiso} deisotoping models, MLP: {'yes' if mlp_ok else 'no'})")
    print(f"Using LLM: {args.model}")
    print(f"Commands: /quit, /status, /reset\n")

    state = SessionState()
    client = AgentClient(state, registry, model=args.model)

    # readline history
    try:
        readline.read_history_file(".medusa_history")
    except FileNotFoundError:
        pass

    while True:
        try:
            user_input = input(f"{GREEN}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break
        elif user_input == "/status":
            print_status(state)
            continue
        elif user_input == "/reset":
            state.reset()
            client.reset_conversation()
            print(f"{DIM}Session reset.{RESET}\n")
            continue

        try:
            response = client.chat(user_input, on_tool_call=print_tool_call)
            print(f"\n{CYAN}Agent:{RESET} {response}\n")
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}\n")
        except Exception as e:
            print(f"\n{YELLOW}Error: {e}{RESET}\n")

    try:
        readline.write_history_file(".medusa_history")
    except Exception:
        pass


if __name__ == "__main__":
    main()
