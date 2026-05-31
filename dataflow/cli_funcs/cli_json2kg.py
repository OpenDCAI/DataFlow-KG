import os
import sys
import shutil
from pathlib import Path
from colorama import Fore, Style


def cli_json2kg_init(dataset: str):
    src = Path(__file__).parent / "json2kg" / f"{dataset}.jsonl"
    dst = Path(os.getcwd()) / f"{dataset}.jsonl"

    if not src.exists():
        print(f"{Fore.RED}Error: dataset '{dataset}' not found (expected {src}){Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}Initializing json2kg dataset '{dataset}' in current working directory...{Style.RESET_ALL}")

    if dst.exists():
        user_input = input(f"  {Fore.YELLOW}Warning: {dataset}.jsonl already exists. Overwrite? (y/n): {Style.RESET_ALL}").strip().lower()
        if user_input != "y":
            print(f"  Skipping {dataset}.jsonl.")
            return

    shutil.copy2(src, dst)
    print(f"{Fore.GREEN}[Copied]\nFrom: {src}\nTo: {dst}{Style.RESET_ALL}")


def cli_json2kg_autoschemakg(dataset: str):
    json2kg_dir = Path(__file__).parent / "json2kg"
    input_path = Path(os.getcwd()) / f"{dataset}.jsonl"
    output_path = Path(os.getcwd()) / "autoschemakg_output.json"

    if not input_path.exists():
        print(f"{Fore.RED}Error: {input_path} not found. Run 'dfkg json2kg init {dataset}' first.{Style.RESET_ALL}")
        return

    api_key = os.getenv("DF_API_KEY")
    api_url = os.getenv("DF_BASE_URL")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    missing = []
    if not api_key:
        missing.append("DF_API_KEY")
    if not api_url:
        missing.append("DF_BASE_URL")
    if missing:
        print(f"{Fore.RED}Error: missing environment variable(s): {', '.join(missing)}{Style.RESET_ALL}")
        return

    sys.path.insert(0, str(json2kg_dir))
    try:
        import atlas_rag
        print(f"{Fore.GREEN}✅ atlas_rag is available{Style.RESET_ALL}")
    except ImportError:
        print(f"{Fore.RED}❌ atlas_rag not available{Style.RESET_ALL}")
        print("Please run: pip install atlas-rag")
        sys.path.pop(0)
        return

    missing_pkgs = []
    for pkg in ("openai", "tenacity", "jsonschema", "json_repair", "datasets", "tqdm", "networkx", "pandas", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        print(f"{Fore.RED}❌ Missing required packages: {', '.join(missing_pkgs)}{Style.RESET_ALL}")
        print(f"Please run: pip install {' '.join(missing_pkgs)}")
        sys.path.pop(0)
        return

    print(f"{Fore.GREEN}Running AutoSchemaKG on {input_path}...{Style.RESET_ALL}")

    from autoschemakg_baseline import AutoSchemaKGBaseline
    import argparse

    runner = AutoSchemaKGBaseline()
    parser = argparse.ArgumentParser()
    runner.add_args(parser)
    args = parser.parse_args([
        "--input", str(input_path),
        "--output", str(output_path),
        "--api-key", api_key,
        "--api-url", api_url,
        "--model", model,
    ])
    result = runner.main(args)
    print(f"{Fore.GREEN}✅ Done. Output saved to {result}{Style.RESET_ALL}")


def cli_json2kg_wikontic(dataset: str):
    json2kg_dir = Path(__file__).parent / "json2kg"
    input_path = Path(os.getcwd()) / f"{dataset}.jsonl"
    output_path = Path(os.getcwd()) / "wikontic_output.json"

    if not input_path.exists():
        print(f"{Fore.RED}Error: {input_path} not found. Run 'dfkg json2kg init {dataset}' first.{Style.RESET_ALL}")
        return

    api_key = os.getenv("DF_API_KEY")
    api_url = os.getenv("DF_BASE_URL")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    missing = []
    if not api_key:
        missing.append("DF_API_KEY")
    if not api_url:
        missing.append("DF_BASE_URL")
    if missing:
        print(f"{Fore.RED}Error: missing environment variable(s): {', '.join(missing)}{Style.RESET_ALL}")
        return

    sys.path.insert(0, str(json2kg_dir))
    try:
        import wikontic
        print(f"{Fore.GREEN}✅ wikontic is available{Style.RESET_ALL}")
    except ImportError:
        print(f"{Fore.RED}❌ wikontic not available{Style.RESET_ALL}")
        sys.path.pop(0)
        return

    missing_pkgs = []
    for pkg in ("openai", "tenacity", "httpx", "pymongo", "pydantic", "transformers", "tqdm", "langchain", "unidecode"):
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        print(f"{Fore.RED}❌ Missing required packages: {', '.join(missing_pkgs)}{Style.RESET_ALL}")
        print(f"Please run: pip install {' '.join(missing_pkgs)}")
        sys.path.pop(0)
        return

    print(f"{Fore.GREEN}Running Wikontic on {input_path}...{Style.RESET_ALL}")

    from wikontic_baseline import WikonticBaseline
    import argparse

    runner = WikonticBaseline()
    parser = argparse.ArgumentParser()
    runner.add_args(parser)
    args = parser.parse_args([
        "--input", str(input_path),
        "--output", str(output_path),
        "--api-key", api_key,
        "--api-url", api_url,
        "--model", model,
    ])
    result = runner.main(args)
    print(f"{Fore.GREEN}✅ Done. Output saved to {result}{Style.RESET_ALL}")
