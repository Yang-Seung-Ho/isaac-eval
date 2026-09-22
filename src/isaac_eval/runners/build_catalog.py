from isaac_eval.config import load_env, load_settings, get
from isaac_eval.catalog import build_catalog_entry, save_catalog


def main():
    load_env()
    cfg = load_settings()
    entry = build_catalog_entry(cfg)
    out_path = f"{get(cfg, 'paths.catalog_dir')}/jeju_phytoplankton.json"
    save_catalog(entry, out_path)
    print(f"catalog written to {out_path}")


if __name__ == "__main__":
    main()
