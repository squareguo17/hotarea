import argparse

from fastreid.config import get_cfg

def getConfig():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument("--config-file", metavar="FILE", default="../configs/Market1501/sbs_R101-ibn.yml")
    parser.add_argument("--opts", default=["MODEL.WEIGHTS", "./models/market_sbs_R101-ibn.pth"], nargs=argparse.REMAINDER,)

    args = parser.parse_args()

    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg

if __name__ == "__main__":
    args = getConfig()
    print("hs")