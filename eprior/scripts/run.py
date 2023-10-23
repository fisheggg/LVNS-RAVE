import parser
import gin
import sys
import logging
import argparse

sys.path.append("../eprior")
from core import EPrior


def main(args):
    gin.parse_config_file(args.config_path)
    logging.basicConfig(level=logging.INFO)
    eprior = EPrior()
    eprior.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, description="Path to the eprior config file, in gin format")

    args = parser.parse_args()
    main(args)
