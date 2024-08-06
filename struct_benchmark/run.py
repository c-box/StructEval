import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'
os.environ["MKL_THREADING_LAYER"] = '1'
os.environ["DATASET_SOURCE"] = "ModelScope"
os.environ["MASTER_PORT"] = "12345"

from opencompass.cli.main import main


if __name__ == '__main__':
    main()