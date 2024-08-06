import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from opencompass.cli.main import main


if __name__ == '__main__':
    main()