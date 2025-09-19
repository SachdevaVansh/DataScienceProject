import logging
import os 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

##Folder ka path 
logs_path=os.path.join(os.getcwd(),"logs")

## Full path to the log file
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

## make sure log directory is present
os.makedirs(logs_path,exist_ok=True)

logging .basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

