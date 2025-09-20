import sys
import os
import traceback
from networksecurity.logging import logger 

class NetworkSecurityException(Exception):
    def __init__(self, error_message:Exception):
        self.error_message = f"{str(error_message)}\n Traceback: {traceback.format_exc()}"
        super().__init__(error_message)

    def __str__(self):
        return self.error_message

