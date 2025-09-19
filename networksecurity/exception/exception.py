import sys
print("Hello",sys.path)
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
from networksecurity.logging import logger 

class NetworkSecurityException(Exception):
    def __init__(self, error_message:Exception, error_detail:sys):
        super().__init__(error_message)
        self.error_message = NetworkSecurityException.get_detailed_error_message(error_message=error_message, error_detail=error_detail)

    @staticmethod
    def get_detailed_error_message(error_message:Exception, error_detail:sys)->str:
        _,_,exec_tb = error_detail.exc_info()
        line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        detailed_error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"
        return detailed_error_message

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return NetworkSecurityException.__name__.str() + ": " + self.error_message

