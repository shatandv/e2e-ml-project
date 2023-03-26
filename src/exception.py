""" Andrey Shataev, 2023
Custom exception handling module.
"""

import sys


# A bit weird we're making a function parameter have type "sys". Is this really right?
def error_message_detail(error, error_detail: sys) -> str:
    """Assemble a custom error message for exceptions.

    Args:
        error (_type_): Exception that occured
        error_detail (sys): ???

    Returns:
        str: Final error message for an exception
    """
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in Python script: [{0}], line number: [{1}], error message: [{2}]".format(
        filename, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)

        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        1 / 0
    except Exception as e:
        raise CustomException(e, sys)
