#!/usr/bin/python
import os
while 1:
    try:
        user_input = input(f"Python:{os.getcwd()} $ ")
        exec(user_input)
    except (NameError, SyntaxError):
        os.system(user_input)
    except Exception as e:
        print(f"Error: {e}")
