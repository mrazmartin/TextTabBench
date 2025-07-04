
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
MAGENTA = "\033[1;35m"
CYAN = "\033[1;36m"
ORANGE = "\033[1;91m"
RESET = "\033[0m"

color_dict = {"red": RED, "green": GREEN,
              "yellow": YELLOW, "blue": BLUE,
              "magenta": MAGENTA, "cyan": CYAN,
              "orange": ORANGE}

def info_msg(message, color=CYAN):
    color = set_color(color)
    text = f"{color}Info:{RESET} {message}"
    print(text)

def warn_msg(message, color=YELLOW):
    color = set_color(color)
    text = f"{color}Warning:{RESET} {message}"
    print(text)

def error_msg(message, color=RED):
    color = set_color(color)
    text = f"\n{color}Error:{RESET} {message}"
    print(text)

def success_msg(message, color=GREEN):
    color = set_color(color)
    text = f"{color}Success:{RESET} {message}"
    print(text)

def set_color(color):
    if color not in color_dict:
        if color in color_dict.values():
            return color
        else:
            raise ValueError(f"Color {color} is not supported.")
    else:
        color = color_dict[color]

    return color

def color_text(text, color='magenta'):
    color = set_color(color)
    return f"{color}{text}{RESET}"