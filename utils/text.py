# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:02
# @Author tyqqj
# @File text.py
# @
# @Aim 


def text_in_box(text, length=65, center=True):
    # Split the text into lines that are at most `length` characters long
    lines = [text[i:i + length] for i in range(0, len(text), length)]

    # Create the box border, with a width of `length` characters
    up_border = '┏' + '━' * (length + 2) + '┓'
    down_border = '┗' + '━' * (length + 2) + '┛'
    # Create the box contents
    contents = '\n'.join(['┃ ' + (line.center(length) if center else line.ljust(length)) + ' ┃' for line in lines])

    # Combine the border and contents to create the final box
    box = '\n'.join([up_border, contents, down_border])

    return box


# 分割线
def split_line(length=65):
    return '\n' + '━' * length + '\n'

