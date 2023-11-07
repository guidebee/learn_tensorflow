import binascii

import numpy as np


def convert_hz24_to_numpy(text_str):
    keys = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]

    font_size = 24

    rect_list = [] * font_size

    for i in range(font_size):
        rect_list.append([] * font_size)

    len_text = len(text_str)
    for text in text_str:

        # 获取中文的gb2312编码，一个汉字是由2个字节编码组成
        gb2312 = text.encode('gb2312')
        # 将二进制编码数据转化为十六进制数据
        hex_str = binascii.b2a_hex(gb2312)
        # 将数据按unicode转化为字符串
        result = str(hex_str, encoding='utf-8')

        # 前两位对应汉字的第一个字节：区码，每一区记录94个字符
        area = eval('0x' + result[:2]) - 0xA0
        # 后两位对应汉字的第二个字节：位码，是汉字在其区的位置
        index = eval('0x' + result[2:]) - 0xA0
        # 汉字在HZK16中的绝对偏移位置，最后乘32是因为字库中的每个汉字字模都需要32字节
        offset = (94 * (area - 1) + (index - 1)) * font_size * 3

        font_rect = None

        # 读取HZK16汉字库文件
        with open("HZK24", "rb") as f:
            # 找到目标汉字的偏移位置
            f.seek(offset)
            # 从该字模数据中读取32字节数据
            font_rect = f.read(font_size * 3)

        # font_rect的长度是32，此处相当于for k in range(16)
        for k in range(len(font_rect) // 3):
            # 每行数据
            row_list = rect_list[k]
            for j in range(3):
                for i in range(8):
                    asc = font_rect[k * 3 + j]
                    # 此处&为Python中的按位与运算符
                    flag = asc & keys[i]
                    # 数据规则获取字模中数据添加到16行每行中16个位置处每个位置
                    row_list.append(flag)

    # 根据获取到的16*16点阵信息，打印到控制台
    y = 0

    numpy_text = np.zeros((24, 24 * len_text))
    for row in rect_list:
        x = 0
        for i in row:
            if i:
                # 前景字符（即用来表示汉字笔画的输出字符）
                numpy_text[y, x] = 1
            else:
                pass

            x += 1
        y += 1

    return numpy_text
