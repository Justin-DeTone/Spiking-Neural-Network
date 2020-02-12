hextorgb = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15
}

rgbtohex = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'
}

def rgbToHex(rgb_list):
    # converts RGB color list to hexadecimal and returns a string
    string1 = "#"
    for item in rgb_list:
        string1 = string1 + rgbtohex[item // 16]
        string1 = string1 + rgbtohex[item % 16]
    return string1

def hexToRGB(hex):
    # input is hexadecimal list with three items for each color, each item is a string
    if hex[0] != '#':
        print("Not a hexadecimal string")
        return
    tmp_hex = hex[1:]
    tmp = []
    for _ in range(3):
        tmp.append(tmp_hex[:2])
        tmp_hex = tmp_hex[2:]
    result = []
    for value in tmp:
        tmp_val = hextorgb[value[0].lower()] * 16 + hextorgb[value[1].lower()]
        result.append(tmp_val)
    return result
