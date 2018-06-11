#!/usr/bin/env python
# coding: utf-8

"""
计算两个字符串的最小编辑距离
"""


def substitute_cost(char1, char2):
    if char1 == char2:
        return 0
    else:
        return 2


def insert_cost():
    return 1


def delete_cost():
    return 1


def min(a, b, c):
    if a < b:
        less = a
    else:
        less = b
    if c < less:
        return c
    else:
        return less


def min_edit_distance(string1, string2):
    string1_length = len(string1)
    string2_length = len(string2)

    # 创建编辑矩阵, row_count = string1_count, col_count = string2_count
    matrix = [[0] * (string2_length + 1) for row in range(string1_length + 1)]
    # 初始化
    index = 0
    while index <= string1_length:
        matrix[index][0] = index
        index = index + 1
    index = 0
    while index <= string2_length:
        matrix[0][index] = index
        index = index + 1

    # 开始计算
    temp_insert_cost = 0
    temp_substitute_cost = 0
    temp_delete_cost = 0
    string1_index = 1
    while string1_index <= string1_length:
        string2_index = 1
        while string2_index <= string2_length:
            # 计算分步距离
            temp_insert_cost = matrix[string1_index - 1][string2_index] + insert_cost()
            temp_substitute_cost = matrix[string1_index - 1][string2_index - 1] + substitute_cost(
                string1[string1_index - 1], string2[string2_index - 1])
            temp_delete_cost = matrix[string1_index][string2_index - 1] + delete_cost()
            # 比较并赋值
            matrix[string1_index][string2_index] = min(temp_insert_cost, temp_substitute_cost, temp_delete_cost)

            string2_index = string2_index + 1
        string1_index = string1_index + 1

    return matrix[string1_length][string2_length]

def relative_min_edit_distance(string1, string2, n):
    return min_edit_distance(string1, string2) / float(n)

if __name__ == "__main__":
    str1 = "akldfa;dfa"
    str2 = "akjdfkad"
    print min_edit_distance(str1, str2), relative_min_edit_distance(str1, str2, 7)