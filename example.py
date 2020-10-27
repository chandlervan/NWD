from NWD import NWD_base

if __name__ == '__main__':
    f = open('./dataset/005.斗破苍穹.txt', 'r')  # 读取文章
    s = f.read()  # 读取为一个字符串
    nwd = NWD_base()
    result = nwd.run(s)
    print(result.shape)