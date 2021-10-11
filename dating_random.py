import random

# 打开dating.txt文件准备数据写入
fo = open("./ins/dating.txt", "w")
# 循环写入1000个数据
fo.write('Mileage,Liters,Pastime,Target\n')
for turn1 in range(1000):
    data = [random.randint(5000, 80000), random.uniform(0, 20), random.uniform(1, 2), random.randint(1, 3)]
    data_new = ','.join(str(turn2) for turn2 in data)
    fo.write(data_new)
    fo.write('\n')
# 关闭dating.txt文件
fo.close()
