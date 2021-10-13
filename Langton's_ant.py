import numpy
import random
import turtle


def set_screen(i, j):
    arr = numpy.empty((i, j), dtype=int)
    for a in range(i):
        for b in range(j):
            arr[a][b] = random.randint(0, 1)
    print(arr)
    return arr


def fill_squ(ward):
    if ward == 'd':
        for i in range(4):
            turtle.forward(10)
            turtle.left(90)
    if ward == 'w':
        for i in range(4):
            turtle.forward(10)
            turtle.right(90)
    if ward == 'a':
        turtle.right(180)
        for i in range(4):
            turtle.forward(10)
            turtle.left(90)
        turtle.left(180)
    if ward == 's':
        turtle.left(90)
        for i in range(4):
            turtle.forward(10)
            turtle.left(90)
        turtle.right(90)


def draw(i, j):
    turtle.setup(0.5, 0.5)
    turtle.speed(0)
    i = i//2
    j = j//2
    sign = ['d', 'w', 'a', 's']
    a = 0
    count = 0
    ward = sign[a]
    while True:
        if arr[i][j]:
            arr[i][j] = 0
            turtle.color('white', 'black')
            turtle.begin_fill()
            fill_squ(ward)
            turtle.end_fill()
            turtle.left(90)
            turtle.forward(10)
            if a == 3:
                a = 0
            else:
                a += 1
            ward = sign[a]
            if ward == 'd':
                i += 1
            if ward == 'w':
                j -= 1
            if ward == 'a':
                i -= 1
            if ward == 's':
                j += 1
            count += 1
            print(count)
        else:
            arr[i][j] = 1
            turtle.color('white', 'white')
            turtle.begin_fill()
            fill_squ(ward)
            turtle.end_fill()
            turtle.right(90)
            turtle.forward(10)
            if a == 0:
                a = 3
            else:
                a -= 1
            ward = sign[a]
            if ward == 'd':
                i += 1
            if ward == 'w':
                j -= 1
            if ward == 'a':
                i -= 1
            if ward == 's':
                j += 1
            count += 1
            print(count)
    else:
        turtle.mainloop()


if __name__ == '__main__':
    i = int(input('输入屏幕的长\n'))
    j = int(input('输入屏幕的宽\n'))
    #arr = set_screen(i,j)
    arr = numpy.ones((i,j))
    draw(i, j)
