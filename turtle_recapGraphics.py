from turtle import*
import turtle

for i in range(5):
    print('forloop')
    
print('-------------')

for n in range(5):
    print(n)

shape('triangle')

for j in range(4):
    forward(100)
    right(90)

def rectangle_func():
    for k in range(4):
        forward(150)
        right(140)

rectangle_func()


def draw_squares():
    for _ in range(60):
        for _ in range(4):
            turtle.forward(100)
            turtle.right(90)
        turtle.right(5)

# Set up the screen
turtle.setup(800, 600)

# Set the turtle speed
turtle.speed(0)

# Draw the squares
draw_squares()

# Finish
turtle.done()
        

    

    
