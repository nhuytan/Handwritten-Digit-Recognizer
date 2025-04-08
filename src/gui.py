import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BOUNDARYINC = 5

IMAGESAVE = False

# MODEL = load_model('mnist_v02.h5')
MODEL = load_model('mnist_v02_50epoch.h5')

LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

# Initialize pygame
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)  # Corrected the font typo
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Draw a digit")

iswriting = False

number_xcord = []
number_ycord = []

img_cnt = 0
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(number_xcord[-1] + BOUNDARYINC, WINDOWSIZEX)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC, WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            #img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x-5:rect_max_x+5, rect_min_y-5:rect_max_y+5].T.astype(np.float32)
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite(f"img_{img_cnt}.png", img_arr)
                img_cnt += 1

            if PREDICT:
                # Resize the image to 28x28 pixels, it is gray scale image
                image = cv2.resize(img_arr, (20, 20))
                image = np.pad(image, (8, 8), 'constant', constant_values=0)
                # convert to binary image
                image = cv2.resize(image, (28, 28)) / 255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                 # Draw the rectangle around the prediction area
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 1)

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                #textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y
                    # Place the text at the center top of the rectangle
                textRecObj.centerx = (rect_min_x + rect_max_x) // 2
                textRecObj.top = rect_min_y - 20  # Slightly above the rectangle

                DISPLAYSURF.blit(textSurface, textRecObj)
                
        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
