import contextlib
with contextlib.redirect_stdout(None):
    import pygame
from pickle import load
import random
import math


# drawMode() used for testing a network with user-created data. Requires Pygame.
def drawMode(path):
    # Create Pygame context
    pygame.init()
    screen = pygame.display.set_mode([500, 500])
    pygame.display.set_caption(f'drawMode("{path}")')
    font = pygame.font.Font('freesansbold.ttf', 32)

    # Drawing grid
    grid = [[float(0) for j in range(28)] for i in range(28)]

    # Import a network
    network = load(open(path, 'rb'))

    running = True
    while running:

        # User events handling, wiping the grid and exiting the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    grid = [[float(0) for j in range(28)] for i in range(28)]

        # Get mouse input
        gridx = math.floor(pygame.mouse.get_pos()[0] / (500 / 28))
        gridy = math.floor(pygame.mouse.get_pos()[1] / (500 / 28))

        # Simple drawing function, approximates the MNIST brush
        if pygame.mouse.get_pressed()[0]:
            grid[gridy][gridx] = 1.0

            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i + j != -1 and i + j != 1:
                        continue

                    if grid[gridy + i][gridx + j] == 0:
                        grid[gridy + i][gridx + j] = random.uniform(0.4, 0.8)

        # Drawing the input
        screen.fill((0, 0, 0))
        for i in range(28):
            for j in range(28):
                color = (int(grid[i][j] * 255), int(grid[i][j] * 255), int(grid[i][j] * 255))
                pygame.draw.rect(screen, color,
                                 pygame.Rect(j * 500 / 28, i * 500 / 28, 500 / 28 + 1, 500 / 28 + 1))

        # Get network guess
        network.inputData([grid[j][i] for j in range(28) for i in range(28)])
        network.propagate()
        guess = str([index for index, item in enumerate(network.output()) if item == max(network.output())][0])

        # Display guess
        text = font.render(guess, True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (20, 20)
        screen.blit(text, textRect)

        pygame.display.flip()  # Render

    pygame.quit()
