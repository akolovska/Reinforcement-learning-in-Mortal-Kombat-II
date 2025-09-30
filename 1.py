import retro
import pygame
import numpy as np
from RetroEnv import RetroEnv

# Initialize environment and pygame
env = RetroEnv('MortalKombatII-Genesis', players=1, use_restricted_actions=retro.Actions.FILTERED)
obs = env.reset()

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

# Get button names
buttons = env.buttons
print("Buttons:", buttons)

# Map keys to button indices
key_map = {
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_a: 'B',  # punch
    pygame.K_s: 'C',  # kick
    pygame.K_d: 'A',  # block or special
}

done = False
while not done:
    action = np.zeros(len(buttons), dtype=np.int8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    for k, btn_name in key_map.items():
        if keys[k]:
            idx = buttons.index(btn_name)
            action[idx] = 1  # press that button

    obs, reward, done, _, info = env.step(action)
    env.render()
    clock.tick(60)  # 60 FPS

env.close()
pygame.quit()
