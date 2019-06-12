#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from gym_selfx.selfx.world import world
from gym_selfx.render.draw import OpencvDrawFuncs

TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS

drawer = OpencvDrawFuncs(w=640, h=480, ppm=20)
drawer.install()

while True:
    key = 0xFF & cv2.waitKey(int(TIME_STEP * 1000))  # milliseconds
    if key == 27:
        break
    drawer.clear_screen()

    drawer.draw_world(world)

    # Make Box2D simulate the physics of our world for one step.
    world.Step(TIME_STEP, 10, 10)

    # Flip the screen and try to keep at the target FPS
    cv2.imshow("world", drawer.screen)
