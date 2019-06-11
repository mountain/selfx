# -*- coding: utf-8 -*-

from Box2D.Box2D import (b2PolygonShape as polygonShape, b2World as world)

world = world(gravity=(0, -10), doSleep=True)

ground_body = world.CreateStaticBody(
    position=(0, 0),
    shapes=polygonShape(box=(50, 1)),
)

# Create a couple dynamic bodies
bodyc = world.CreateDynamicBody(position=(20, 45))
circle = bodyc.CreateCircleFixture(radius=0.5, density=1, friction=0.3)

bodyb = world.CreateDynamicBody(position=(30, 45), angle=15)
box = bodyb.CreatePolygonFixture(box=(2, 1), density=1, friction=0.3)

world.CreateWeldJoint(bodyA=bodyc, bodyB=bodyb, anchor=bodyb.worldCenter)
