#!/usr/bin/env python
from __future__ import print_function
# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
'''
try:
    sys.path.append("/media/zidong/f97c40bb-b3c0-41d4-a17f-2df375ae1d1e/zidong/future2/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg")
except IndexError:
    pass
## buffer
sys.path.append('/home/zidong/Desktop/CARLA_0.9.5/data_collector')
## PID
sys.path.append('/media/zidong/f97c40bb-b3c0-41d4-a17f-2df375ae1d1e/zidong/future2/CARLA_0.9.6/PythonAPI/carla')
'''

try:
    sys.path.append("/home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-5-g255683a7-dirty/LinuxNoEditor/PythonAPI/carla/dist/carla-0.9.6-py3.6-linux-x86_64.egg")
except IndexError:
    pass
## buffer
sys.path.append('/home/siyun/wzd/data_collector')
## PID
sys.path.append('/home/siyun/CARLA/repo/carla/Dist/CARLA_Shipping_0.9.6-5-g255683a7-dirty/LinuxNoEditor/PythonAPI/carla')

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
try:
    import pygame
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


from buffered_saver import BufferedImageSaver

## pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import cv2  # for resizing image


## PID controler as expert
from basic_agent import BasicAgent
from roaming_agent import RoamingAgent


## global variable
out_dir = '_temp_PID'
episode_count = 0

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, args):
        self.world = carla_world
        self.actor_role_name = 'hero' #args.rolename
        self.map = self.world.get_map()
        self.player = None
        self.camera_rgb = None
        #self.camera_rgb_1 = None
        #self.camera_rgb_2 = None
        #self.camera_rgb_3 = None
        self.camera_lidar = None
        self.actor_list = []
        self._actor_filter = args.filter
        self.fps = args.fps
        self.restart()
        self.frame = None
        self.delta_seconds = 1.0 / int(args.fps)
        self._queues = []
        self._settings = None
        self.world.wait_for_tick()

    def restart(self):
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            print("spawn point is : " + spawn_point)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            print('Spawning player at location = ', self.player.get_location())

        self.actor_list.append(self.player)

        print("Initializing custom rgb and lidar sensors")
        bound_y = 0.5 + self.player.bounding_box.extent.y
        sensor_location = carla.Location(x=1.6, z=1.7)

        if self.camera_rgb is not None:
            self.camera_rgb.destroy()
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')

        # self.camera_rgb = self.world.spawn_actor(
        #     bp,
        #     carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        #     attach_to=self.player)
        self.camera_rgb = self.world.spawn_actor(bp, carla.Transform(sensor_location), attach_to=self.player)

        self.actor_list.append(self.camera_rgb)
        
        '''
        if self.camera_rgb_1 is not None:
            self.camera_rgb_1.destroy()
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')

        self.camera_rgb_1 = self.world.spawn_actor(bp, carla.Transform(sensor_location, carla.Rotation(yaw=90)), attach_to=self.player)

        self.actor_list.append(self.camera_rgb_1)

        if self.camera_rgb_2 is not None:
            self.camera_rgb_2.destroy()
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')

        self.camera_rgb_2 = self.world.spawn_actor(bp, carla.Transform(sensor_location, carla.Rotation(yaw=180)), attach_to=self.player)

        self.actor_list.append(self.camera_rgb_2)

        if self.camera_rgb_3 is not None:
            self.camera_rgb_3.destroy()
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')

        self.camera_rgb_3 = self.world.spawn_actor(bp, carla.Transform(sensor_location, carla.Rotation(yaw=270)), attach_to=self.player)

        self.actor_list.append(self.camera_rgb_3)
        '''
        if self.camera_lidar is not None:
            self.camera_lidar.destroy()
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '5000')
        bp.set_attribute('rotation_frequency', self.fps)
        
        
        self.camera_lidar = self.world.spawn_actor(bp, carla.Transform(sensor_location), attach_to=self.player)

        self.actor_list.append(self.camera_lidar)

        self.rgb_saver = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
            100, 1280, 720, 3, 'CameraRGB')
        '''
        self.rgb_saver_1 = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
            100, 1280, 720, 3, 'CameraRGB_1')
        self.rgb_saver_2 = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
            100, 1280, 720, 3, 'CameraRGB_2')
        self.rgb_saver_3 = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
            100, 1280, 720, 3, 'CameraRGB_3')
        '''
        self.lidar_saver = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count), 
            100, 8000, 1, 3, 'Lidar')

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)        
        make_queue(self.camera_rgb.listen)
        #make_queue(self.camera_rgb_1.listen)
        #make_queue(self.camera_rgb_2.listen)
        #make_queue(self.camera_rgb_3.listen)
        make_queue(self.camera_lidar.listen)
        
        return self

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def destroy(self):
        for actor in self.actor_list:
            if actor is not None:
                actor.destroy()

    ## Customized parse image
    def parse_image_custom(self, surface, image, sensor_name):
        if sensor_name == 'Lidar':
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(1280, 720) / 100.0
            lidar_data += (0.5 * 1280, 0.5 * 720)
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (1280, 720, 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            # self.surface = pygame.surfarray.make_surface(lidar_img)
            self.lidar_saver.add_image(image.raw_data, sensor_name)
        elif sensor_name == 'CameraRGB':
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            #if sensor_name == 'CameraRGB':
            surface.blit(image_surface, (0, 0))
            self.rgb_saver.add_image(image.raw_data, sensor_name)
            

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 3.333 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        carla_world = client.get_world()
        # settings = carla_world.get_settings()
        # carla_world.apply_settings(carla.WorldSettings(
        #     no_rendering_mode=False,
        #     synchronous_mode=True,
        #     fixed_delta_seconds=1.0 / 20))

        ## save control signal: throttle, steer, brake, speed
        saver_control = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
                               100, 1, 1, 4, 'Control')
        ## save used control signal: throttle, steer, brake, speed
        saver_control_real = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
                               100, 1, 1, 4, 'Control_real')
        
        clock = pygame.time.Clock()

        world = World(client.get_world(), args)
        controller = KeyboardControl(world, args.autopilot)

        ## PID agent
        print('current pos =', world.map.get_spawn_points()[0])
        print('dest pos =', world.map.get_spawn_points()[10])
        
        #world.player.set_transform(world.map.get_spawn_points()[0])
        world.player.set_location(world.map.get_spawn_points()[0].location)
        print('set: ', world.map.get_spawn_points()[0].location)
        print('NNPID: current location, ', world.player.get_location())
        
        ## training road
        world.player.set_transform(carla.Transform(carla.Location(x=305.0, y=129.0, z=2.0), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)))
        
        ## testing road
        #world.player.set_transform(carla.Transform(carla.Location(x=310.0, y=326.0, z=2.0), carla.Rotation(pitch=0.138523, yaw=180.0, roll=0.000112504)))

        agent = RoamingAgent(world.player)

        with world:

            while True:
                # clock.tick_busy_loop(60)
                # if controller.parse_events(client, world, clock):
                #     carla_world.tick()
                #     ts = carla_world.wait_for_tick()
                #     return
                if should_quit():
                    return

                ## set custom control
                
                pid_control = agent.run_step()
                #pid_control.manual_gear_shift = False
                #pid_control.throttle = 0.5
                
                world.player.apply_control(pid_control)
                
                ## control
                c = pid_control
                throttle = c.throttle  # 0.0, 1.0
                steer = c.steer #-1.0, 1.0
                brake = c.brake # 0.0, 1.0
                
                #print(throttle, steer, brake)
                v = world.player.get_velocity()
                speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                #print('Speed:   % 15.0f km/h' % (speed))
                
               
                control = np.array([throttle, steer, brake, speed])
                control_used = np.array([throttle, steer, brake, speed])
                
                # if not tst_mode:
                saver_control.add_image(control, "Control")
                saver_control_real.add_image(control_used, "Control")
                
                clock.tick()
                ret = world.tick(timeout=2.0)
                if ret:
                    #snapshot, img_rgb, img_rgb_1, img_rgb_2, img_rgb_3, img_lidar = ret
                    snapshot, img_rgb, img_lidar = ret
                    world.parse_image_custom(display, img_rgb, 'CameraRGB')
                    #world.parse_image_custom(display, img_rgb_1, 'CameraRGB_1')
                    #world.parse_image_custom(display, img_rgb_2, 'CameraRGB_2')
                    #world.parse_image_custom(display, img_rgb_3, 'CameraRGB_3')
                    world.parse_image_custom(display, img_lidar, 'Lidar')
                else:
                    print("Nothing is returned from world.tick :(")

                pygame.display.flip()

    finally:

        print("Destroying actors...")
        if world is not None:
            world.destroy()

        pygame.quit()
        print("Done")


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--fps',
        default='20',
        help='FPS')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
