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

##CNN
#from VCNN import Net
from CNN import Net

## PID controler as expert
from basic_agent import BasicAgent
from roaming_agent import RoamingAgent

## local planner: turning options, used to chose waypoint (center of lane when turning)
#from local_planner import _retrieve_options
from agents.tools.misc import distance_vehicle, draw_waypoints

## global variable
out_dir = 'dagger_data'
episode_count = -1
collision_glb = False
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
        self.camera_lidar = None
        self.sensor_collision = None
        # self.collision_history = []
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

        if self.camera_rgb is not None:
            self.camera_rgb.destroy()
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')
        
        #self.camera_rgb = self.world.spawn_actor(
        #    bp,
        #    carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        #    attach_to=self.player)
        
        sensor_location = carla.Location(x=1.6, z=1.7)
        self.camera_rgb = self.world.spawn_actor(bp, carla.Transform(sensor_location), attach_to=self.player)
        self.actor_list.append(self.camera_rgb)
        
        bound_y = 0.5 + self.player.bounding_box.extent.y

        if self.camera_lidar is not None:
            self.camera_lidar.destroy()
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '5000')
        bp.set_attribute('rotation_frequency', self.fps)
        self.camera_lidar = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)),
            attach_to=self.player)

        self.actor_list.append(self.camera_lidar)

        self.rgb_saver = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
            100, 1280, 720, 3, 'CameraRGB')
        self.lidar_saver = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count), 
            100, 8000, 1, 3, 'Lidar')

        print("Initializing collision sensor")
        if self.sensor_collision is not None:
            self.sensor_collision.destroy()
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.sensor_collision = self.world.spawn_actor(
            bp,
            carla.Transform(),
            attach_to=self.player)

        self.sensor_collision.listen(lambda event : self.on_collision(event))

        self.actor_list.append(self.sensor_collision)

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
        make_queue(self.camera_lidar.listen)
        # make_queue(self.sensor_collision.listen)
        
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
            self.lidar_saver.add_image(image.raw_data, "Lidar")
        else: # sensor_name == 'CameraRGB'
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            surface.blit(image_surface, (0, 0))
            self.rgb_saver.add_image(image.raw_data, 'CameraRGB')

    def on_collision(weak_self, event):
        # self = weak_self()
        # if not self:
        #     return
        # actor_type = get_actor_display_name(event.other_actor)
        # self.hud.notification('Collision with %r' % actor_type)
        # impulse = event.normal_impulse
        # intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # self.history.append((event.frame_number, intensity))
        # if len(self.history) > 4000:
        #     self.history.pop(0)
        
        ## terminate if collide
        global collision_glb
        collision_glb = True




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

    ## Set up pointnet - Lidar
    num_classes = 1
    feature_transform = False
    net = Net()

    global episode_count
    episode_count = int(args.iter)

    load_pretrained = True

    if load_pretrained:
        weights_path = ('./result/dagger_%d.pth' % episode_count)
        print('loading pretrained model from.. '+ weights_path)
        net.load_state_dict(torch.load(weights_path))
    net.cuda()

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
        episode_count += 1

        saver_control = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
                               100, 1, 1, 4, 'Control')
                               
        ## save used control signal: throttle, steer, brake, speed
        saver_control_real = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
                               100, 1, 1, 4, 'Control_real')
                               
                               
        clock = pygame.time.Clock()
        world = World(client.get_world(), args)
        controller = KeyboardControl(world, args.autopilot)

        ## PID agent

        world.player.set_location(world.map.get_spawn_points()[0].location)

        ## training road
        world.player.set_transform(carla.Transform(carla.Location(x=305.0, y=129.0, z=2.0), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)))
        
        clock.tick()
        ret = world.tick(timeout=10.0)

        agent = RoamingAgent(world.player)
        print('NNPID: current location, ', world.player.get_location())

        position = []

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
                waypt_buffer = agent._waypoint_buffer

                while not waypt_buffer:
                    pid_control = agent.run_step()

                global collision_glb
                if collision_glb:
                    player_loc = world.player.get_location()
                    #waypt = carla_world.get_map().get_waypoint(player_loc)
                    waypt, _ = waypt_buffer[0]
                    world.player.set_transform(waypt.transform)
                    #world.player.set_location(waypt.transform.location)
                    collision_glb = False
                    print('hit! respawn')
                    pid_control = agent.run_step()

                pid_control.manual_gear_shift = False
                
                ## Neural net control
                cust_ctrl = controller._control
                #cust_ctrl.throttle = 0.5    # 18km/h
                #cust_ctrl.brake = 0

                clock.tick()
                ret = world.tick(timeout=10.0)
                if ret:
                    snapshot = ret[0]
                    img_rgb = ret[1]
                    img_lidar = ret[2]
                    world.parse_image_custom(display, img_rgb, 'CameraRGB')
                    world.parse_image_custom(display, img_lidar, 'Lidar')
                    # if ret[3]:
                    #     world.on_collision(ret[3])
                    '''
                    ## Lidar
                    tst_inputs = np.frombuffer(img_lidar.raw_data, dtype=np.dtype('f4'))
                    tst_inputs = np.reshape(tst_inputs, (int(tst_inputs.shape[0] / 3), 3))
                    tst_inputs = np.asarray(tst_inputs, dtype=np.float32)

                    # test if there's large points
                    sum_ = np.sum(np.absolute(tst_inputs), axis=1)
                    mask = np.logical_and(sum_ < 50*3, sum_ > 0.0001)
                    pts_filter = tst_inputs[mask]
                    
                    if(pts_filter.shape != tst_inputs.shape):
                        print('pts filter : pts =', pts_filter.shape, tst_inputs.shape)
                    
                    tst_inputs = torch.from_numpy(tst_inputs)
                    #print(tst_inputs.shape)
                    
                    tst_inputs = tst_inputs[0:1900,:]   #
                    
                    tst_inputs = tst_inputs.unsqueeze(0)
                    tst_inputs = tst_inputs.transpose(2, 1)
                    tst_inputs = tst_inputs.cuda()
                    
                    points = tst_inputs
                    '''
                    #print(tst_inputs)
                    
                    ## images
                    raw_img = np.frombuffer(img_rgb.raw_data, dtype=np.uint8)
                    raw_img = raw_img.reshape(720, 1280, -1)
                    raw_img = raw_img[:, :, :3]
                    raw_img = cv2.resize(raw_img, dsize=(180, 180))
                    #print(raw_img)
                    tst_inputs = raw_img /255
                    tst_inputs = np.transpose(tst_inputs, (2, 0, 1))
                    tst_inputs = np.asarray(tst_inputs, dtype=np.float32)
                    tst_inputs = torch.from_numpy(tst_inputs)
                    tst_inputs = tst_inputs.unsqueeze(0)
                    tst_inputs = tst_inputs.cuda()
                    
                    image = tst_inputs
                    
                    net = net.eval()
                    
                    
                    v = world.player.get_velocity()
                    speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                    speed = torch.tensor(speed).cuda()
                    speed = speed.unsqueeze(0)
                    speed = torch.unsqueeze(speed, -1)
                    print(speed)
                    
                    #print(image)
                    with torch.no_grad():
                        
                        ah = net(image, speed)
                        ah = torch.squeeze(ah)
                        #print(a_list[0].detach().squeeze().tolist())
                    #print(ah)
                    outputs = ah
                    
                    print(outputs)
                    #cust_ctrl.throttle = 0.5 #outputs[0].item()
                    cust_ctrl.steer = outputs[1].item()
                    #cust_ctrl.brake = 0.0 #outputs[2].item()
                    
                    
                    # after iteration 2
                    #if episode_count >= 1:
                    cust_ctrl.brake = outputs[2].item()
                    
                    if cust_ctrl.brake < 0.5:
                        cust_ctrl.brake = 0.0
                    
                    cust_ctrl.throttle = outputs[0].item()
                        
                    
                    
                    #'''
                    world.player.apply_control(cust_ctrl)

                    player_loc = world.player.get_location()
                    position.append([player_loc.x, player_loc.y, player_loc.z])
                    
                    #'''
                    ## check the center of the lane
                    #waypt = carla_world.get_map().get_waypoint(player_loc)
                    
                    waypt, road_option = waypt_buffer[0]
                    lane_center = waypt.transform.location
                    
                    #print(_current_lane_info.lane_type)
                    #print('waypt ', lane_center)
                    #print('player ', player_loc)
                    
                    dist = math.sqrt((lane_center.x - player_loc.x)**2 + (lane_center.y - player_loc.y)**2)  
                    
                    ## dif in direction
                    next_dir = waypt.transform.rotation.yaw % 360.0
                    player_dir = world.player.get_transform().rotation.yaw % 360.0
                    
                    diff_angle = (next_dir - player_dir) % 180.0
                    
                    ## too far from road, use PID control
                    if (diff_angle > 85 and diff_angle < 105) or dist >= 15: 
                        #print('pid_control')
                        #world.player.apply_control(pid_control)
                        #draw_waypoints(carla_world, [waypt], player_loc.z + 2.0)
                        player_loc = world.player.get_location()
                        #waypt = carla_world.get_map().get_waypoint(player_loc)
                        waypt, _ = waypt_buffer[0]
                        world.player.set_transform(waypt.transform)
                        #world.player.set_location(waypt.transform.location)
                        collision_glb = False
                        print('too far! respawn')
                        pid_control = agent.run_step()

                    #''' 
                    # draw_waypoints(carla_world, [waypt], player_loc.z + 2.0)
                    #world.player.apply_control(pid_control)
                else:
                    print("Nothing is returned from world.tick :(")

                ## Record expert (PID) control
                c = pid_control
                throttle = c.throttle  # 0.0, 1.0
                steer = c.steer #-1.0, 1.0
                brake = c.brake # 0.0, 1.0
                
                #print(throttle, steer, brake)
                v = world.player.get_velocity()
                speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                
                #print('Speed:   % 15.0f km/h' % (speed))
                
                control = np.array([throttle, steer, brake, speed])
                saver_control.add_image(control, "Control")
                
                control_used = np.array([throttle, float(cust_ctrl.steer), brake, speed])
                saver_control_real.add_image(control_used, "Control")

                if len(position) == 3050:
                    break

                pygame.display.flip()

    finally:

        print("Destroying actors...")
        if world is not None:
            world.destroy()

        ## save position
        position = np.asarray(position)
        save_name = './dagger_data/ep_%d/path.npy' % (episode_count)
        np.save(save_name, position)
        print('position saved in ',save_name)

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
        '--iter',
        default=-1,
        help='Dagger iteration number (default: -1')
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
