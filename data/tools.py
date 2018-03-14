__author__ = 'justinarmstrong'

import os
import pygame as pg
from . import model
from . import realtime
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()

epsilon = model.INITIAL_EPSILON
time_elapsed, loss = 0, 0
sess = tf.InteractiveSession()
model_nn = model.Model()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
fig, axes = plt.subplots(figsize = (3, 3))
display = realtime.RealtimePlot(axes)

try:
    saver.restore(sess, os.getcwd() + "/model.ckpt")
    print("Done load checkpoint")
except:
    print ("start from fresh variables")

keybinding = {
    'action':pg.K_s,
    'jump':pg.K_a,
    'left':pg.K_LEFT,
    'right':pg.K_RIGHT,
    'down':pg.K_DOWN
}

keypress = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def edit_keypress(matrix):
    keypress[keybinding['jump']] = matrix[0]
    keypress[keybinding['left']] = matrix[1]
    keypress[keybinding['right']] = matrix[2]

class Control(object):
    """Control class for entire project. Contains the game loop, and contains
    the event_loop which passes events to States as needed. Logic for flipping
    states is also found here."""
    def __init__(self, caption):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.caption = caption
        self.fps = 60
        self.show_fps = False
        self.current_time = 0.0
        self.keys = keypress
        self.state_dict = {}
        self.state_name = None
        self.state = None

    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]

    def update(self):
        global time_elapsed
        global epsilon
        global loss
        self.current_time = pg.time.get_ticks()
        if self.state.quit:
            self.done = True
        elif self.state.done:
            self.flip_state()
        return_tuple = self.state.update(self.screen, self.keys, self.current_time)
        if return_tuple is not None:
            if not model_nn.middle_game:
                for i in range(model_nn.initial_stack_images.shape[2]):
                    model_nn.initial_stack_images[:, :, i] = return_tuple[0]
                model_nn.middle_game = True
            action = np.zeros([model.ACTIONS], dtype = np.int)    
            if time_elapsed % model.FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print('step: ', time_elapsed, ', do random actions')
                    action = np.random.randint(2, size = model.ACTIONS)
                    action[1] = 0
                else:
                    actions = sess.run([model_nn.logits_space, model_nn.logits_left, model_nn.logits_right], 
                                              feed_dict = {model_nn.X: [model_nn.initial_stack_images]})
                    for i in range(len(actions)):
                        if np.argmax(actions[i]) == 1:
                            action[i] = 1
            else:
                action = np.zeros([model.ACTIONS], dtype = np.int)
            print(action)
            edit_keypress(action)
            if epsilon > model.FINAL_EPSILON and time_elapsed > model.OBSERVE:
                epsilon -= (model.INITIAL_EPSILON - model.FINAL_EPSILON) / model.EXPLORE
            stack_images = np.append(return_tuple[0].reshape([80, 80, 1]), model_nn.initial_stack_images[:, :, :3], axis = 2)
            model_nn.memory.append((model_nn.initial_stack_images, action, return_tuple[1], stack_images, return_tuple[2]))
            if len(model_nn.memory) > model.REPLAY_MEMORY_SIZE:
                model_nn.memory.popleft()

            if time_elapsed > model.OBSERVE:
                minibatch = random.sample(model_nn.memory, model.BATCH)
                initial_image_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]
                reward_batch = [d[2] for d in minibatch]
                image_batch = [d[3] for d in minibatch]
                y_batch = []
                action_space = np.zeros((model.BATCH, 2))
                action_left = np.zeros((model.BATCH, 2))
                action_right = np.zeros((model.BATCH, 2))
                actions = sess.run([model_nn.logits_space, model_nn.logits_left, model_nn.logits_right], 
                                              feed_dict = {model_nn.X: image_batch})
                for i in range(len(minibatch)):
                    if minibatch[i][4]:
                        y_batch.append(reward_batch[i])
                    else:
                        y_batch.append(reward_batch[i] + model.GAMMA * np.argmax(actions[0][i]) + model.GAMMA * np.argmax(actions[1][i]) + model.GAMMA * np.argmax(actions[2][i]))
                    if action_batch[i][0] == 0:
                        action_space[i][0] = 1
                    else:
                        action_space[i][1] = 1
                    if action_batch[i][1] == 0:
                        action_left[i][0] = 1
                    else:
                        action_left[i][1] = 1
                    if action_batch[i][2] == 0:
                        action_right[i][0] = 1
                    else:
                        action_right[i][1] = 1
                loss, _ = sess.run([model_nn.cost, model_nn.optimizer], feed_dict = {model_nn.Y: y_batch, 
                                                                                     model_nn.action_space: action_space,
                                                                                     model_nn.action_left: action_left,
                                                                                     model_nn.action_right: action_right,
                                                                                     model_nn.X: initial_image_batch})
                print('step: ', time_elapsed, ', loss: ', loss)

            time_elapsed += 1
            display.add(time_elapsed, loss)
            plt.pause(0.001)
            model_nn.initial_stack_images = stack_images
            if (time_elapsed + 1) % 1000 == 0:
                print('step: ', time_elapsed)
            if time_elapsed % 10000 == 0:
                print('checkpoint saved')
                saver.save(sess, os.getcwd() + "/model.ckpt")

    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)
        self.state.previous = previous


    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                self.toggle_show_fps(event.key)
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
            self.state.get_event(event)


    def toggle_show_fps(self, key):
        if key == pg.K_F5:
            self.show_fps = not self.show_fps
            if not self.show_fps:
                pg.display.set_caption(self.caption)


    def main(self):
        """Main loop for entire program"""
        while not self.done:
            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)
            if self.show_fps:
                fps = self.clock.get_fps()
                with_fps = "{} - {:.2f} FPS".format(self.caption, fps)
                pg.display.set_caption(with_fps)


class _State(object):
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.quit = False
        self.next = None
        self.previous = None
        self.persist = {}

    def get_event(self, event):
        pass

    def startup(self, current_time, persistant):
        self.persist = persistant
        self.start_time = current_time

    def cleanup(self):
        self.done = False
        return self.persist

    def update(self, surface, keys, current_time):
        pass



def load_all_gfx(directory, colorkey=(255,0,255), accept=('.png', 'jpg', 'bmp')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name]=img
    return graphics


def load_all_music(directory, accept=('.wav', '.mp3', '.ogg', '.mdi')):
    songs = {}
    for song in os.listdir(directory):
        name,ext = os.path.splitext(song)
        if ext.lower() in accept:
            songs[name] = os.path.join(directory, song)
    return songs


def load_all_fonts(directory, accept=('.ttf')):
    return load_all_music(directory, accept)


def load_all_sfx(directory, accept=('.wav','.mpe','.ogg','.mdi')):
    effects = {}
    for fx in os.listdir(directory):
        name, ext = os.path.splitext(fx)
        if ext.lower() in accept:
            effects[name] = pg.mixer.Sound(os.path.join(directory, fx))
    return effects











