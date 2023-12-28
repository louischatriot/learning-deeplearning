# A lot of code taken from https://github.com/Wazzaps/fingerpaint/blob/master/fingerpaint/fingerpaint.py
# because I want to run it within a Python script

import _tkinter
import argparse
import contextlib
import os
import pkg_resources
import subprocess as sp
import sys
import tkinter
import tkinter.font
from io import BytesIO

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps, PIL.ImageChops
import evdev
import pyudev
import numpy as np

from multilayer_perceptron import model

DEFAULT_WIDTH = 600
AA_FACTOR = 4  # Anti-aliasing
OUTPUT_SCALE = 2


@contextlib.contextmanager
def lock_pointer_x11(devname):
    sp.call(['xinput', 'disable', devname])
    try:
        yield
    finally:
        sp.call(['xinput', 'enable', devname])


@contextlib.contextmanager
def lock_pointer_wayland():
    prev_value = sp.check_output(['gsettings', 'get', 'org.gnome.desktop.peripherals.touchpad', 'send-events']).strip()

    # Fix for arch based distros
    if prev_value == '':
        prev_value = "'enabled'"

    if prev_value not in (b"'enabled'", b"'disabled'", b"'disabled-on-external-mouse'"):
        print(f'Unexpected touchpad state: "{prev_value.decode()}", are you using Gnome?', file=sys.stderr)
        exit(1)

    sp.call(['dconf', 'write', '/org/gnome/desktop/peripherals/touchpad/send-events', "'disabled'"])
    try:
        yield
    finally:
        sp.call(['dconf', 'write', '/org/gnome/desktop/peripherals/touchpad/send-events', prev_value])


def make_ui(events, image_size, devname):
    top = tkinter.Tk()

    top.resizable(False, False)
    window_size = image_size

    top.title("Draw a digit")

    def exit_handler(_):
        top.destroy()

    top.bind('<Key>', exit_handler)
    top.bind('<Button>', exit_handler)


    hf = "Ubuntu"
    hs = 16
    hfw = "bold"
    hint_font = tkinter.font.Font(family=hf, size=hs, weight=hfw)

    bckg = '#eeeeee'
    hc = '#aaaaaa'
    h = 'Press any key or click to finish drawing'
    canvas = tkinter.Canvas(top, bg=bckg, height=window_size[1], width=window_size[0], borderwidth=0, highlightthickness=0)
    canvas.create_text(
        (window_size[0] / 2, window_size[1] * 9 / 10), fill=hc, font=hint_font, justify=tkinter.CENTER,
        text=h
    )
    aa_factor = AA_FACTOR * OUTPUT_SCALE
    image = PIL.Image.new("RGBA", (image_size[0] * aa_factor, image_size[1] * aa_factor), (0, 0, 0, 0))
    image_canvas = PIL.ImageDraw.Draw(image)

    lt = 6
    lc = '#000000'

    canvas.pack(fill=tkinter.BOTH, expand=True)
    try:
        if os.environ['XDG_SESSION_TYPE'] == 'wayland':
            lock_pointer = lock_pointer_wayland()
        else:
            lock_pointer = lock_pointer_x11(devname)

        with lock_pointer:
            while True:
                lines = next(events)
                for line in lines:
                    screen_projected_start = (line[0][0] * window_size[0], line[0][1] * window_size[1])
                    screen_projected_end = (line[1][0] * window_size[0], line[1][1] * window_size[1])
                    image_projected_start = (line[0][0] * image_size[0], line[0][1] * image_size[1])
                    image_projected_end = (line[1][0] * image_size[0], line[1][1] * image_size[1])
                    canvas.create_line(
                        screen_projected_start, screen_projected_end,
                        width=lt, capstyle=tkinter.ROUND, fill=lc
                    )
                    image_canvas.line(
                        ((int(image_projected_start[0] * aa_factor), int(image_projected_start[1] * aa_factor)),
                         (int(image_projected_end[0] * aa_factor), int(image_projected_end[1] * aa_factor))),
                        width=int(lt * aa_factor), joint='curve', fill=(0, 0, 0)
                    )
                    offset = (lt * aa_factor - 1) / 2
                    image_canvas.ellipse(
                        (int(image_projected_start[0] * aa_factor - offset), int(image_projected_start[1] * aa_factor - offset),
                         int(image_projected_start[0] * aa_factor + offset), int(image_projected_start[1] * aa_factor + offset)),
                        fill=(0, 0, 0)
                    )
                    image_canvas.ellipse(
                        (int(image_projected_end[0] * aa_factor - offset), int(image_projected_end[1] * aa_factor - offset),
                         int(image_projected_end[0] * aa_factor + offset), int(image_projected_end[1] * aa_factor + offset)),
                        fill=(0, 0, 0)
                    )

                top.update_idletasks()
                top.update()
    except (KeyboardInterrupt, _tkinter.TclError):
        del events

        image = image.resize((image_size[0] * OUTPUT_SCALE, image_size[1] * OUTPUT_SCALE), resample=PIL.Image.LANCZOS)

        image = replace_transparent_background(image)
        image = trim_borders(image)
        image = pad_image(image)
        image = to_grayscale(image)
        image = invert_colors(image)
        image = resize_image(image)

        # image.show()

        l = list(image.getdata())
        test_data = np.array([1.0 * n for n in l])
        test_data = test_data.reshape(28, 28)




        one_prediction = model.predict(np.array([test_data]))

        print(one_prediction)


        exit(0)

# Shamelessly taken from https://www.toptal.com/data-science/machine-learning-number-recognition
def replace_transparent_background(image):
    image_arr = np.array(image)

    if len(image_arr.shape) == 2:
        return image

    alpha1 = 0
    r2, g2, b2, alpha2 = 255, 255, 255, 255

    red, green, blue, alpha = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2], image_arr[:, :, 3]
    mask = (alpha == alpha1)
    image_arr[:, :, :4][mask] = [r2, g2, b2, alpha2]

    return PIL.Image.fromarray(image_arr)

def trim_borders(image):
    bg = PIL.Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = PIL.ImageChops.difference(image, bg)
    diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)

    return image

def pad_image(image):
    return PIL.ImageOps.expand(image, border=30, fill='#fff')

def to_grayscale(image):
    return image.convert('L')

def invert_colors(image):
    return PIL.ImageOps.invert(image)

def resize_image(image):
    return image.resize((28, 28))



def get_touchpads(udev):
    for device in udev.list_devices(ID_INPUT_TOUCHPAD='1'):
        if device.device_node is not None and device.device_node.rpartition('/')[2].startswith('event'):
            yield device


def get_device_name(dev):
    while dev is not None:
        name = dev.properties.get('NAME')
        if name:
            return name
        else:
            dev = next(dev.ancestors, None)


def permission_error():
    print('Failed to access touchpad!', file=sys.stderr)
    if sys.stdin.isatty():
        print('Touchpad access is currently restricted. Would you like to unrestrict it?', file=sys.stderr)
        response = input('[Yes]/no: ')
        if response.lower() in ('y', 'ye', 'yes', 'ok', 'sure', ''):
            sp.call(['pkexec', pkg_resources.resource_filename('fingerpaint', 'data/fix_permissions.sh')])
        else:
            print('Canceled.', file=sys.stderr)

    exit(1)


def get_touchpad(udev):
    for device in get_touchpads(udev):
        dev_name = get_device_name(device).strip('"')
        print('Using touchpad:', dev_name, file=sys.stderr)
        try:
            return evdev.InputDevice(device.device_node), dev_name
        except PermissionError:
            permission_error()
    return None, None


def main():
    udev = pyudev.Context()
    touchpad, devname = get_touchpad(udev)
    if touchpad is None:
        print('No touchpad found', file=sys.stderr)
        exit(1)
    x_absinfo = touchpad.absinfo(evdev.ecodes.ABS_X)
    y_absinfo = touchpad.absinfo(evdev.ecodes.ABS_Y)
    val_range = (x_absinfo.max - x_absinfo.min, y_absinfo.max - y_absinfo.min)

    def handler_loop():
        last_pos = (-1, -1)
        curr_pos = (-1, -1)
        wip_pos = (-1, -1)
        while True:
            event = touchpad.read_one()
            if event:
                if event.type == evdev.ecodes.EV_ABS:
                    if event.code == evdev.ecodes.ABS_X:
                        wip_pos = ((event.value - x_absinfo.min) / (x_absinfo.max - x_absinfo.min), wip_pos[1])
                    if event.code == evdev.ecodes.ABS_Y:
                        wip_pos = (wip_pos[0], (event.value - y_absinfo.min) / (y_absinfo.max - y_absinfo.min))
                if event.type == evdev.ecodes.EV_KEY:
                    if event.code == evdev.ecodes.BTN_TOUCH and event.value == 0:
                        wip_pos = (-1, -1)
                    if (event.code == evdev.ecodes.BTN_LEFT or event.code == evdev.ecodes.BTN_RIGHT) \
                            and event.value == 1:
                        raise KeyboardInterrupt()
                if event.type == evdev.ecodes.EV_SYN:
                    curr_pos = wip_pos

            if last_pos != curr_pos:
                if (last_pos[0] == -1 or last_pos[1] == -1) and curr_pos[0] != -1 and curr_pos[1] != -1:
                    # Work with light taps
                    last_pos = curr_pos
                if last_pos[0] != -1 and last_pos[1] != -1 and curr_pos[0] != -1 and curr_pos[1] != -1:
                    yield [(last_pos, curr_pos)]
                else:
                    yield []
                last_pos = curr_pos
            else:
                yield []

    scaled = (DEFAULT_WIDTH, int(DEFAULT_WIDTH / val_range[0] * val_range[1]))
    make_ui(handler_loop(), scaled, devname)
    del touchpad



main()


